"""
AGIBOT G1 Isaac Lab 双机对比仿真运行器
======================================
功能:
1. 在同一场景中生成两台 G1 机器人
2. Robot_Obs (参考机): 运动学模式，直接覆写关节状态
3. Robot_Act (测试机): 物理驱动模式，使用 PD 控制
4. 支持 30Hz 数据源与 60Hz 仿真的对齐 (Decimation = 2)

技术要点:
- Robot_Obs: 使用 write_joint_state_to_sim() 直接设置关节位置
- Robot_Act: 使用 set_joint_position_target() 进行 PD 控制
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import numpy as np

from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf, UsdPhysics
import omni.usd

@dataclass
class ActuatorConfig:
    """执行器配置"""
    head_stiffness: float = 100.0
    head_damping: float = 10.0
    arm_stiffness: float = 400.0
    arm_damping: float = 40.0
    gripper_stiffness: float = 100.0
    gripper_damping: float = 10.0


@dataclass
class DualSimConfig:
    """双机仿真配置"""
    physics_dt: float = 1.0 / 90.0      # 物理频率 90Hz
    control_dt: float = 1.0 / 30.0      # 控制频率 30Hz (源数据)

    urdf_path: str = ""                 # Robot_Act 使用的 URDF (物理驱动)
    urdf_path_obs: str = ""             # Robot_Obs 使用的 URDF (运动学模式，无碰撞)
    fix_base: bool = True

    # 机器人位置
    robot_obs_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_act_pos: Tuple[float, float, float] = (1, 0.0, 0.0)

    # 执行器配置
    actuator: ActuatorConfig = field(default_factory=ActuatorConfig)

    # 调试
    print_every_n_frames: int = 100

    @property
    def decimation(self) -> int:
        """每个控制步的物理步数"""
        return max(1, int(round(self.control_dt / self.physics_dt)))


@dataclass
class DualSimRecord:
    """双机仿真记录 (单帧)"""
    sim_time: float
    frame_index: int

    # Robot_Obs (参考机) - 运动学模式
    obs_joint_pos: np.ndarray       # 16维
    obs_joint_vel: np.ndarray       # 16维

    # Robot_Act (测试机) - 物理驱动模式
    act_joint_pos: np.ndarray       # 16维
    act_joint_vel: np.ndarray       # 16维
    act_joint_torque: np.ndarray    # 16维

    # 目标值
    real_joint_pos: np.ndarray      # 16维 - 真实关节位置 (给 Robot_Obs)
    target_joint_pos: np.ndarray    # 16维 - 目标关节位置 (给 Robot_Act)

    # 夹爪
    obs_gripper_l: float = 0.0
    obs_gripper_r: float = 0.0
    act_gripper_l: float = 0.0
    act_gripper_r: float = 0.0
    target_gripper_l: float = 0.0
    target_gripper_r: float = 0.0


class IsaacLabDualSimRunner:
    """
    Isaac Lab 双机仿真运行器

    用法:
        runner = IsaacLabDualSimRunner(config)
        runner.initialize()
        records = runner.run_dual_replay(real_positions, target_positions, ...)
        runner.close()
    """

    def __init__(self, config: DualSimConfig):
        self.config = config

        # Isaac Lab 组件 (延迟初始化)
        self.sim = None
        self.robot_obs = None  # 参考机 (运动学模式)
        self.robot_act = None  # 测试机 (物理驱动模式)
        self.simulation_app = None

        # 关节名称映射
        self.robot_joint_names: List[str] = []
        self.control_joint_indices: Dict[str, int] = {}

        # 夹爪常量
        self.GRIPPER_STATE_MAX = 120.0
        self.GRIPPER_URDF_MAX = 0.785

    
    def initialize(self):
        """初始化仿真环境"""
        import isaaclab.sim as sim_utils
        from isaaclab.sim import SimulationContext, SimulationCfg, PhysxCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg

        # 创建仿真上下文
        sim_cfg = SimulationCfg(
            dt=self.config.physics_dt,
            render_interval=self.config.decimation,
            physx=PhysxCfg(
                enable_stabilization=True,
            ),
        )
        self.sim = SimulationContext(sim_cfg)
        self.sim.set_camera_view(eye=[2.5, 2.5, 2.0], target=[0.25, 0.0, 0.8])

        # 创建场景
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/GroundPlane", ground_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/Light", light_cfg)

        # 创建两台机器人 (使用 URDF，make_instanceable=False 创建独立副本)
        # Robot_Obs: 使用无碰撞 URDF (运动学模式)
        obs_urdf = self.config.urdf_path_obs if self.config.urdf_path_obs else self.config.urdf_path
        robot_obs_cfg = self._create_robot_cfg(
            sim_utils, ArticulationCfg, ImplicitActuatorCfg,
            prim_path="/World/Robot_Obs",
            position=self.config.robot_obs_pos,
            urdf_path=obs_urdf
        )

        # Robot_Act: 使用物理驱动 URDF
        robot_act_cfg = self._create_robot_cfg(
            sim_utils, ArticulationCfg, ImplicitActuatorCfg,
            prim_path="/World/Robot_Act",
            position=self.config.robot_act_pos,
            urdf_path=self.config.urdf_path
        )

        self.robot_obs = Articulation(robot_obs_cfg)
        self.robot_act = Articulation(robot_act_cfg)

        # 重置仿真
        self.sim.reset()
        self.robot_obs.reset()
        self.robot_act.reset()

        # 设置 Robot_Act 为半透明
        self._set_robot_ghost("/World/Robot_Act", color=(1.0, 0.3, 0.3))

        self.robot_joint_names = list(self.robot_obs.joint_names)
        self._build_joint_mapping()

        print(f"[DualSimRunner] 仿真初始化完成")
        print(f"[DualSimRunner] 物理频率: {1.0/self.config.physics_dt:.1f} Hz")
        print(f"[DualSimRunner] 控制频率: {1.0/self.config.control_dt:.1f} Hz")
        print(f"[DualSimRunner] Decimation: {self.config.decimation}")
        print(f"[DualSimRunner] 关节数量: {len(self.robot_joint_names)}")
        print(f"[DualSimRunner] Robot_Obs 位置: {self.config.robot_obs_pos}")
        print(f"[DualSimRunner] Robot_Act 位置: {self.config.robot_act_pos}")
    
    def _create_robot_cfg(self, sim_utils, ArticulationCfg, ImplicitActuatorCfg,
                          prim_path: str, position: Tuple[float, float, float],
                          urdf_path: str):
        """创建机器人配置 (非实例化，独立副本)"""
        urdf_cfg = sim_utils.UrdfFileCfg(
            asset_path=str(urdf_path),
            fix_base=self.config.fix_base,
            force_usd_conversion=True,
            make_instanceable=False,  # 创建独立副本，可自由修改材质/Mesh
        )

        # 清除 URDF 默认驱动
        if hasattr(urdf_cfg, 'joint_drive'):
            jd = urdf_cfg.joint_drive
            if hasattr(jd, 'gains'):
                jd.gains.stiffness = 0.0
                jd.gains.damping = 0.0
            if hasattr(jd, 'drive_type'):
                jd.drive_type = "force"

        act = self.config.actuator

        return ArticulationCfg(
            prim_path=prim_path,
            spawn=urdf_cfg,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=position,
                joint_pos={".*": 0.0},
            ),
            actuators={
                "head": ImplicitActuatorCfg(
                    joint_names_expr=["idx1.*_head_joint.*"],
                    stiffness=act.head_stiffness,
                    damping=act.head_damping,
                ),
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=["idx.*_arm_[lr]_joint.*"],
                    stiffness=act.arm_stiffness,
                    damping=act.arm_damping,
                ),
                "grippers": ImplicitActuatorCfg(
                    joint_names_expr=["idx.*_gripper_[lr]_.*"],
                    stiffness=act.gripper_stiffness,
                    damping=act.gripper_damping,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )

    def _build_joint_mapping(self):
        """构建关节名称到索引的映射"""
        self.control_joint_indices = {
            name: i for i, name in enumerate(self.robot_joint_names)
        }

    def _set_robot_ghost(self, prim_path: str, color=(1.0, 0.3, 0.3)):
        """
        设置机器人颜色 (针对 arm 和 gripper)
        """
        from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
        import omni.usd

        print(f"\n{'='*60}")
        print(f"[Ghost] 开始处理: {prim_path}")
        print(f"{'='*60}")

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(prim_path)

        if not root.IsValid():
            print(f"[Ghost] 错误: 找不到 {prim_path}")
            return

        # 创建红色材质
        mat_path = "/World/RedMaterial"
        mat_prim = stage.GetPrimAtPath(mat_path)

        if not mat_prim.IsValid():
            mat = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            ghost_mat = mat
            print(f"[Ghost] 红色材质创建成功")
        else:
            ghost_mat = UsdShade.Material(mat_prim)

        # 定义需要处理的 link 名称
        arm_link_names = [
            "arm_l_link1", "arm_l_link2", "arm_l_link3",
            "arm_l_link4", "arm_l_link5", "arm_l_link6",
            "arm_r_link1", "arm_r_link2", "arm_r_link3",
            "arm_r_link4", "arm_r_link5", "arm_r_link6",
        ]
        gripper_link_names = [
            "gripper_l_inner_link1", "gripper_l_outer_link2", "gripper_l_inner_link2",
            "gripper_l_outer_link3", "gripper_l_inner_link3", "gripper_l_outer_link4",
            "gripper_l_inner_link4", "gripper_r_inner_link1", "gripper_r_outer_link2",
            "gripper_r_inner_link2", "gripper_r_outer_link3", "gripper_r_inner_link3",
            "gripper_r_outer_link4", "gripper_r_inner_link4",
        ]

        bound_count = 0

        # 处理 arm links - visuals 路径: {prim_path}/{name}/visuals/mesh_0/cylinder
        print(f"\n[Ghost] 处理 arm links...")
        for name in arm_link_names:
            # 尝试多种可能的 visuals 路径
            possible_paths = [
                f"{prim_path}/{name}/visuals/mesh_0/cylinder",
                f"{prim_path}/{name}/visuals/mesh_0",
                f"{prim_path}/{name}/visuals",
            ]
            for path in possible_paths:
                prim = stage.GetPrimAtPath(path)
                if prim.IsValid():
                    try:
                        binding = UsdShade.MaterialBindingAPI(prim)
                        binding.UnbindAllBindings()
                        binding.Bind(ghost_mat, UsdShade.Tokens.strongerThanDescendants)
                        bound_count += 1
                        print(f"  [OK] {name} @ {path}")
                    except Exception as e:
                        print(f"  [ERR] {name}: {e}")
                    break
            else:
                print(f"  [MISS] {name} - 未找到 visuals")

        # 处理 gripper links
        print(f"\n[Ghost] 处理 gripper links...")
        for name in gripper_link_names:
            # gripper 的 visuals 路径可能是: {prim_path}/{name}/visuals/{suffix}/Scene/mesh
            suffix = name.split('gripper_')[1] if 'gripper_' in name else name
            possible_paths = [
                f"{prim_path}/{name}/visuals/{suffix}/Scene/mesh",
                f"{prim_path}/{name}/visuals/{suffix}/Scene",
                f"{prim_path}/{name}/visuals/{suffix}",
                f"{prim_path}/{name}/visuals",
            ]
            for path in possible_paths:
                prim = stage.GetPrimAtPath(path)
                if prim.IsValid():
                    try:
                        binding = UsdShade.MaterialBindingAPI(prim)
                        binding.UnbindAllBindings()
                        binding.Bind(ghost_mat, UsdShade.Tokens.strongerThanDescendants)
                        bound_count += 1
                        print(f"  [OK] {name} @ {path}")
                    except Exception as e:
                        print(f"  [ERR] {name}: {e}")
                    break
            else:
                print(f"  [MISS] {name} - 未找到 visuals")

        print(f"\n{'='*60}")
        print(f"[Ghost] 处理完成! 成功绑定: {bound_count}")
        print(f"{'='*60}\n")

    def _convert_gripper_value(self, state_value: float) -> float:
        """夹爪值转换: [0,120] -> [0.785, 0]"""
        normalized = np.clip(state_value / self.GRIPPER_STATE_MAX, 0.0, 1.0)
        return self.GRIPPER_URDF_MAX * (1.0 - normalized)

    def _map_targets_to_sim(
        self,
        joint_targets: np.ndarray,
        gripper_l: float,
        gripper_r: float
    ) -> np.ndarray:
        """将 16 维关节目标映射到仿真关节数组"""
        result = np.zeros(len(self.robot_joint_names), dtype=np.float32)

        joint_names_16d = [
            "idx11_head_joint1", "idx12_head_joint2",
            "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
            "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6",
            "idx27_arm_l_joint7",
            "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
            "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6",
            "idx67_arm_r_joint7",
        ]

        for i, name in enumerate(joint_names_16d):
            if name in self.control_joint_indices:
                result[self.control_joint_indices[name]] = joint_targets[i]

        # 夹爪映射
        gripper_l_urdf = self._convert_gripper_value(gripper_l)
        gripper_r_urdf = self._convert_gripper_value(gripper_r)

        gripper_mapping = {
            "idx41_gripper_l_outer_joint1": gripper_l_urdf,
            "idx81_gripper_r_outer_joint1": gripper_r_urdf,
        }

        mimic_mapping = {
            "idx31_gripper_l_inner_joint1": ("idx41_gripper_l_outer_joint1", -1.0),
            "idx49_gripper_l_outer_joint2": ("idx41_gripper_l_outer_joint1", 1.5),
            "idx39_gripper_l_inner_joint2": ("idx41_gripper_l_outer_joint1", -1.5),
            "idx42_gripper_l_outer_joint3": ("idx41_gripper_l_outer_joint1", 0.0),
            "idx32_gripper_l_inner_joint3": ("idx41_gripper_l_outer_joint1", 0.0),
            "idx43_gripper_l_outer_joint4": ("idx41_gripper_l_outer_joint1", 0.0),
            "idx33_gripper_l_inner_joint4": ("idx41_gripper_l_outer_joint1", 0.0),
            "idx71_gripper_r_inner_joint1": ("idx81_gripper_r_outer_joint1", -1.0),
            "idx89_gripper_r_outer_joint2": ("idx81_gripper_r_outer_joint1", 1.5),
            "idx79_gripper_r_inner_joint2": ("idx81_gripper_r_outer_joint1", -1.5),
            "idx82_gripper_r_outer_joint3": ("idx81_gripper_r_outer_joint1", 0.0),
            "idx72_gripper_r_inner_joint3": ("idx81_gripper_r_outer_joint1", 0.0),
            "idx83_gripper_r_outer_joint4": ("idx81_gripper_r_outer_joint1", 0.0),
            "idx73_gripper_r_inner_joint4": ("idx81_gripper_r_outer_joint1", 0.0),
        }

        for name, value in gripper_mapping.items():
            if name in self.control_joint_indices:
                result[self.control_joint_indices[name]] = value

        for name, (master, mult) in mimic_mapping.items():
            if name in self.control_joint_indices and master in gripper_mapping:
                result[self.control_joint_indices[name]] = gripper_mapping[master] * mult

        return result

    def _extract_16d_from_sim(self, full_array: np.ndarray) -> np.ndarray:
        """从仿真关节数组提取 16 维数据"""
        joint_names_16d = [
            "idx11_head_joint1", "idx12_head_joint2",
            "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
            "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6",
            "idx27_arm_l_joint7",
            "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
            "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6",
            "idx67_arm_r_joint7",
        ]

        result = np.zeros(16, dtype=np.float32)
        for i, name in enumerate(joint_names_16d):
            if name in self.control_joint_indices:
                result[i] = full_array[self.control_joint_indices[name]]

        return result

    def run_dual_replay(
        self,
        real_positions: np.ndarray,
        target_positions: np.ndarray,
        real_grippers_l: np.ndarray,
        real_grippers_r: np.ndarray,
        target_grippers_l: np.ndarray,
        target_grippers_r: np.ndarray,
        timestamps: np.ndarray,
        simulation_app,
    ) -> List[DualSimRecord]:
        """
        执行双机对比回放 (带插值)

        Args:
            real_positions: (N, 16) 真实关节位置 (给 Robot_Obs)
            target_positions: (N, 16) 目标关节位置 (给 Robot_Act)
            real_grippers_l/r: (N,) 真实夹爪值
            target_grippers_l/r: (N,) 目标夹爪值
            timestamps: (N,) 时间戳
            simulation_app: IsaacSim 应用实例

        Returns:
            List[DualSimRecord]: 仿真记录
        """
        import torch

        n_frames = len(timestamps)
        records = []
        decimation = self.config.decimation  # 90Hz/30Hz = 3

        print(f"[DualSimRunner] 开始双机回放 {n_frames} 帧...")
        print(f"[DualSimRunner] 物理频率: {1.0/self.config.physics_dt:.0f}Hz, 数据频率: 30Hz")
        print(f"[DualSimRunner] Decimation: {decimation} (每帧数据执行{decimation}个物理步)")
        print(f"[DualSimRunner] Robot_Obs: 运动学模式 (插值平滑)")
        print(f"[DualSimRunner] Robot_Act: 物理驱动模式 (PD控制)")

        print("[DualSimRunner] 正在对齐初始姿态并预热物理引擎...")
                
                # 1. 获取第0帧数据
        start_pos_obs = real_positions[0]
        start_pos_act = target_positions[0] # Act 应该从 Action 的第一个目标开始，或者从 Real 的第一个位置开始？
                                            # 通常为了对比 Sim2Real，Act 应该从 Real 的初始位置开始，然后接收 Action
                                            # 这里我们让 Act 从 Real[0] 开始，以保证起点一致
        
        start_gripper_l = real_grippers_l[0]
        start_gripper_r = real_grippers_r[0]

        # 2. 映射到 Sim 格式
        # 注意：这里让 Robot_Act 也从 Real Position 开始初始化，确保起点完全重合
        sim_start_state = self._map_targets_to_sim(start_pos_obs, start_gripper_l, start_gripper_r)
        
        start_tensor = torch.tensor(
            sim_start_state, dtype=torch.float32, device=self.sim.device
        ).unsqueeze(0)
        zero_vel = torch.zeros_like(start_tensor)

        # 3. 强制写入状态 (Teleport)
        # Robot_Obs
        try:
            self.robot_obs.write_joint_state_to_sim(start_tensor, zero_vel)
        except TypeError:
            self.robot_obs.data.joint_pos[:] = start_tensor
            self.robot_obs.data.joint_vel[:] = zero_vel
            self.robot_obs.write_data_to_sim()
            
        # Robot_Act (关键步骤: 把物理机也瞬移到起点，消除 T-Pose)
        try:
            self.robot_act.write_joint_state_to_sim(start_tensor, zero_vel)
        except TypeError:
            self.robot_act.data.joint_pos[:] = start_tensor
            self.robot_act.data.joint_vel[:] = zero_vel
            self.robot_act.write_data_to_sim()

        # 4. 设置初始 PD 目标并预热 (Warmup)
        # 让物理引擎在原地稳定 20 帧，消除重力/碰撞导致的初始微动
        self.robot_act.set_joint_position_target(start_tensor)
        self.robot_act.write_data_to_sim()
        
        for _ in range(20):
            self.sim.step()
            self.robot_obs.write_joint_state_to_sim(start_tensor, zero_vel) # Obs 保持不动
            self.robot_act.update(self.config.physics_dt)
            
        print("[DualSimRunner] 预热完成，开始回放循环")

        for frame_idx in range(n_frames):
            if not simulation_app.is_running():
                print("[DualSimRunner] 仿真被中断")
                break

            # 获取当前帧和下一帧数据 (用于插值)
            real_pos_curr = real_positions[frame_idx]
            target_pos = target_positions[frame_idx]
            real_gripper_l_curr = real_grippers_l[frame_idx]
            real_gripper_r_curr = real_grippers_r[frame_idx]
            target_gripper_l = target_grippers_l[frame_idx]
            target_gripper_r = target_grippers_r[frame_idx]

            # 下一帧数据 (用于插值，最后一帧使用当前帧)
            next_idx = min(frame_idx + 1, n_frames - 1)
            real_pos_next = real_positions[next_idx]
            real_gripper_l_next = real_grippers_l[next_idx]
            real_gripper_r_next = real_grippers_r[next_idx]

            # Robot_Act 目标 (整个控制周期保持不变)
            act_sim_targets = self._map_targets_to_sim(target_pos, target_gripper_l, target_gripper_r)
            act_tensor = torch.tensor(
                act_sim_targets, dtype=torch.float32, device=self.sim.device
            ).unsqueeze(0)

            # ========== 执行 decimation 个物理步骤，每步插值 ==========
            for sub_step in range(decimation):
                # 计算插值因子 alpha: 0 -> 1
                alpha = sub_step / decimation

                # 线性插值 Robot_Obs 的位置
                interp_pos = real_pos_curr * (1.0 - alpha) + real_pos_next * alpha
                interp_gripper_l = real_gripper_l_curr * (1.0 - alpha) + real_gripper_l_next * alpha
                interp_gripper_r = real_gripper_r_curr * (1.0 - alpha) + real_gripper_r_next * alpha

                # 映射到仿真关节数组
                obs_sim_targets = self._map_targets_to_sim(interp_pos, interp_gripper_l, interp_gripper_r)
                obs_tensor = torch.tensor(
                    obs_sim_targets, dtype=torch.float32, device=self.sim.device
                ).unsqueeze(0)
                zero_vel = torch.zeros_like(obs_tensor)

                # Robot_Obs: 直接覆写关节状态
                try:
                    self.robot_obs.write_joint_state_to_sim(obs_tensor, zero_vel)
                except TypeError:
                    self.robot_obs.data.joint_pos[:] = obs_tensor
                    self.robot_obs.data.joint_vel[:] = zero_vel

                # Robot_Act: 设置 PD 控制目标
                self.robot_act.set_joint_position_target(act_tensor)
                self.robot_act.write_data_to_sim()

                # 执行一个物理步
                self.sim.step()

                # 更新机器人状态
                self.robot_obs.update(self.config.physics_dt)
                self.robot_act.update(self.config.physics_dt)

            # ========== 读取状态 ==========
            # Robot_Obs
            obs_pos_full = self.robot_obs.data.joint_pos[0].cpu().numpy()
            obs_vel_full = self.robot_obs.data.joint_vel[0].cpu().numpy()

            # Robot_Act
            act_pos_full = self.robot_act.data.joint_pos[0].cpu().numpy()
            act_vel_full = self.robot_act.data.joint_vel[0].cpu().numpy()

            # 尝试获取力矩
            if hasattr(self.robot_act.data, 'applied_torque'):
                act_torque_full = self.robot_act.data.applied_torque[0].cpu().numpy()
            elif hasattr(self.robot_act.data, 'joint_effort'):
                act_torque_full = self.robot_act.data.joint_effort[0].cpu().numpy()
            else:
                act_torque_full = np.zeros_like(act_pos_full)

            # 提取 16 维数据
            obs_pos_16d = self._extract_16d_from_sim(obs_pos_full)
            obs_vel_16d = self._extract_16d_from_sim(obs_vel_full)
            act_pos_16d = self._extract_16d_from_sim(act_pos_full)
            act_vel_16d = self._extract_16d_from_sim(act_vel_full)
            act_torque_16d = self._extract_16d_from_sim(act_torque_full)

            # 提取夹爪位置
            obs_gripper_l_idx = self.control_joint_indices.get("idx41_gripper_l_outer_joint1", 0)
            obs_gripper_r_idx = self.control_joint_indices.get("idx81_gripper_r_outer_joint1", 0)
            obs_gripper_l_urdf = obs_pos_full[obs_gripper_l_idx]
            obs_gripper_r_urdf = obs_pos_full[obs_gripper_r_idx]
            act_gripper_l_urdf = act_pos_full[obs_gripper_l_idx]
            act_gripper_r_urdf = act_pos_full[obs_gripper_r_idx]

            # 转换回 [0,120] 范围
            obs_gripper_l_val = (1.0 - obs_gripper_l_urdf / self.GRIPPER_URDF_MAX) * self.GRIPPER_STATE_MAX
            obs_gripper_r_val = (1.0 - obs_gripper_r_urdf / self.GRIPPER_URDF_MAX) * self.GRIPPER_STATE_MAX
            act_gripper_l_val = (1.0 - act_gripper_l_urdf / self.GRIPPER_URDF_MAX) * self.GRIPPER_STATE_MAX
            act_gripper_r_val = (1.0 - act_gripper_r_urdf / self.GRIPPER_URDF_MAX) * self.GRIPPER_STATE_MAX

            # 创建记录
            record = DualSimRecord(
                sim_time=timestamps[frame_idx],
                frame_index=frame_idx,
                obs_joint_pos=obs_pos_16d.copy(),
                obs_joint_vel=obs_vel_16d.copy(),
                act_joint_pos=act_pos_16d.copy(),
                act_joint_vel=act_vel_16d.copy(),
                act_joint_torque=act_torque_16d.copy(),
                real_joint_pos=real_pos_curr.copy(),
                target_joint_pos=target_pos.copy(),
                obs_gripper_l=obs_gripper_l_val,
                obs_gripper_r=obs_gripper_r_val,
                act_gripper_l=act_gripper_l_val,
                act_gripper_r=act_gripper_r_val,
                target_gripper_l=target_gripper_l,
                target_gripper_r=target_gripper_r,
            )
            records.append(record)

            # 进度输出
            if frame_idx % 60 == 0:
                progress = frame_idx / n_frames * 100
                # 计算误差
                pos_error = np.abs(act_pos_16d - real_pos_curr).mean()
                print(f"\n[Frame {frame_idx}] 进度: {progress:.1f}%")
                print(f"  Robot_Act vs Real 平均误差: {np.rad2deg(pos_error):.4f} deg")

        print(f"\n[DualSimRunner] 回放完成，共 {len(records)} 帧")
        return records

    def close(self):
        """关闭仿真"""
        if self.sim is not None:
            self.sim = None
            self.robot_obs = None
            self.robot_act = None


def dual_records_to_arrays(records: List[DualSimRecord]) -> Dict[str, np.ndarray]:
    """将双机仿真记录转换为数组"""
    return {
        "sim_time": np.array([r.sim_time for r in records]),
        "frame_index": np.array([r.frame_index for r in records]),
        # Robot_Obs
        "obs_joint_pos": np.stack([r.obs_joint_pos for r in records]),
        "obs_joint_vel": np.stack([r.obs_joint_vel for r in records]),
        # Robot_Act
        "act_joint_pos": np.stack([r.act_joint_pos for r in records]),
        "act_joint_vel": np.stack([r.act_joint_vel for r in records]),
        "act_joint_torque": np.stack([r.act_joint_torque for r in records]),
        # 原始数据
        "real_joint_pos": np.stack([r.real_joint_pos for r in records]),
        "target_joint_pos": np.stack([r.target_joint_pos for r in records]),
        # 夹爪
        "obs_gripper_l": np.array([r.obs_gripper_l for r in records]),
        "obs_gripper_r": np.array([r.obs_gripper_r for r in records]),
        "act_gripper_l": np.array([r.act_gripper_l for r in records]),
        "act_gripper_r": np.array([r.act_gripper_r for r in records]),
        "target_gripper_l": np.array([r.target_gripper_l for r in records]),
        "target_gripper_r": np.array([r.target_gripper_r for r in records]),
    }
