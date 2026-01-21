"""
AGIBOT G1 Isaac Lab 仿真运行器
==============================
功能:
1. 初始化 Isaac Lab 仿真环境
2. 配置 Implicit Actuator (支持自定义 Kp/Kd)
3. 执行回放循环并记录仿真数据
4. 支持降采样以匹配源数据频率

技术要点:
- 仿真频率: 通常 60Hz~400Hz (由 SimulationCfg.dt 控制)
- 控制频率: 30Hz (与源数据一致，通过 decimation 实现)
- 数据记录: 仅在控制步骤时记录，确保与源数据对齐
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ActuatorConfig:
    """
    执行器配置，这里是默认的，实际上看replay_config.yaml
    """
    # 腰部升降 (prismatic) - 单位: N/m, Ns/m
    # 注意: 需要高刚度支撑上半身重量（约30-40kg）
    waist_lift_stiffness: float = 1000000.0  # 1M N/m
    waist_lift_damping: float = 1000.0       # 1K Ns/m

    # 腰部旋转 (revolute) - 单位: Nm/rad, Nms/rad
    waist_rotate_stiffness: float = 800.0
    waist_rotate_damping: float = 80.0
    
    # 头部
    head_stiffness: float = 100.0
    head_damping: float = 10.0
    
    # 手臂
    arm_stiffness: float = 200.0
    arm_damping: float = 20.0
    
    # 夹爪
    gripper_stiffness: float = 50.0
    gripper_damping: float = 5.0


@dataclass
class SimConfig:
    """
    仿真配置
    
    关键参数:
    - physics_dt: 物理步进时间 (建议 1/60 ~ 1/400)
    - control_dt: 控制步进时间 (应与源数据 fps 一致)
    - decimation: 每个控制步骤的物理步骤数 = control_dt / physics_dt
    """
    physics_dt: float = 1.0 / 60.0      # 物理频率 60Hz
    control_dt: float = 1.0 / 30.0      # 控制频率 30Hz (与源数据一致)
    
    urdf_path: str = ""
    fix_base: bool = True
    
    # 执行器配置
    actuator: ActuatorConfig = field(default_factory=ActuatorConfig)
    
    # 调试
    print_every_n_frames: int = 100
    
    @property
    def decimation(self) -> int:
        """每个控制步的物理步数"""
        return max(1, int(round(self.control_dt / self.physics_dt)))
    
    @property
    def sim_frequency(self) -> float:
        """仿真频率 (Hz)"""
        return 1.0 / self.physics_dt
    
    @property
    def control_frequency(self) -> float:
        """控制频率 (Hz)"""
        return 1.0 / self.control_dt

@dataclass
class SimRecord:
    """仿真记录 (单帧)"""
    sim_time: float
    frame_index: int

    joint_pos: np.ndarray       # 关节位置 (16维: 头2+臂14)
    joint_vel: np.ndarray       # 关节速度 (16维)
    joint_torque: np.ndarray    # 关节力矩 (16维)

    target_pos: np.ndarray      # 目标位置 (16维)

    # 夹爪数据
    gripper_l: float = 0.0      # 左夹爪位置 [0,120]
    gripper_r: float = 0.0      # 右夹爪位置 [0,120]
    target_gripper_l: float = 0.0  # 左夹爪目标
    target_gripper_r: float = 0.0  # 右夹爪目标

    left_ee_pos: Optional[np.ndarray] = None  # 左末端执行器位置 (x,y,z)
    right_ee_pos: Optional[np.ndarray] = None # 右末端执行器位置 (x,y,z)

class IsaacLabSimRunner:
    """
    Isaac Lab 仿真运行器
    
    用法:
        runner = IsaacLabSimRunner(config)
        runner.initialize()
        records = runner.run_replay(target_positions, timestamps)
        runner.close()
    """
    
    def __init__(self, config: SimConfig):
        self.config = config
        
        # Isaac Lab 组件 (延迟初始化)
        self.sim = None
        self.robot = None
        self.simulation_app = None
        
        # 关节名称映射
        self.robot_joint_names: List[str] = []
        self.control_joint_indices: Dict[str, int] = {}
        
        # 夹爪常量
        self.GRIPPER_STATE_MAX = 120.0
        self.GRIPPER_URDF_MAX = 0.785
    def get_joint_limits(self):
        """
        查询机器人所有关节的限制
        """
        if self.robot is None:
            print("[SimRunner] 机器人尚未初始化")
            return {}

        joint_limits = {}
        success = False

        try:
            # 方法1: 尝试从 soft_joint_pos_limits 读取
            if hasattr(self.robot.data, 'soft_joint_pos_limits') and self.robot.data.soft_joint_pos_limits is not None:
                joint_pos_limits = self.robot.data.soft_joint_pos_limits

                # 检查是否有有效数据（不全为0）
                if joint_pos_limits.shape[0] > 0:
                    has_valid_data = False
                    num_joints = min(len(self.robot.joint_names), joint_pos_limits.shape[0])

                    for i in range(num_joints):
                        try:
                            lower_limit = float(joint_pos_limits[i, 0].flatten()[0])
                            upper_limit = float(joint_pos_limits[i, 1].flatten()[0])

                            # 检查是否为有效限位（不是全0或异常值）
                            if abs(lower_limit) > 1e-6 or abs(upper_limit) > 1e-6:
                                has_valid_data = True

                            joint_limits[self.robot.joint_names[i]] = {
                                'lower': lower_limit,
                                'upper': upper_limit,
                                'range': upper_limit - lower_limit
                            }
                        except (IndexError, AttributeError, ValueError) as e:
                            print(f"[SimRunner] 获取关节 {self.robot.joint_names[i]} 限制时出错: {e}")

                    if has_valid_data:
                        success = True
                        print("[SimRunner] 成功从 soft_joint_pos_limits 读取关节限位")

            # 方法2: 尝试从 root_physx_view 读取
            if not success and hasattr(self.robot, 'root_physx_view'):
                print("[SimRunner] 尝试从 root_physx_view 读取关节限位...")
                try:
                    physx_view = self.robot.root_physx_view
                    if hasattr(physx_view, 'get_dof_limits'):
                        dof_limits = physx_view.get_dof_limits()
                        if dof_limits is not None and len(dof_limits) > 0:
                            for i, joint_name in enumerate(self.robot.joint_names):
                                if i < len(dof_limits):
                                    joint_limits[joint_name] = {
                                        'lower': float(dof_limits[i][0]),
                                        'upper': float(dof_limits[i][1]),
                                        'range': float(dof_limits[i][1] - dof_limits[i][0])
                                    }
                            success = True
                            print("[SimRunner] 成功从 root_physx_view 读取关节限位")
                except Exception as e:
                    print(f"[SimRunner] 从 root_physx_view 读取失败: {e}")

            # 方法3: 从URDF文件直接解析（备用方案）
            if not success:
                print("[SimRunner] 尝试从URDF文件直接解析关节限位...")
                joint_limits = self._parse_urdf_joint_limits()
                if joint_limits:
                    success = True
                    print("[SimRunner] 成功从URDF文件解析关节限位")

        except Exception as e:
            print(f"[SimRunner] 获取关节限制失败: {e}")
            import traceback
            traceback.print_exc()

        # 如果所有方法都失败，返回空字典并警告
        if not success or not joint_limits:
            print("[SimRunner] ⚠️  警告: 无法获取关节限位，所有方法均失败")
            return {}

        return joint_limits

    def _parse_urdf_joint_limits(self):
        """
        从URDF文件直接解析关节限位（备用方案）
        """
        import xml.etree.ElementTree as ET

        joint_limits = {}

        try:
            urdf_path = self.config.urdf_path
            if not urdf_path or not Path(urdf_path).exists():
                print(f"[SimRunner] URDF文件不存在: {urdf_path}")
                return {}

            # 解析URDF XML
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            # 遍历所有关节
            for joint in root.findall('joint'):
                joint_name = joint.get('name')
                joint_type = joint.get('type')

                # 查找limit标签
                limit = joint.find('limit')
                if limit is not None:
                    try:
                        lower = float(limit.get('lower', '0.0'))
                        upper = float(limit.get('upper', '0.0'))

                        joint_limits[joint_name] = {
                            'lower': lower,
                            'upper': upper,
                            'range': upper - lower,
                            'type': joint_type
                        }
                    except (ValueError, TypeError) as e:
                        print(f"[SimRunner] 解析关节 {joint_name} 限位失败: {e}")

            print(f"[SimRunner] 从URDF解析到 {len(joint_limits)} 个关节限位")

        except Exception as e:
            print(f"[SimRunner] 解析URDF文件失败: {e}")
            import traceback
            traceback.print_exc()

        return joint_limits

    def print_joint_limits(self, joint_name_pattern: str = ".*"):
        """
        打印符合模式的关节限制

        Args:
            joint_name_pattern: 用于匹配关节名称的字符串模式（部分匹配）
        """
        limits = self.get_joint_limits()

        if not limits:
            print("\n[Joint Limits] ⚠️  无法获取关节限位信息")
            return

        print(f"\n[Joint Limits] 关节限制信息 (共 {len(limits)} 个关节):")
        matched_any = False

        # 按关节类型分组显示
        waist_joints = []
        head_joints = []
        arm_joints = []
        gripper_joints = []
        other_joints = []

        for joint_name, limit_info in limits.items():
            if joint_name_pattern in joint_name or joint_name_pattern == ".*":
                matched_any = True

                if 'body_joint' in joint_name:
                    waist_joints.append((joint_name, limit_info))
                elif 'head_joint' in joint_name:
                    head_joints.append((joint_name, limit_info))
                elif 'arm' in joint_name and 'joint' in joint_name:
                    arm_joints.append((joint_name, limit_info))
                elif 'gripper' in joint_name:
                    gripper_joints.append((joint_name, limit_info))
                else:
                    other_joints.append((joint_name, limit_info))

        # 显示腰部关节
        if waist_joints:
            print("\n  [腰部关节]")
            for joint_name, limit_info in waist_joints:
                joint_type = limit_info.get('type', 'unknown')
                print(f"    {joint_name} ({joint_type}): [{limit_info['lower']:.4f}, {limit_info['upper']:.4f}], "
                      f"范围: {limit_info['range']:.4f}")

        # 显示头部关节
        if head_joints:
            print("\n  [头部关节]")
            for joint_name, limit_info in head_joints:
                joint_type = limit_info.get('type', 'unknown')
                print(f"    {joint_name} ({joint_type}): [{limit_info['lower']:.4f}, {limit_info['upper']:.4f}], "
                      f"范围: {limit_info['range']:.4f}")

        # 显示手臂关节（仅显示前几个）
        if arm_joints:
            print(f"\n  [手臂关节] (共 {len(arm_joints)} 个，显示前5个)")
            for joint_name, limit_info in arm_joints[:5]:
                joint_type = limit_info.get('type', 'unknown')
                print(f"    {joint_name} ({joint_type}): [{limit_info['lower']:.4f}, {limit_info['upper']:.4f}], "
                      f"范围: {limit_info['range']:.4f}")

        # 显示夹爪关节（仅显示前几个）
        if gripper_joints:
            print(f"\n  [夹爪关节] (共 {len(gripper_joints)} 个，显示前5个)")
            for joint_name, limit_info in gripper_joints[:5]:
                joint_type = limit_info.get('type', 'unknown')
                print(f"    {joint_name} ({joint_type}): [{limit_info['lower']:.4f}, {limit_info['upper']:.4f}], "
                      f"范围: {limit_info['range']:.4f}")

        if not matched_any and joint_name_pattern != ".*":
            print(f"  未找到匹配 '{joint_name_pattern}' 的关节")
    def initialize(self, headless: bool = False):
        """
        初始化仿真环境
        
        Args:
            headless: 是否无头模式运行
        """
        # 导入 Isaac Lab 
        # 这里假设已经在主脚本中启动了 AppLauncher
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
        self.sim.set_camera_view(eye=[2.5, 2.5, 2.0], target=[0.0, 0.0, 0.8])
        
        # 创建场景
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/GroundPlane", ground_cfg)
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/Light", light_cfg)
        
        # 创建机器人
        robot_cfg = self._create_robot_cfg(sim_utils, ArticulationCfg, ImplicitActuatorCfg)
        self.robot = Articulation(robot_cfg)
        
        # 重置仿真
        self.sim.reset()
        self.robot.reset()
        
        # 获取关节信息
        self.robot_joint_names = list(self.robot.joint_names)
        self._build_joint_mapping()
        
        print(f"[SimRunner] 仿真初始化完成")
        print(f"[SimRunner] 物理频率: {1.0/self.config.physics_dt:.1f} Hz")
        print(f"[SimRunner] 控制频率: {1.0/self.config.control_dt:.1f} Hz")
        print(f"[SimRunner] Decimation: {self.config.decimation}")
        print(f"[SimRunner] 关节数量: {len(self.robot_joint_names)}")
        self.print_joint_limits(".*")
    
    def _create_robot_cfg(self, sim_utils, ArticulationCfg, ImplicitActuatorCfg):
        """创建机器人配置"""
        urdf_cfg = sim_utils.UrdfFileCfg(
            asset_path=str(self.config.urdf_path),
            fix_base=self.config.fix_base,
            force_usd_conversion=True,
        )
        
        # 清除 URDF 默认驱动 （自己手动设置驱动参数）
        if hasattr(urdf_cfg, 'joint_drive'):
            jd = urdf_cfg.joint_drive
            if hasattr(jd, 'gains'):
                jd.gains.stiffness = 0.0
                jd.gains.damping = 0.0
            if hasattr(jd, 'drive_type'):
                jd.drive_type = "force"
        
        act = self.config.actuator
        
        return ArticulationCfg(
            prim_path="/World/Robot",
            spawn=urdf_cfg,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                joint_pos={".*": 0.0},
            ),
            actuators={
                # 注意: 腰部关节已在URDF中固定为fixed类型，不需要执行器
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
        """
        将 16 维关节目标映射到仿真关节数组

        Args:
            joint_targets: 16维目标位置 (头部2+左臂7+右臂7)
            gripper_l: 左夹爪值 [0,120]
            gripper_r: 右夹爪值 [0,120]

        Returns:
            仿真关节目标数组
        """
        result = np.zeros(len(self.robot_joint_names), dtype=np.float32)

        # 16维关节名称映射（移除腰部关节）
        joint_names_16d = [
            "idx11_head_joint1", "idx12_head_joint2",
            "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
            "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6",
            "idx27_arm_l_joint7",
            "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
            "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6",
            "idx67_arm_r_joint7",
        ]

        # 映射 16 维关节
        for i, name in enumerate(joint_names_16d):
            if name in self.control_joint_indices:
                result[self.control_joint_indices[name]] = joint_targets[i]
        
        # 映射夹爪
        gripper_l_urdf = self._convert_gripper_value(gripper_l)
        gripper_r_urdf = self._convert_gripper_value(gripper_r)
        
        # 夹爪主关节
        gripper_mapping = {
            "idx41_gripper_l_outer_joint1": gripper_l_urdf,
            "idx81_gripper_r_outer_joint1": gripper_r_urdf,
        }
        
        # 从动关节
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
        """从仿真关节数组提取 16 维数据（移除腰部关节）"""
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
    
    def _validate_targets(self, target_positions: np.ndarray):
        """验证目标值是否在 URDF 限位范围内"""
        
        # URDF 定义的关节限位
        joint_limits = {
            0: ("waist_lift", 0.0, 0.55, "m"),         # prismatic
            1: ("waist_rotate", 0.0, 1.5708, "rad"),   # revolute
            2: ("head_yaw", -1.5708, 1.5708, "rad"),
            3: ("head_pitch", -0.3491, 0.5236, "rad"),
        }
        
        print("\n[验证] 目标值范围 vs URDF 限位:")
        for idx, (name, lower, upper, unit) in joint_limits.items():
            if idx < target_positions.shape[1]:
                data_min = target_positions[:, idx].min()
                data_max = target_positions[:, idx].max()
                
                in_range = (data_min >= lower) and (data_max <= upper)
                status = "✓" if in_range else "✗ 超限!"
                
                print(f"  [{idx}] {name}: data[{data_min:.4f}, {data_max:.4f}] "
                      f"vs limit[{lower:.4f}, {upper:.4f}] {unit} {status}")
    def _get_left_ee_position(self):
        """获取左末端执行器位置"""
        try:
            # 查找左末端执行器的链接名称 - 根据G1机器人的URDF命名约定
            left_ee_link_names = [name for name in self.robot.body_names if 'arm_r' in name and ('link7' in name or 'joint7' in name or 'wrist' in name.lower())]
            if not left_ee_link_names:
                # 再次确认：根据注释，右臂应该是idx61-arm_r_joint1到idx67-arm_r_joint7
                left_ee_link_names = [name for name in self.robot.body_names if 'idx67' in name]
            
            if left_ee_link_names:
                link_name = left_ee_link_names[0]
                # 获取链接索引
                link_idx = self.robot.body_names.index(link_name)
                # 获取位置
                ee_pos = self.robot.data.body_pos_w[0, link_idx].cpu().numpy()
                return ee_pos
        except Exception as e:
            print(f"[SimRunner] 获取左末端执行器位置失败: {e}")
            return None
        return None

    def _get_right_ee_position(self):
        """获取右末端执行器位置"""
        try:
            # 查找右末端执行器的链接名称 - 根据G1机器人的URDF命名约定
            right_ee_link_names = [name for name in self.robot.body_names if 'arm_l' in name and ('link7' in name or 'joint7' in name or 'wrist' in name.lower())]
            if not right_ee_link_names:
                # 再次确认：根据注释，左臂应该是idx21-arm_l_joint1到idx27-arm_l_joint7
                right_ee_link_names = [name for name in self.robot.body_names if 'idx27' in name]
            
            if right_ee_link_names:
                link_name = right_ee_link_names[0]
                # 获取链接索引
                link_idx = self.robot.body_names.index(link_name)
                # 获取位置
                ee_pos = self.robot.data.body_pos_w[0, link_idx].cpu().numpy()
                return ee_pos
        except Exception as e:
            print(f"[SimRunner] 获取右末端执行器位置失败: {e}")
            return None
        return None
    def run_replay(
        self,
        target_positions: np.ndarray,
        target_grippers_l: np.ndarray,
        target_grippers_r: np.ndarray,
        timestamps: np.ndarray,
        simulation_app,
        initial_position: Optional[np.ndarray] = None,
    ) -> List[SimRecord]:
        """
        执行回放
        
        Args:
            target_positions: (N, 16) 关节目标位置
            target_grippers_l: (N,) 左夹爪目标
            target_grippers_r: (N,) 右夹爪目标
            timestamps: (N,) 时间戳
            simulation_app: IsaacSim 应用实例
            initial_position: 可选的初始位置
            
        Returns:
            List[SimRecord]: 仿真记录
        """
        import torch
        
        n_frames = len(timestamps)
        records = []
        
        # 设置初始位置
        if initial_position is not None:
            print("[SimRunner] 初始化关节位置...")
            print(f"  waist_lift target: {initial_position[0]:.4f}")
            print(f"  waist_rotate target: {initial_position[1]:.4f}")
            
            init_targets = self._map_targets_to_sim(
                initial_position,
                target_grippers_l[0],
                target_grippers_r[0]
            )
            
            # 打印初始目标 (调试)
            print(f"  waist_lift target: {initial_position[0]:.4f}")
            print(f"  waist_rotate target: {initial_position[1]:.4f}")
            
            init_tensor = torch.tensor(
                init_targets, 
                dtype=torch.float32, 
                device=self.sim.device
            ).unsqueeze(0)
            
            zero_vel = torch.zeros_like(init_tensor)
            
            # ★ 修复: 使用正确的 API
            # Isaac Lab 2.x 的 write_joint_state_to_sim 使用位置参数
            # 签名: write_joint_state_to_sim(position, velocity, joint_ids, env_ids)
            try:
                # 尝试 Isaac Lab 2.x API (位置参数)
                self.robot.write_joint_state_to_sim(init_tensor, zero_vel)
            except TypeError:
                # 备选: 直接设置 data 属性
                self.robot.data.joint_pos[:] = init_tensor
                self.robot.data.joint_vel[:] = zero_vel
            
            # 稳定仿真
            for _ in range(50):
                self.robot.set_joint_position_target(init_tensor)
                self.robot.write_data_to_sim()
                self.sim.step()
                self.robot.update(self.config.physics_dt)
            
            # 验证初始化结果
            actual_pos = self.robot.data.joint_pos[0].cpu().numpy()
            actual_16d = self._extract_16d_from_sim(actual_pos)
            print(f"  waist_lift actual: {actual_16d[0]:.4f}")
            print(f"  waist_rotate actual: {actual_16d[1]:.4f}")
        
        print(f"[SimRunner] 开始回放 {n_frames} 帧...")
        
        # 添加调试输出
        DEBUG_FRAMES = [0, 1, 2, 10, 100]
        
        for frame_idx in range(n_frames):
            if not simulation_app.is_running():
                print("[SimRunner] 仿真被中断")
                break
            
            # 获取当前目标
            target_pos = target_positions[frame_idx]
            gripper_l = target_grippers_l[frame_idx]
            gripper_r = target_grippers_r[frame_idx]
            
            # 映射到仿真
            sim_targets = self._map_targets_to_sim(target_pos, gripper_l, gripper_r)
            
            # 发送控制命令
            target_tensor = torch.tensor(
                sim_targets,
                dtype=torch.float32,
                device=self.sim.device
            ).unsqueeze(0)
            
            self.robot.set_joint_position_target(target_tensor)
            self.robot.write_data_to_sim()
            
            # 执行 decimation 个物理步骤
            for _ in range(self.config.decimation):
                self.sim.step()
            
            self.robot.update(self.config.physics_dt * self.config.decimation)
            
            # 读取状态
            joint_pos_full = self.robot.data.joint_pos[0].cpu().numpy()
            joint_vel_full = self.robot.data.joint_vel[0].cpu().numpy()
            
            # 尝试获取力矩 (可能不可用)
            if hasattr(self.robot.data, 'applied_torque'):
                joint_torque_full = self.robot.data.applied_torque[0].cpu().numpy()
            elif hasattr(self.robot.data, 'joint_effort'):
                joint_torque_full = self.robot.data.joint_effort[0].cpu().numpy()
            elif hasattr(self.robot.data, 'computed_torque'):
                joint_torque_full = self.robot.data.computed_torque[0].cpu().numpy()
            else:
                joint_torque_full = np.zeros_like(joint_pos_full)
            
            # 提取 16 维数据（移除腰部关节）
            joint_pos_16d = self._extract_16d_from_sim(joint_pos_full)
            joint_vel_16d = self._extract_16d_from_sim(joint_vel_full)
            joint_torque_16d = self._extract_16d_from_sim(joint_torque_full)

            left_ee_pos = self._get_left_ee_position()
            right_ee_pos = self._get_right_ee_position()


            # 提取夹爪位置 (从仿真关节中读取)
            gripper_l_urdf = joint_pos_full[self.control_joint_indices.get("idx41_gripper_l_outer_joint1", 0)]
            gripper_r_urdf = joint_pos_full[self.control_joint_indices.get("idx81_gripper_r_outer_joint1", 0)]
            # 转换回 [0,120] 范围
            sim_gripper_l = (1.0 - gripper_l_urdf / self.GRIPPER_URDF_MAX) * self.GRIPPER_STATE_MAX
            sim_gripper_r = (1.0 - gripper_r_urdf / self.GRIPPER_URDF_MAX) * self.GRIPPER_STATE_MAX

            # 记录
            record = SimRecord(
                sim_time=timestamps[frame_idx],
                frame_index=frame_idx,
                joint_pos=joint_pos_16d.copy(),
                joint_vel=joint_vel_16d.copy(),
                joint_torque=joint_torque_16d.copy(),
                target_pos=target_pos.copy(),
                gripper_l=sim_gripper_l,
                gripper_r=sim_gripper_r,
                target_gripper_l=gripper_l,
                target_gripper_r=gripper_r,
                left_ee_pos=left_ee_pos.copy() if left_ee_pos is not None else None,
                right_ee_pos=right_ee_pos.copy() if right_ee_pos is not None else None,
            )
            records.append(record)
            
            # 调试输出 每60帧（两秒）输出一次调试信息
            if frame_idx % 60 == 0:
                progress = frame_idx / n_frames * 100
                print(f"\n[Frame {frame_idx}] 调试信息:")
                print(f"[SimRunner] 进度: {frame_idx+1}/{n_frames} ({progress:.1f}%)")
                print(f"  === 头部 (腰部已固定) ===")
                print(f"  Target head_yaw: {target_pos[0]:.6f}, Sim: {joint_pos_16d[0]:.6f}, Error: {joint_pos_16d[0] - target_pos[0]:.6f}")
                print(f"  Target head_pitch: {target_pos[1]:.6f}, Sim: {joint_pos_16d[1]:.6f}, Error: {joint_pos_16d[1] - target_pos[1]:.6f}")
                print(f"  === 左臂 (7关节) ===")
                for i in range(7):
                    idx = 2 + i  # 头部2个关节后
                    print(f"  Target arm_l_j{i+1}: {target_pos[idx]:.6f}, Sim: {joint_pos_16d[idx]:.6f}, Error: {joint_pos_16d[idx] - target_pos[idx]:.6f}")
                print(f"  === 右臂 (7关节) ===")
                for i in range(7):
                    idx = 9 + i  # 头部2+左臂7后
                    print(f"  Target arm_r_j{i+1}: {target_pos[idx]:.6f}, Sim: {joint_pos_16d[idx]:.6f}, Error: {joint_pos_16d[idx] - target_pos[idx]:.6f}")
                print(f"  === 夹爪 ===")
                print(f"  Target gripper_l: {gripper_l:.2f}, Sim: {sim_gripper_l:.2f}, Error: {sim_gripper_l - gripper_l:.2f}")
                print(f"  Target gripper_r: {gripper_r:.2f}, Sim: {sim_gripper_r:.2f}, Error: {sim_gripper_r - gripper_r:.2f}")
                if left_ee_pos is not None:
                    print(f"  Left EE Pos: [{left_ee_pos[0]:.4f}, {left_ee_pos[1]:.4f}, {left_ee_pos[2]:.4f}]")
                if right_ee_pos is not None:
                    print(f"  Right EE Pos: [{right_ee_pos[0]:.4f}, {right_ee_pos[1]:.4f}, {right_ee_pos[2]:.4f}]")
        
        print(f"[SimRunner] 回放完成，共 {len(records)} 帧")
        print(f"[SimRunner] 进度: {n_frames}/{n_frames} ({100.0:.1f}%)")

        return records
    
    def close(self):
        """关闭仿真"""
        if self.sim is not None:
            self.sim = None
            self.robot = None


def records_to_arrays(records: List[SimRecord]) -> Dict[str, np.ndarray]:
    """将仿真记录转换为数组"""
    has_ee_pos = records and records[0].left_ee_pos is not None and records[0].right_ee_pos is not None

    result = {
        "sim_time": np.array([r.sim_time for r in records]),
        "frame_index": np.array([r.frame_index for r in records]),
        "sim_joint_pos": np.stack([r.joint_pos for r in records]),
        "sim_joint_vel": np.stack([r.joint_vel for r in records]),
        "sim_joint_torque": np.stack([r.joint_torque for r in records]),
        "target_pos": np.stack([r.target_pos for r in records]),
        "sim_gripper_l": np.array([r.gripper_l for r in records]),
        "sim_gripper_r": np.array([r.gripper_r for r in records]),
        "target_gripper_l": np.array([r.target_gripper_l for r in records]),
        "target_gripper_r": np.array([r.target_gripper_r for r in records]),
    }

    # 添加末端执行器位置 (如果存在)
    if has_ee_pos:
        # 组合左右手末端执行器位置为 (N, 6) 数组
        left_ee = np.stack([r.left_ee_pos for r in records])  # (N, 3)
        right_ee = np.stack([r.right_ee_pos for r in records])  # (N, 3)
        result["sim_end_effector_pos"] = np.concatenate([left_ee, right_ee], axis=1)  # (N, 6)

    return result