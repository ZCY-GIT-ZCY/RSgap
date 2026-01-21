"""
AGIBOT G1 IsaacLab Action 回放脚本
基于 Action (34维) 解析并驱动机器人，记录仿真数据
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from typing import List, Dict, Optional

# 路径配置 =====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
URDF_PATH = PROJECT_ROOT / "assets/G1_omnipicker/urdf/G1_omnipicker_fixed.urdf"
DATA_PATH = PROJECT_ROOT / "data/H3_example"
OUTPUT_PATH = PROJECT_ROOT / "data/sim_output"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

ISAACLAB_ROOT_PATH = Path("/home/jianing/isaacsim/IsaacLab")
if isaaclab_source := ISAACLAB_ROOT_PATH / "source":
    if isaaclab_source.exists():
        sys.path.insert(0, str(isaaclab_source))
        for ext_dir in isaaclab_source.iterdir():
            if ext_dir.is_dir():
                sys.path.insert(0, str(ext_dir))

# IsaacLab 启动 ================================================================
try:
    from isaaclab.app import AppLauncher
except ImportError:
    print("[Error] 无法导入 isaaclab.app.AppLauncher")
    sys.exit(1)

parser = argparse.ArgumentParser(description="G1 Action Replay")
parser.add_argument("--episode", type=str, default="0")
# ★ 删除 speed 参数，确保 1:1 帧对应
parser.add_argument("--loop", action="store_true")
parser.add_argument("--output_dir", type=str, default=str(OUTPUT_PATH), help="输出目录")
parser.add_argument("--stabilize_steps", type=int, default=1000, help="初始化稳定步数")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 延迟导入
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg, PhysxCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from data_loader import AgibotDataLoader


# 仿真数据记录器 ================================================================
class SimDataRecorder:
    """仿真数据记录器 - 记录仿真中的关节状态"""
    
    # URDF 关节名到数据索引的映射
    JOINT_NAME_TO_ARM_IDX = {
        "idx21_arm_l_joint1": 0, "idx22_arm_l_joint2": 1, "idx23_arm_l_joint3": 2,
        "idx24_arm_l_joint4": 3, "idx25_arm_l_joint5": 4, "idx26_arm_l_joint6": 5,
        "idx27_arm_l_joint7": 6,
        "idx61_arm_r_joint1": 7, "idx62_arm_r_joint2": 8, "idx63_arm_r_joint3": 9,
        "idx64_arm_r_joint4": 10, "idx65_arm_r_joint5": 11, "idx66_arm_r_joint6": 12,
        "idx67_arm_r_joint7": 13,
    }
    
    JOINT_NAME_TO_HEAD_IDX = {"idx11_head_joint1": 0, "idx12_head_joint2": 1}
    JOINT_NAME_TO_WAIST_IDX = {"idx01_body_joint1": 0, "idx02_body_joint2": 1}
    
    GRIPPER_L_MASTER = "idx41_gripper_l_outer_joint1"
    GRIPPER_R_MASTER = "idx81_gripper_r_outer_joint1"
    GRIPPER_STATE_MAX = 120.0
    GRIPPER_URDF_MAX = 0.785
    
    def __init__(self, output_dir: str, episode_index: int):
        self.output_dir = Path(output_dir)
        self.episode_index = episode_index
        self.frames: List[Dict] = []
        self.robot_joint_names: List[str] = []
        self.joint_name_to_idx: Dict[str, int] = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def set_joint_names(self, names: List[str]):
        self.robot_joint_names = list(names)
        self.joint_name_to_idx = {name: i for i, name in enumerate(names)}
        
    def _convert_gripper_urdf_to_state(self, urdf_value: float) -> float:
        """URDF [0, 0.785] -> state [120, 0]"""
        normalized = np.clip(urdf_value / self.GRIPPER_URDF_MAX, 0.0, 1.0)
        return self.GRIPPER_STATE_MAX * (1.0 - normalized)
    
    def record_frame(
        self,
        frame_index: int,  # ★ 现在是仿真侧的连续索引
        timestamp: float,
        joint_positions: torch.Tensor,
    ):
        """记录一帧仿真数据"""
        pos = joint_positions.cpu().numpy().flatten()
        
        # 提取夹爪位置
        left_effector = 0.0
        right_effector = 0.0
        if self.GRIPPER_L_MASTER in self.joint_name_to_idx:
            idx = self.joint_name_to_idx[self.GRIPPER_L_MASTER]
            left_effector = self._convert_gripper_urdf_to_state(pos[idx])
        if self.GRIPPER_R_MASTER in self.joint_name_to_idx:
            idx = self.joint_name_to_idx[self.GRIPPER_R_MASTER]
            right_effector = self._convert_gripper_urdf_to_state(pos[idx])
        
        # 提取关节位置 (14维)
        joint_position = np.zeros(14, dtype=np.float32)
        for joint_name, data_idx in self.JOINT_NAME_TO_ARM_IDX.items():
            if joint_name in self.joint_name_to_idx:
                joint_position[data_idx] = pos[self.joint_name_to_idx[joint_name]]
        
        # 提取头部位置 (2维)
        head_position = np.zeros(2, dtype=np.float32)
        for joint_name, data_idx in self.JOINT_NAME_TO_HEAD_IDX.items():
            if joint_name in self.joint_name_to_idx:
                head_position[data_idx] = pos[self.joint_name_to_idx[joint_name]]
        
        # 提取腰部位置 (2维)
        waist_position = np.zeros(2, dtype=np.float32)
        for joint_name, data_idx in self.JOINT_NAME_TO_WAIST_IDX.items():
            if joint_name in self.joint_name_to_idx:
                waist_position[data_idx] = pos[self.joint_name_to_idx[joint_name]]
        
        self.frames.append({
            "frame_index": frame_index,
            "timestamp": timestamp,
            "left_effector_position": left_effector,
            "right_effector_position": right_effector,
            "joint_position": joint_position.tolist(),
            "head_position": head_position.tolist(),
            "waist_position": waist_position.tolist(),
        })
    
    def get_frame_count(self) -> int:
        """返回已记录的帧数"""
        return len(self.frames)
        
    def save_parquet(self):
        """保存为 parquet 格式"""
        try:
            import pandas as pd
        except ImportError:
            print("[Error] 需要安装 pandas: pip install pandas pyarrow")
            return
        
        if not self.frames:
            print("[Warning] 没有数据可保存")
            return
            
        df = pd.DataFrame(self.frames)
        
        chunk_index = self.episode_index // 1000
        chunk_dir = self.output_dir / f"chunk-{chunk_index:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = chunk_dir / f"episode_{self.episode_index:06d}.parquet"
        df.to_parquet(output_file, index=False)
        print(f"[Recorder] 已保存 {len(self.frames)} 帧到 {output_file}")


# Action 映射类 ================================================================
class G1ActionMapper:
    """处理 34维 Action -> IsaacLab 关节目标的映射"""
    
    GRIPPER_STATE_MAX = 120.0
    GRIPPER_URDF_MAX = 0.785
    
    # Action 索引定义
    IDX_GRIPPER_L = 0
    IDX_GRIPPER_R = 1
    IDX_ARM_L_START = 16
    IDX_ARM_R_START = 23
    IDX_HEAD_START = 30
    IDX_WAIST_START = 32
    
    def __init__(self):
        self.robot_joint_names = []
        
        # 直接控制的关节映射
        self.ctrl_map = {
            "idx01_body_joint1": (self.IDX_WAIST_START + 0, 1.0),
            "idx02_body_joint2": (self.IDX_WAIST_START + 1, 1.0),
            "idx11_head_joint1": (self.IDX_HEAD_START + 0, 1.0),
            "idx12_head_joint2": (self.IDX_HEAD_START + 1, 1.0),
        }
        for i in range(7):
            self.ctrl_map[f"idx2{i+1}_arm_l_joint{i+1}"] = (self.IDX_ARM_L_START + i, 1.0)
            self.ctrl_map[f"idx6{i+1}_arm_r_joint{i+1}"] = (self.IDX_ARM_R_START + i, 1.0)
        
        # 夹爪主关节
        self.gripper_master_joints = {
            "idx41_gripper_l_outer_joint1": (self.IDX_GRIPPER_L, 1.0),
            "idx81_gripper_r_outer_joint1": (self.IDX_GRIPPER_R, 1.0),
        }
        
        # 夹爪从动关节
        self.mimic_map = {
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

    def set_joint_names(self, names):
        self.robot_joint_names = list(names)
    
    def _convert_gripper_value(self, action_value: float) -> float:
        """action [0, 120] -> URDF [0.785, 0]"""
        normalized = np.clip(action_value / self.GRIPPER_STATE_MAX, 0.0, 1.0)
        return self.GRIPPER_URDF_MAX * (1.0 - normalized)

    def action_to_sim(self, action: np.ndarray) -> np.ndarray:
        """将 action 前34维映射到仿真关节目标"""
        if not self.robot_joint_names:
            return np.zeros(34, dtype=np.float32)
            
        targets = np.zeros(len(self.robot_joint_names), dtype=np.float32)
        
        # 计算夹爪值
        gripper_urdf_values = {}
        for joint_name, (action_idx, multiplier) in self.gripper_master_joints.items():
            base_value = self._convert_gripper_value(action[action_idx])
            gripper_urdf_values[joint_name] = base_value * multiplier
        
        # 映射所有关节
        for i, name in enumerate(self.robot_joint_names):
            if name in self.ctrl_map:
                idx, mul = self.ctrl_map[name]
                targets[i] = action[idx] * mul
            elif name in self.gripper_master_joints:
                targets[i] = gripper_urdf_values[name]
            elif name in self.mimic_map:
                master_name, multiplier = self.mimic_map[name]
                if master_name in gripper_urdf_values:
                    targets[i] = gripper_urdf_values[master_name] * multiplier
        
        return targets


# 控制器 =======================================================================
class G1ActionReplayController:
    def __init__(self, episode_arg: str, loop: bool = False, 
                 output_dir: str = str(OUTPUT_PATH), stabilize_steps: int = 100):
        self.episode_arg = episode_arg
        self.loop = loop
        self.output_dir = output_dir
        self.stabilize_steps = stabilize_steps  # ★ 初始化稳定步数
        
        self.loader = AgibotDataLoader(str(DATA_PATH))
        self.mapper = G1ActionMapper()
        self.recorder: Optional[SimDataRecorder] = None
        
        self.frames = []
        self.current_frame = 0
        self.sim = None
        self.robot = None
        
        # 仿真频率设置：物理60Hz，数据30Hz
        self.sim_dt = 1.0 / 60.0
        self.sim_steps_per_frame = 2  # 每个数据帧执行2次物理步
        self.sub_step = 0
        
        # ★ 初始化状态缓存
        self.initial_targets: Optional[np.ndarray] = None

    def setup_simulation(self):
        sim_cfg = SimulationCfg(
            dt=self.sim_dt, 
            render_interval=1, 
            physx=PhysxCfg(enable_stabilization=True)
        )
        self.sim = SimulationContext(sim_cfg)
        self.sim.set_camera_view(eye=[2.5, 2.5, 2.0], target=[0.0, 0.0, 0.8])
        
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/GroundPlane", ground_cfg)
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/Light", light_cfg)
        
        robot_cfg = self._create_robot_cfg()
        self.robot = Articulation(robot_cfg)
        print("[Setup] 仿真环境初始化完成")

    def _create_robot_cfg(self) -> ArticulationCfg:
        urdf_cfg = sim_utils.UrdfFileCfg(
            asset_path=str(URDF_PATH),
            fix_base=True,
            force_usd_conversion=True,
        )
        
        if hasattr(urdf_cfg, 'joint_drive'):
            jd = urdf_cfg.joint_drive
            if hasattr(jd, 'gains'):
                jd.gains.stiffness = 0.0
                jd.gains.damping = 0.0
            if hasattr(jd, 'drive_type'):
                jd.drive_type = "force"
        
        return ArticulationCfg(
            prim_path="/World/Robot",
            spawn=urdf_cfg,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                joint_pos={".*": 0.0},  
            ),
            actuators={
                "waist": ImplicitActuatorCfg(
                    joint_names_expr=["idx0.*_body_joint.*"],
                    stiffness=4000.0,
                    damping=400.0,
                ),
                "head": ImplicitActuatorCfg(
                    joint_names_expr=["idx1.*_head_joint.*"],
                    stiffness=100.0,
                    damping=10.0,
                ),
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=["idx.*_arm_[lr]_joint.*"],
                    stiffness=200.0,
                    damping=20.0,
                ),
                "grippers": ImplicitActuatorCfg(
                    joint_names_expr=["idx.*_gripper_[lr]_.*"],
                    stiffness=50.0,
                    damping=5.0,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )

    def _initialize_robot_state(self):
        """
        根据第0帧的 action 强制设置机器人初始状态并稳定：
        - 先执行 force_steps 步“强制写入”（每步把关节位置/速度直接写入 sim），以避免 actuators 抵抗导致的大偏差
        - 然后执行 stabilize_steps 步 position-target 稳定并检查误差
        """
        if len(self.frames) == 0:
            print("[Warning] 没有数据帧，跳过初始化")
            return

        first_frame = self.frames[0]
        action_vec = first_frame.action if hasattr(first_frame, 'action') and first_frame.action is not None else np.zeros(36)

        # 目标关节（仿真顺序）
        self.initial_targets = self.mapper.action_to_sim(action_vec[:34])

        print(f"[Init] 根据第0帧 action 初始化机器人状态...")
        print(f"  目标腰部: [{self.initial_targets[self._get_joint_idx('idx01_body_joint1')]:.4f}, "
              f"{self.initial_targets[self._get_joint_idx('idx02_body_joint2')]:.4f}]")

        # 创建张量（放到 sim 设备上）
        init_tensor = torch.tensor(self.initial_targets, dtype=torch.float32, device=self.sim.device).unsqueeze(0)
        zero_vel = torch.zeros_like(init_tensor)

        # 强制写入函数（兼容不同 API）
        def force_write_state_once():
            try:
                # 首选 API
                self.robot.write_joint_state_to_sim(init_tensor, zero_vel)
                return True
            except Exception:
                try:
                    # 逐元素拷贝到 data（batch dim 0）
                    self.robot.data.joint_pos[0].copy_(init_tensor[0])
                    self.robot.data.joint_vel[0].copy_(zero_vel[0])
                    self.robot.write_data_to_sim()
                    return True
                except Exception:
                    try:
                        # 最后退回到整体赋值
                        self.robot.data.joint_pos[:] = init_tensor
                        self.robot.data.joint_vel[:] = zero_vel
                        self.robot.write_data_to_sim()
                        return True
                    except Exception:
                        return False

        # 先做几步强制写入，抵消 actuator 力矩
        force_steps = min(300, max(20, self.stabilize_steps // 5))
        print(f"[Init] 强制写入初始位置 {force_steps} 步...")
        wrote_any = False
        for step in range(force_steps):
            ok = force_write_state_once()
            print(f"force_write ok={ok}, pos_after_write={self.robot.data.joint_pos[0].cpu().numpy()[:6]}")
            wrote_any = wrote_any or ok
            # 小步仿真推进
            self.sim.step()
            self.robot.update(self.sim_dt)
            if (step + 1) % 20 == 0:
                # 打印当前误差以观察收敛
                try:
                    cur = self.robot.data.joint_pos[0].cpu().numpy()
                    err = np.abs(cur - self.initial_targets).max()
                    print(f"    Force Step {step+1}: max error = {err:.6f}")
                except Exception:
                    pass

        if not wrote_any:
            print("[Warning] 无法强制写入初始关节状态，改用目标控制尝试稳定")

        # 然后以 position target 运行稳定步数并检查误差
        self.robot.set_joint_position_target(init_tensor)
        self.robot.write_data_to_sim()

        print(f"[Init] 执行 {self.stabilize_steps} 步 position-target 稳定并检查收敛...")
        final_error = float("inf")
        for step in range(self.stabilize_steps):
            self.robot.set_joint_position_target(init_tensor)
            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(self.sim.get_physics_dt())

            if (step + 1) % 20 == 0:
                cur = self.robot.data.joint_pos[0].cpu().numpy()
                err = np.abs(cur - self.initial_targets).max()
                print(f"    Step {step+1}: max error = {err:.6f}")
                final_error = err

        # 最终检查并报告
        final_pos = self.robot.data.joint_pos[0].cpu().numpy()
        final_error = np.abs(final_pos - self.initial_targets).max()
        if final_error < 1e-4:
            print(f"[Init] 初始化成功，最大误差: {final_error:.6f}")
        else:
            print(f"[Warning] 初始化未完全收敛 (max_error={final_error:.6f})，已尽力强制设置，继续回放但结果可能存在偏差")

    def _get_joint_idx(self, joint_name: str) -> int:
        """获取关节在仿真中的索引"""
        try:
            return list(self.robot.joint_names).index(joint_name)
        except ValueError:
            return 0

    def _debug_mapping_and_actuators(self, frame):
        # 打印原始 action / observation（如果有）
        obs_state = None
        if hasattr(frame, "observation"):
            obs = getattr(frame, "observation")
            obs_state = getattr(obs, "state", None) or getattr(obs, "robot_state", None)
        obs_vals = obs_state[84:86] if (obs_state is not None and len(obs_state) >= 86) else None
        print("DEBUG: frame.action[32:34] =", getattr(frame, "action", None)[32:34])
        print("DEBUG: frame.observation.waist =", obs_vals)

        # 比较 action 映射 vs（如果存在）position 映射
        action_targets = self.mapper.action_to_sim(getattr(frame, "action", np.zeros(36))[:34])
        print("DEBUG: action->sim waist:", action_targets[self._get_joint_idx("idx01_body_joint1")],
              action_targets[self._get_joint_idx("idx02_body_joint2")])

        # 打印每个关节名及其目标（便于人工比对）
        for i, name in enumerate(self.robot.joint_names):
            print(f"  joint[{i}] {name}: target={action_targets[i]:.6f}, pos={self.robot.data.joint_pos[0].cpu().numpy()[i]:.6f}")

        # 列出 actuator 匹配到的关节（用 regex 与 joint_names 匹配）
        import re
        print("DEBUG: Actuator coverage:")
        for act_name, cfg in self._create_robot_cfg().actuators.items():
            expr = cfg.joint_names_expr if hasattr(cfg, "joint_names_expr") else getattr(cfg, "joint_names", None)
            if expr is None:
                print(f"  {act_name}: NO joint_names_expr")
                continue
            matched = [jn for jn in self.robot.joint_names if re.match(expr[0], jn)]
            print(f"  {act_name}: matches {len(matched)} joints: {matched}")

        # 验证强制写入后是否立即被物理步改变
        init_tensor = torch.tensor(action_targets, dtype=torch.float32, device=self.sim.device).unsqueeze(0)
        zero_vel = torch.zeros_like(init_tensor)
        ok = False
        try:
            self.robot.write_joint_state_to_sim(init_tensor, zero_vel)
            ok = True
        except Exception:
            try:
                self.robot.data.joint_pos[0].copy_(init_tensor[0])
                self.robot.data.joint_vel[0].copy_(zero_vel[0])
                self.robot.write_data_to_sim()
                ok = True
            except Exception:
                ok = False
        print("DEBUG: force write ok =", ok)
        # 读写后立刻检查并再做几步物理观察
        cur = self.robot.data.joint_pos[0].cpu().numpy().copy()
        print("  after write pos[:6]=", cur[:6])
        for s in range(5):
            self.sim.step()
            self.robot.update(self.sim_dt)
        cur2 = self.robot.data.joint_pos[0].cpu().numpy().copy()
        print("  after 5 steps pos[:6]=", cur2[:6])
        diffs = np.abs(cur2 - action_targets)
        # 打印 top-5 最大差异关节
        idx_sorted = np.argsort(-diffs)[:5]
        print("DEBUG: top-5 diffs (name, diff, pos, target):")
        for idx in idx_sorted:
            print(f"  {self.robot.joint_names[idx]}: {diffs[idx]:.6f}, pos={cur2[idx]:.6f}, target={action_targets[idx]:.6f}")

    def load_episode(self, idx: int):
        self.frames = self.loader.load_episode(idx)
        self.current_frame = 0
        self.sub_step = 0
        
        self.recorder = SimDataRecorder(self.output_dir, idx)
        self.recorder.set_joint_names(self.robot.joint_names)
        
        # ★ 初始化机器人到第0帧状态
        self._initialize_robot_state()
        if self.initial_targets is not None:
            waist_idx_start = self._get_joint_idx("idx01_body_joint1")
            waist_idx_end = self._get_joint_idx("idx02_body_joint2") + 1
            print("mapped initial_targets[waist]:", self.initial_targets[waist_idx_start:waist_idx_end])
        else:
            print("[Info] initial_targets 未设置（初始化失败或无帧）")
        
        # 运行诊断（基于第0帧）
        if len(self.frames) > 0:
            self._debug_mapping_and_actuators(self.frames[0])
        
        print(f"[Episode {idx}] 加载 {len(self.frames)} 帧，输出到 {self.output_dir}")


    def step(self) -> bool:
        """
        执行一步仿真，返回是否继续
        
        ★ 关键修改：
        1. 删除 speed 参数，每次只前进 1 帧
        2. frame_index 使用仿真侧的连续索引
        3. timestamp 从源数据获取，保持对应关系
        """
        if self.current_frame >= len(self.frames):
            if self.loop:
                self.current_frame = 0
                self.sub_step = 0
                self._initialize_robot_state()  # 重新初始化
            else:
                return False
            
        frame = self.frames[self.current_frame]
        action_vec = frame.action if hasattr(frame, 'action') and frame.action is not None else np.zeros(36)
        
        # 只使用前34维 (不含 robot_velocity)
        targets = self.mapper.action_to_sim(action_vec[:34])
        
        # 发送控制指令
        self.robot.set_joint_position_target(
            torch.tensor(targets, device=self.sim.device).unsqueeze(0)
        )
        self.robot.write_data_to_sim()
        
        # 执行物理仿真
        self.sim.step()
        self.robot.update(self.sim.get_physics_dt())
        
        # 每2个物理步记录一帧 (30Hz)
        self.sub_step += 1
        if self.sub_step >= self.sim_steps_per_frame:
            self.sub_step = 0
            
            # ★ 记录数据 - 使用 current_frame 作为连续索引
            if self.recorder is not None:
                self.recorder.record_frame(
                    frame_index=self.current_frame,  # ★ 使用仿真侧连续索引
                    timestamp=frame.timestamp,        # 保留源数据时间戳
                    joint_positions=self.robot.data.joint_pos[0],
                )
            
            # ★ 固定步进 1 帧，确保 1:1 对应
            self.current_frame += 1
        
        return True

    def run(self):
        self.setup_simulation()
        self.loader.load_meta()
        
        # 解析 episode 参数
        if self.episode_arg.lower() == "all":
            episode_list = list(range(self.loader.info.total_episodes))
        else:
            episode_list = [int(self.episode_arg)]
        
        self.sim.reset()
        self.mapper.set_joint_names(self.robot.joint_names)
        print("Robot joints:", list(self.robot.joint_names))
        for name in ["idx01_body_joint1","idx02_body_joint2"]:
            if name in self.robot.joint_names:
                print(name, "index", self.robot.joint_names.index(name))
            else:
                print(f"[Error] joint {name} not found in robot.joint_names")
        # 避免在 initial_targets 为空时访问
        if self.initial_targets is not None:
            waist_idx_start = self._get_joint_idx("idx01_body_joint1")
            waist_idx_end = self._get_joint_idx("idx02_body_joint2") + 1
            print("mapped initial_targets[waist]:", self.initial_targets[waist_idx_start:waist_idx_end])
        else:
            print("[Info] mapped initial_targets 还未设置（将在 load_episode 中初始化）")
        
        print(f"[Map] 配置 {len(self.robot.joint_names)} 个关节")
        
        for ep_idx in episode_list:
            self.load_episode(ep_idx)
            
            while simulation_app.is_running():
                if not self.step():
                    break
            
            # ★ 验证帧数对应
            source_frame_count = len(self.frames)
            sim_frame_count = self.recorder.get_frame_count() if self.recorder else 0
            
            if source_frame_count != sim_frame_count:
                print(f"[Warning] 帧数不匹配! 源数据: {source_frame_count}, 仿真: {sim_frame_count}")
            else:
                print(f"[OK] 帧数匹配: {sim_frame_count} 帧")
            
            # 保存当前 episode 数据
            if self.recorder is not None:
                self.recorder.save_parquet()
                print(f"[Episode {ep_idx}] 完成")


def main():
    try:
        ctrl = G1ActionReplayController(
            episode_arg=args_cli.episode,
            loop=args_cli.loop,
            output_dir=args_cli.output_dir,
            stabilize_steps=args_cli.stabilize_steps,
        )
        ctrl.run()
    except Exception as e:
        import traceback
        print(f"\n[Error] {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()