"""
AGIBOT G1 IsaacLab 数据回放脚本 (IsaacLab 2.3.0 / Isaac Sim 5.1.0)
参数说明:
    --episode <值>      指定回放的 episode 编号或 "all" 回放全部
                        默认值: "0"
                        示例: --episode 5      回放第5个episode
                              --episode all    依次回放所有episode
    
    --speed <值>        回放速度倍率 (跳帧数)
                        默认值: 1.0
                        示例: --speed 2.0      2倍速回放 (每次跳2帧)
    
    --loop              循环回放模式
                        默认: 关闭
                        启用后回放完毕会从头开始
    
    --num_envs <值>     并行环境数量 (IsaacLab 参数)
                        默认: 1
"""

import argparse
import sys
from pathlib import Path

# 路径设置
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
URDF_PATH = PROJECT_ROOT / "assets/G1_omnipicker/urdf/G1_omnipicker_fixed.urdf"
DATA_PATH = PROJECT_ROOT / "data/H3_example"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

ISAACLAB_ROOT_PATH = Path("/home/jianing/isaacsim/IsaacLab")
if not ISAACLAB_ROOT_PATH.exists():
    print(f"[Error] 未找到 IsaacLab: {ISAACLAB_ROOT_PATH}")
    sys.exit(1)

isaaclab_source = ISAACLAB_ROOT_PATH / "source"
if isaaclab_source.exists():
    sys.path.insert(0, str(isaaclab_source))
    for ext_dir in isaaclab_source.iterdir():
        if ext_dir.is_dir():
            sys.path.insert(0, str(ext_dir))

#导入isaaclab

try:
    from isaaclab.app import AppLauncher
except ImportError:
    print("[Error] 无法导入 isaaclab.app.AppLauncher")
    sys.exit(1)

parser = argparse.ArgumentParser(description="G1 Motion Replay")
parser.add_argument("--episode", type=str, default="0")
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--loop", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg, PhysxCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from data_utils import DataLoader, FrameRecord
from joint_mapping import IsaacLabJointMapper, StateExtractor


class G1ReplayController:
    """G1 回放控制器"""
    
    def __init__(self, episode_arg: str, speed: float = 1.0, loop: bool = False):
        self.episode_arg = episode_arg
        self.speed = speed
        self.loop = loop

        self.loader = DataLoader(str(DATA_PATH))
        self.mapper = IsaacLabJointMapper()
        self.state_extractor = StateExtractor()

        self.frames = []
        self.current_frame = 0
        self.current_episode = 0
        self.episode_list = []

        self.sim: SimulationContext = None
        self.robot: Articulation = None
        self.debug_printed = False
    
    def setup_simulation(self):
        """初始化仿真环境"""
        sim_cfg = SimulationCfg(
            dt=1.0 / 60.0,
            render_interval=1,
            physx=PhysxCfg(
                enable_stabilization=True,
            ),
        )
        self.sim = SimulationContext(sim_cfg)
        self.sim.set_camera_view(eye=[2.5, 2.5, 2.0], target=[0.0, 0.0, 0.8])
        
        # 场景
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/GroundPlane", ground_cfg)
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/Light", light_cfg)
        
        # 机器人
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
                "head": ImplicitActuatorCfg(
                    joint_names_expr=["idx1.*_head_joint.*"],
                    stiffness=50.0,
                    damping=5.0,
                ),
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=["idx.*_arm_[lr]_joint.*"],
                    stiffness=100.0,
                    damping=10.0,
                ),
                "grippers": ImplicitActuatorCfg(
                    joint_names_expr=["idx.*_gripper_[lr]_.*"],
                    stiffness=20.0,
                    damping=2.0,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )
    
    def load_episodes(self):
        num_episodes = self.loader.config.total_episodes

        if self.episode_arg.lower() == "all":
            self.episode_list = list(range(num_episodes))
        else:
            ep_idx = int(self.episode_arg)
            if 0 <= ep_idx < num_episodes:
                self.episode_list = [ep_idx]
            else:
                print(f"[Error] Episode {ep_idx} 不存在")
                sys.exit(1)

        self._load_episode(self.episode_list[0])
    
    def _load_episode(self, episode_idx: int):
        self.frames = self.loader.load_episode(episode_idx)
        self.current_frame = 0
        self.current_episode = episode_idx

        duration = self.frames[-1].timestamp if self.frames else 0
        print(f"\n[Episode {episode_idx}] 帧数: {len(self.frames)}, 时长: {duration:.2f}s")
    
    def _next_episode(self) -> bool:
        current_idx = self.episode_list.index(self.current_episode)
        next_idx = current_idx + 1
        
        if next_idx < len(self.episode_list):
            self._load_episode(self.episode_list[next_idx])
            return True
        elif self.loop:
            self._load_episode(self.episode_list[0])
            return True
        return False
    
    def set_joint_targets(self, frame: FrameRecord):
        """设置关节目标位置"""
        # 从 FrameRecord 中提取关节位置和夹爪位置
        joint_pos_16d = frame.real_joint_pos  # 16维: 头部2 + 左臂7 + 右臂7
        gripper_l = frame.real_gripper_l
        gripper_r = frame.real_gripper_r

        # 使用 mapper 转换为 IsaacLab 格式
        positions = self.mapper.map_16d_to_sim(
            joint_pos_16d, gripper_l, gripper_r
        )

        if not self.debug_printed:
            print(f"\n[Debug] 16维关节位置: {joint_pos_16d}")
            print(f"[Debug] 夹爪 L/R: {gripper_l:.2f}, {gripper_r:.2f}")
            print(f"[Debug] IsaacLab 位置数组 ({len(positions)}维): {positions}")
            self.debug_printed = True

        positions_tensor = torch.tensor(
            positions, dtype=torch.float32, device=self.sim.device
        ).unsqueeze(0)

        self.robot.set_joint_position_target(positions_tensor)
        self.robot.write_data_to_sim()
    
    def step(self) -> bool:
        if self.current_frame >= len(self.frames):
            if not self._next_episode():
                return False

        frame = self.frames[self.current_frame]
        self.set_joint_targets(frame)
        self.current_frame += max(1, int(self.speed))
        return True
    
    def run(self):
        self.setup_simulation()
        self.load_episodes()
        
        self.sim.reset()
        self.robot.reset()
        
        # 获取机器人实际的关节顺序并更新映射
        actual_joint_names = list(self.robot.joint_names)
        self.mapper.set_robot_joint_names(actual_joint_names)

        print(f"\n[Robot] 关节数量: {self.robot.num_joints}")
        print(f"[Robot] 关节顺序: {actual_joint_names}")
        
        print("\n" + "="*50)
        print("[回放开始] 关闭窗口退出")
        print("="*50 + "\n")
        
        step_count = 0
        while simulation_app.is_running():
            if not self.step():
                print("\n[Done] 回放完毕")
                break
            
            # 执行仿真步进
            self.sim.step()
            # 从仿真读取最新状态
            self.robot.update(self.sim.get_physics_dt())
            
            step_count += 1
            if step_count % 60 == 0:
                progress = self.current_frame / max(1, len(self.frames)) * 100
                print(f"\r[Episode {self.current_episode}] Frame {self.current_frame}/{len(self.frames)} ({progress:.1f}%)", end="", flush=True)


def main():
    controller = G1ReplayController(
        episode_arg=args_cli.episode,
        speed=args_cli.speed,
        loop=args_cli.loop,
    )
    
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n[Exit] 用户中断")
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()