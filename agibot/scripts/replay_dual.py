"""
AGIBOT G1 Isaac Lab 双机对比回放脚本
====================================
功能:
1. 在同一场景中生成两台 G1 机器人进行对比
2. Robot_Obs (参考机): 运动学模式，显示真实关节位置 (Ground Truth)
3. Robot_Act (测试机): 物理驱动模式，显示 Action 的物理响应

用法:
    python replay_dual.py --episode 0
    python replay_dual.py --episode 0 --headless
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List
import pickle

# =============================================================================
# 路径配置
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# IsaacLab 路径配置
ISAACLAB_ROOT_PATH = Path("/home/jianing/isaacsim/IsaacLab")
if (isaaclab_source := ISAACLAB_ROOT_PATH / "source").exists():
    sys.path.insert(0, str(isaaclab_source))
    for ext_dir in isaaclab_source.iterdir():
        if ext_dir.is_dir() and ext_dir.name.startswith("ext"):
            sys.path.insert(0, str(ext_dir))


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class DualReplayConfig:
    """双机回放配置"""
    # 数据集
    dataset_path: str
    episode_index: str

    # URDF
    urdf_path: str          # Robot_Act 使用的 URDF (物理驱动)
    urdf_path_obs: str      # Robot_Obs 使用的 URDF (运动学模式，无碰撞)
    fix_base: bool

    # 仿真
    physics_dt: float
    control_dt: float

    # 机器人位置
    robot_obs_pos: tuple
    robot_act_pos: tuple

    # 执行器增益
    head_stiffness: float
    head_damping: float
    arm_stiffness: float
    arm_damping: float
    gripper_stiffness: float
    gripper_damping: float

    # 回放控制
    start_frame: int
    end_frame: int

    # 记录
    recording_enabled: bool
    output_dir: str

    # Isaac Lab
    headless: bool

    # 调试
    verbose: bool

    @classmethod
    def default(cls) -> "DualReplayConfig": #这里是设置配置的地方
        """默认配置"""
        return cls(
            dataset_path=str(PROJECT_ROOT / "data/H3_example"),
            episode_index="0",
            urdf_path=str(PROJECT_ROOT / "assets/G1_omnipicker/urdf/G1_omnipicker_fixed.urdf"),
            urdf_path_obs=str(PROJECT_ROOT / "assets/G1_omnipicker/urdf/G1_omnipicker_nocollision.urdf"),
            fix_base=True,
            physics_dt=1.0/90.0,
            control_dt=1.0/30.0,
            robot_obs_pos=(0.0, 0.0, 0.0),
            robot_act_pos=(0, 0.0, 0.0),
            head_stiffness=100.0,
            head_damping=10.0,
            arm_stiffness=400.0,
            arm_damping=40.0,
            gripper_stiffness=100.0,
            gripper_damping=10.0,
            start_frame=0,
            end_frame=-1,
            recording_enabled=True,
            output_dir="outputs/dual_replay_results",
            headless=False,
            verbose=True,
        )


# =============================================================================
# 命令行参数解析
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="AGIBOT G1 Isaac Lab Dual Robot Replay",
    )

    parser.add_argument(
        "--episode", "-e",
        type=str,
        default="0",
        help="Episode 编号"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="输出目录"
    )

    # 添加 IsaacLab 参数
    try:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    except ImportError:
        pass

    return parser.parse_args()


# =============================================================================
# 双机回放控制器
# =============================================================================

class DualReplayController:
    """双机回放控制器"""

    def __init__(self, config: DualReplayConfig, simulation_app):
        self.config = config
        self.simulation_app = simulation_app

        # 延迟导入
        from data_utils import DataLoader, frames_to_arrays, JointNameMapper
        from sim_dual_runner import (
            IsaacLabDualSimRunner, DualSimConfig, ActuatorConfig,
            dual_records_to_arrays
        )

        self.DataLoader = DataLoader
        self.frames_to_arrays = frames_to_arrays
        self.JointNameMapper = JointNameMapper
        self.IsaacLabDualSimRunner = IsaacLabDualSimRunner
        self.DualSimConfig = DualSimConfig
        self.ActuatorConfig = ActuatorConfig
        self.dual_records_to_arrays = dual_records_to_arrays

        # 组件
        self.data_loader = None
        self.sim_runner = None

        # 数据
        self.frames = []
        self.source_arrays = {}
        self.sim_records = []
        self.aligned_data = {}

    def initialize(self):
        """初始化所有组件"""
        print("\n" + "=" * 70)
        print("[DualReplayController] 初始化...")
        print("=" * 70)

        self.data_loader = self.DataLoader(self.config.dataset_path)

        sim_config = self.DualSimConfig(
            physics_dt=self.config.physics_dt,
            control_dt=self.config.control_dt,
            urdf_path=self.config.urdf_path,
            urdf_path_obs=self.config.urdf_path_obs,
            fix_base=self.config.fix_base,
            robot_obs_pos=self.config.robot_obs_pos,
            robot_act_pos=self.config.robot_act_pos,
            actuator=self.ActuatorConfig(
                head_stiffness=self.config.head_stiffness,
                head_damping=self.config.head_damping,
                arm_stiffness=self.config.arm_stiffness,
                arm_damping=self.config.arm_damping,
                gripper_stiffness=self.config.gripper_stiffness,
                gripper_damping=self.config.gripper_damping,
            ),
        )

        self.sim_runner = self.IsaacLabDualSimRunner(sim_config)
        self.sim_runner.initialize()

        print("\n[DualReplayController] 初始化完成!")

    def load_episode(self, episode_index: int):
        """加载单个 episode 数据"""
        print(f"\n[DualReplayController] 加载 Episode {episode_index}...")

        self.frames = self.data_loader.load_episode(episode_index)

        # 应用帧范围
        start = self.config.start_frame
        end = self.config.end_frame if self.config.end_frame > 0 else len(self.frames)
        self.frames = self.frames[start:end]

        # 转换为数组
        self.source_arrays = self.frames_to_arrays(self.frames)

        print(f"[DualReplayController] 加载 {len(self.frames)} 帧")

        if self.config.verbose:
            print("\n[数据摘要]:")
            for k, v in self.source_arrays.items():
                print(f"  {k}: {v.shape}")

    def run_replay(self) -> List:
        """执行双机回放"""
        print("\n[DualReplayController] 开始双机回放...")

        timestamps = self.source_arrays["timestamps"]
        real_positions = self.source_arrays["real_joint_pos"]
        target_positions = self.source_arrays["target_joint_pos"]
        real_gripper_l = self.source_arrays["real_gripper_l"]
        real_gripper_r = self.source_arrays["real_gripper_r"]
        target_gripper_l = self.source_arrays["target_gripper_l"]
        target_gripper_r = self.source_arrays["target_gripper_r"]

        self.sim_records = self.sim_runner.run_dual_replay(
            real_positions=real_positions,
            target_positions=target_positions,
            real_grippers_l=real_gripper_l,
            real_grippers_r=real_gripper_r,
            target_grippers_l=target_gripper_l,
            target_grippers_r=target_gripper_r,
            timestamps=timestamps,
            simulation_app=self.simulation_app,
        )

        return self.sim_records

    def align_data(self):
        """对齐数据"""
        print("\n[DualReplayController] 对齐数据...")

        sim_arrays = self.dual_records_to_arrays(self.sim_records)

        self.aligned_data = {
            "timestamps": self.source_arrays["timestamps"],
            "real_joint_pos": self.source_arrays["real_joint_pos"],
            "target_joint_pos": self.source_arrays["target_joint_pos"],
            "obs_joint_pos": sim_arrays["obs_joint_pos"],
            "act_joint_pos": sim_arrays["act_joint_pos"],
            "act_joint_vel": sim_arrays["act_joint_vel"],
            "act_joint_torque": sim_arrays["act_joint_torque"],
            "joint_names": self.JointNameMapper.get_joint_names(),
        }

        print(f"[DualReplayController] 对齐完成")

    def save_results(self, episode_index: int):
        """保存结果"""
        if not self.config.recording_enabled:
            return

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"dual_episode_{episode_index:06d}.pkl"

        with open(output_file, 'wb') as f:
            pickle.dump(self.aligned_data, f)

        print(f"[DualReplayController] 保存结果: {output_file}")

    def run_episode(self, episode_index: int):
        """运行单个 episode"""
        print("\n" + "=" * 70)
        print(f"[DualReplayController] 处理 Episode {episode_index}")
        print("=" * 70)

        self.load_episode(episode_index)
        self.run_replay()
        self.align_data()
        self.save_results(episode_index)

        print(f"\n[DualReplayController] Episode {episode_index} 完成!")

    def run(self):
        """运行主循环"""
        episode_arg = self.config.episode_index

        if episode_arg == "all":
            total = self.data_loader.get_episode_count()
            print(f"[DualReplayController] 处理所有 {total} 个 episode")
            for ep_idx in range(total):
                if not self.simulation_app.is_running():
                    break
                self.run_episode(ep_idx)
        else:
            ep_idx = int(episode_arg)
            self.run_episode(ep_idx)

    def close(self):
        """清理资源"""
        if self.sim_runner:
            self.sim_runner.close()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    args = parse_args()

    print("\n" + "=" * 70)
    print("AGIBOT G1 Isaac Lab Dual Robot Replay")
    print("=" * 70)

    try:
        from isaaclab.app import AppLauncher
    except ImportError as e:
        print(f"[Error] 无法导入 IsaacLab: {e}")
        sys.exit(1)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        # 使用默认配置
        config = DualReplayConfig.default()

        # 覆盖命令行参数
        config.episode_index = args.episode
        if args.headless:
            config.headless = True
        if args.output_dir:
            config.output_dir = args.output_dir

        # 创建控制器
        controller = DualReplayController(config, simulation_app)
        controller.initialize()
        controller.run()

        print("\n" + "=" * 70)
        print("[Main] 双机回放完成!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n[Main] 用户中断")
    except Exception as e:
        print(f"\n[Main] 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'controller' in locals():
            controller.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
