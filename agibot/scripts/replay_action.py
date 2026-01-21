"""
AGIBOT G1 Isaac Lab Action 回放脚本
====================================
主入口脚本，整合所有模块完成以下工作流程：

1. 加载配置文件 (replay_config.yaml)
2. 加载源数据 (parquet 格式)
3. 初始化 Isaac Lab 仿真环境
4. 执行回放循环，记录仿真数据
5. 时间对齐与插值
6. 生成对比图表
7. 导出结果

用法:
    # 基础用法
    python replay_action.py --episode 0
    
    # 指定配置文件
    python replay_action.py --config configs/replay_config.yaml --episode 0
    
    # 无头模式运行
    python replay_action.py --episode 0 --headless
    
    # 回放所有 episode
    python replay_action.py --episode all

依赖模块:
    - data_utils.py: 数据加载与预处理
    - sim_runner.py: Isaac Lab 仿真运行器
    - plotter.py: 可视化模块
    - joint_mapping.py: 关节映射配置
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pickle
import yaml
import numpy as np

# =============================================================================
# 路径配置
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 确保模块路径正确
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
# 配置解析
# =============================================================================

@dataclass
class ReplayConfig:
    """回放配置"""
    # 数据集
    dataset_path: str
    episode_index: Any  # int 或 "all"

    # URDF
    urdf_path: str
    fix_base: bool

    # 仿真
    physics_dt: float
    control_dt: float

    # 执行器增益 - 分离腰部移动和旋转关节

    head_stiffness: float
    head_damping: float
    arm_stiffness: float
    arm_damping: float
    gripper_stiffness: float
    gripper_damping: float

    # 夹爪转换
    gripper_state_max: float
    gripper_urdf_max: float

    # 回放控制
    speed: float
    loop: bool
    start_frame: int
    end_frame: int

    # 记录
    recording_enabled: bool
    output_dir: str
    save_pkl: bool

    # 可视化
    viz_enabled: bool
    viz_output_dir: str
    viz_format: str
    viz_dpi: int
    use_degrees: bool
    plot_groups: List[str]

    # Isaac Lab
    headless: bool

    # 调试
    verbose: bool
    print_mapping: bool
    print_every_n_frames: int
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ReplayConfig":
        """从 YAML 文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        return cls(
            # 数据集
            dataset_path=cfg['dataset']['path'],
            episode_index=cfg['dataset']['episode_index'],
            
            # URDF
            urdf_path=cfg['robot']['urdf_path'],
            fix_base=cfg['robot']['fix_base'],
            
            # 仿真
            physics_dt=cfg['simulation']['physics_dt'],
            control_dt=cfg['simulation']['control_dt'],
            
            # 执行器增益 - 分离腰部配置
            head_stiffness=cfg['actuators']['head']['stiffness'],
            head_damping=cfg['actuators']['head']['damping'],
            arm_stiffness=cfg['actuators']['arms']['stiffness'],
            arm_damping=cfg['actuators']['arms']['damping'],
            gripper_stiffness=cfg['actuators']['grippers']['stiffness'],
            gripper_damping=cfg['actuators']['grippers']['damping'],
            
            # 夹爪转换
            gripper_state_max=cfg['gripper']['state_max'],
            gripper_urdf_max=cfg['gripper']['urdf_max'],
            
            # 回放控制
            speed=cfg['playback']['speed'],
            loop=cfg['playback']['loop'],
            start_frame=cfg['playback']['start_frame'],
            end_frame=cfg['playback']['end_frame'],
            
            # 记录
            recording_enabled=cfg['recording']['enabled'],
            output_dir=cfg['recording']['output_dir'],
            save_pkl=cfg['recording']['save_pkl'],
            
            # 可视化
            viz_enabled=cfg['visualization']['enabled'],
            viz_output_dir=cfg['visualization']['output_dir'],
            viz_format=cfg['visualization']['format'],
            viz_dpi=cfg['visualization']['dpi'],
            use_degrees=cfg['visualization']['use_degrees'],
            plot_groups=cfg['visualization']['plot_groups'],
            
            # Isaac Lab
            headless=cfg['isaac_lab']['headless'],

            # 调试
            verbose=cfg['debug']['verbose'],
            print_mapping=cfg['debug']['print_mapping'],
            print_every_n_frames=cfg['debug']['print_every_n_frames'],
        )
    
    @classmethod
    def default(cls) -> "ReplayConfig":
        """默认配置"""
        return cls(
            dataset_path=str(PROJECT_ROOT / "data/H3_example"),
            episode_index=0,
            urdf_path=str(PROJECT_ROOT / "assets/G1_omnipicker/urdf/G1_omnipicker_fixed.urdf"),
            fix_base=True,
            physics_dt=1.0/60.0,
            control_dt=1.0/30.0,
            
            head_stiffness=100.0,
            head_damping=10.0,
            arm_stiffness=200.0,
            arm_damping=20.0,
            gripper_stiffness=50.0,
            gripper_damping=5.0,
            gripper_state_max=120.0,
            gripper_urdf_max=0.785,
            speed=1.0,
            loop=False,
            start_frame=0,
            end_frame=-1,
            recording_enabled=True,
            output_dir="outputs/replay_results",
            save_pkl=True,
            viz_enabled=True,
            viz_output_dir="outputs/plots",
            viz_format="png",
            viz_dpi=150,
            use_degrees=True,
            plot_groups=["head", "arm_l", "arm_r"],
            headless=False,
            verbose=True,
            print_mapping=True,
            print_every_n_frames=100,
        )


# =============================================================================
# 命令行参数解析
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="AGIBOT G1 Isaac Lab Action Replay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python replay_action.py --episode 0
    python replay_action.py --episode all --headless
    python replay_action.py --config my_config.yaml --episode 5
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=str(SCRIPT_DIR / "configs/replay_config.yaml"),
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--episode", "-e",
        type=str,
        default="0",
        help="Episode 编号 (整数或 'all')"
    )
    
    
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="禁用可视化"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="输出目录"
    )

    # 添加 IsaacLab 所需的参数
    try:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    except ImportError:
        pass
    
    return parser.parse_args()


# =============================================================================
# 主回放控制器
# =============================================================================

class ReplayController:
    """
    回放控制器
    
    整合数据加载、仿真运行、数据记录和可视化
    """
    
    def __init__(self, config: ReplayConfig, simulation_app):
        self.config = config
        self.simulation_app = simulation_app

        from data_utils import DataLoader, frames_to_arrays, TimeAligner, JointNameMapper
        from sim_runner import IsaacLabSimRunner, SimConfig, ActuatorConfig, records_to_arrays
        from plotter import JointPlotter, PlotConfig
        from joint_mapping import IsaacLabJointMapper, StateExtractor, ActionExtractor

        self.DataLoader = DataLoader
        self.frames_to_arrays = frames_to_arrays
        self.TimeAligner = TimeAligner
        self.JointNameMapper = JointNameMapper
        self.IsaacLabSimRunner = IsaacLabSimRunner
        self.SimConfig = SimConfig
        self.ActuatorConfig = ActuatorConfig
        self.records_to_arrays = records_to_arrays
        self.JointPlotter = JointPlotter
        self.PlotConfig = PlotConfig
        self.IsaacLabJointMapper = IsaacLabJointMapper
        self.StateExtractor = StateExtractor
        self.ActionExtractor = ActionExtractor

        # 组件
        self.data_loader = None
        self.sim_runner = None
        self.plotter = None
        
        # 数据
        self.frames = []
        self.source_arrays = {}
        self.sim_records = []
        self.aligned_data = {}
        
    def initialize(self):
        """初始化所有组件"""
        print("\n" + "=" * 70)
        print("[ReplayController] 初始化...")
        print("=" * 70)

        self.data_loader = self.DataLoader(self.config.dataset_path)

        sim_config = self.SimConfig(
            physics_dt=self.config.physics_dt,
            control_dt=self.config.control_dt,
            urdf_path=self.config.urdf_path,
            fix_base=self.config.fix_base,
            actuator=self.ActuatorConfig(
                head_stiffness=self.config.head_stiffness,
                head_damping=self.config.head_damping,
                arm_stiffness=self.config.arm_stiffness,
                arm_damping=self.config.arm_damping,
                gripper_stiffness=self.config.gripper_stiffness,
                gripper_damping=self.config.gripper_damping,
            ),
        )
        self.sim_runner = self.IsaacLabSimRunner(sim_config)

        self.sim_runner.initialize(headless=self.config.headless)
        self.plotter = self.JointPlotter(self.PlotConfig(dpi=self.config.viz_dpi))

        print("\n[ReplayController] 初始化完成!")
    
    def load_episode(self, episode_index: int):
        """加载单个 episode 数据"""
        print(f"\n[ReplayController] 加载 Episode {episode_index}...")
        
        self.frames = self.data_loader.load_episode(episode_index)
        
        # 应用帧范围
        start = self.config.start_frame
        end = self.config.end_frame if self.config.end_frame > 0 else len(self.frames)
        self.frames = self.frames[start:end]
        
        # 转换为数组
        self.source_arrays = self.frames_to_arrays(self.frames)
        
        print(f"[ReplayController] 加载 {len(self.frames)} 帧 (帧范围: {start} - {end})")
        
        # 打印数据摘要
        if self.config.verbose:
            print("\n[数据摘要]:")
            for k, v in self.source_arrays.items():
                print(f"  {k}: {v.shape}")
    
    def run_replay(self) -> List:
        """执行回放"""
        print("\n[ReplayController] 开始回放...")

        # 准备数据
        timestamps = self.source_arrays["timestamps"]
        target_positions = self.source_arrays["target_joint_pos"]
        target_gripper_l = self.source_arrays["target_gripper_l"]
        target_gripper_r = self.source_arrays["target_gripper_r"]

        self.sim_records = self.sim_runner.run_replay(
            target_positions=target_positions,
            target_grippers_l=target_gripper_l,
            target_grippers_r=target_gripper_r,
            timestamps=timestamps,
            simulation_app=self.simulation_app,
            initial_position=target_positions[0],
        )

        return self.sim_records
    
    def align_data(self):
        """对齐真实数据和仿真数据"""
        print("\n[ReplayController] 对齐数据...")

        # 获取源时间戳
        source_timestamps = self.source_arrays["timestamps"]

        # 单机器人模式
        sim_arrays = self.records_to_arrays(self.sim_records)
        sim_timestamps = sim_arrays["sim_time"]

        aligner = self.TimeAligner()

        if len(sim_timestamps) != len(source_timestamps):
            print(f"  [注意] 时间戳长度不匹配: sim={len(sim_timestamps)}, source={len(source_timestamps)}")
            print(f"  进行线性插值对齐...")

            aligned_sim_pos = aligner.interpolate_to_target_timestamps(
                sim_timestamps, sim_arrays["sim_joint_pos"], source_timestamps
            )
            aligned_sim_vel = aligner.interpolate_to_target_timestamps(
                sim_timestamps, sim_arrays["sim_joint_vel"], source_timestamps
            )
            aligned_sim_torque = aligner.interpolate_to_target_timestamps(
                sim_timestamps, sim_arrays["sim_joint_torque"], source_timestamps
            )
            aligned_sim_gripper_l = aligner.interpolate_to_target_timestamps(
                sim_timestamps, sim_arrays["sim_gripper_l"].reshape(-1, 1), source_timestamps
            ).flatten()
            aligned_sim_gripper_r = aligner.interpolate_to_target_timestamps(
                sim_timestamps, sim_arrays["sim_gripper_r"].reshape(-1, 1), source_timestamps
            ).flatten()
            if "sim_end_effector_pos" in sim_arrays:
                aligned_sim_ee_pos = aligner.interpolate_to_target_timestamps(
                    sim_timestamps, sim_arrays["sim_end_effector_pos"], source_timestamps
                )
            else:
                aligned_sim_ee_pos = None
        else:
            aligned_sim_pos = sim_arrays["sim_joint_pos"]
            aligned_sim_vel = sim_arrays["sim_joint_vel"]
            aligned_sim_torque = sim_arrays["sim_joint_torque"]
            aligned_sim_gripper_l = sim_arrays["sim_gripper_l"]
            aligned_sim_gripper_r = sim_arrays["sim_gripper_r"]
            aligned_sim_ee_pos = sim_arrays.get("sim_end_effector_pos")

        self.aligned_data = {
            "timestamps": source_timestamps,
            "real_joint_pos": self.source_arrays["real_joint_pos"],
            "target_joint_pos": self.source_arrays["target_joint_pos"],
            "sim_joint_pos": aligned_sim_pos,
            "sim_joint_vel": aligned_sim_vel,
            "sim_joint_torque": aligned_sim_torque,
            "real_gripper_l": self.source_arrays["real_gripper_l"],
            "real_gripper_r": self.source_arrays["real_gripper_r"],
            "target_gripper_l": self.source_arrays["target_gripper_l"],
            "target_gripper_r": self.source_arrays["target_gripper_r"],
            "sim_gripper_l": aligned_sim_gripper_l,
            "sim_gripper_r": aligned_sim_gripper_r,
            "real_end_effector_pos": self.source_arrays.get("real_end_effector_pos"),
            "target_end_effector_pos": self.source_arrays.get("target_end_effector_pos"),
            "sim_end_effector_pos": aligned_sim_ee_pos,
            "joint_names": self.JointNameMapper.get_joint_names(),
        }

        print(f"[ReplayController] 对齐完成，共 {len(source_timestamps)} 帧")
    def save_results(self, episode_index: int):
        """保存结果"""
        if not self.config.recording_enabled:
            return
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"episode_{episode_index:06d}_aligned.pkl"
        
        with open(output_file, 'wb') as f:
            pickle.dump(self.aligned_data, f)
        
        print(f"[ReplayController] 保存结果: {output_file}")
    
    def generate_plots(self, episode_index: int):
        """生成可视化图表"""
        if not self.config.viz_enabled:
            return

        print("\n[ReplayController] 生成可视化图表...")

        viz_dir = Path(self.config.viz_output_dir) / f"episode_{episode_index:06d}"
        viz_dir.mkdir(parents=True, exist_ok=True)

        timestamps = self.aligned_data["timestamps"]
        real_pos = self.aligned_data["real_joint_pos"]
        target_pos = self.aligned_data["target_joint_pos"]
        real_gripper_l = self.aligned_data["real_gripper_l"]
        real_gripper_r = self.aligned_data["real_gripper_r"]
        target_gripper_l = self.aligned_data["target_gripper_l"]
        target_gripper_r = self.aligned_data["target_gripper_r"]

        # 单机器人模式
        sim_pos = self.aligned_data["sim_joint_pos"]
        sim_vel = self.aligned_data["sim_joint_vel"]
        sim_torque = self.aligned_data["sim_joint_torque"]
        sim_gripper_l = self.aligned_data.get("sim_gripper_l")
        sim_gripper_r = self.aligned_data.get("sim_gripper_r")

        # 绘制所有关节
        self.plotter.plot_all_joints(
            timestamps=timestamps,
            real_pos=real_pos,
            target_pos=target_pos,
            sim_pos=sim_pos,
            sim_vel=sim_vel,
            sim_torque=sim_torque,
            use_degrees=self.config.use_degrees,
        )

        # 夹爪对比图
        self.plotter.plot_gripper_compare(
            timestamps=timestamps,
            real_gripper_l=real_gripper_l,
            target_gripper_l=target_gripper_l,
            real_gripper_r=real_gripper_r,
            target_gripper_r=target_gripper_r,
            sim_gripper_l=sim_gripper_l,
            sim_gripper_r=sim_gripper_r,
            output_dir=viz_dir,
            format=self.config.viz_format,
            dpi=self.config.viz_dpi,
        )

        # 绘制组概览
        for group in self.config.plot_groups:
            self.plotter.plot_group_overview(
                group_name=group,
                timestamps=timestamps,
                real_pos=real_pos,
                target_pos=target_pos,
                sim_pos=sim_pos,
                use_degrees=self.config.use_degrees,
            )
        
        # 绘制误差汇总
        self.plotter.plot_error_summary(
            timestamps=timestamps,
            real_pos=real_pos,
            sim_pos=sim_pos,
            use_degrees=self.config.use_degrees,
        )
        
        # 保存
        self.plotter.save_all(viz_dir, format=self.config.viz_format, dpi=self.config.viz_dpi)
        self.plotter.close_all()
        
        print(f"[ReplayController] 图表保存至: {viz_dir}")
    
    def run_episode(self, episode_index: int):
        """运行单个 episode 的完整流程"""
        print("\n" + "=" * 70)
        print(f"[ReplayController] 处理 Episode {episode_index}")
        print("=" * 70)
        
        # 1. 加载数据
        self.load_episode(episode_index)
        
        # 2. 执行回放
        self.run_replay()
        
        # 3. 对齐数据
        self.align_data()
        
        # 4. 保存结果
        self.save_results(episode_index)
        
        # 5. 生成可视化
        self.generate_plots(episode_index)
        
        print(f"\n[ReplayController] Episode {episode_index} 处理完成!")
    
    def run(self):
        """运行主循环"""
        episode_arg = self.config.episode_index
        
        if episode_arg == "all":
            # 运行所有 episode
            total_episodes = self.data_loader.get_episode_count()
            print(f"[ReplayController] 将处理所有 {total_episodes} 个 episode")
            
            for ep_idx in range(total_episodes):
                if not self.simulation_app.is_running():
                    print("[ReplayController] 仿真被中断")
                    break
                self.run_episode(ep_idx)
        else:
            # 运行单个 episode
            ep_idx = int(episode_arg)
            self.run_episode(ep_idx)
    
    def close(self):
        """清理资源"""
        if self.sim_runner:
            self.sim_runner.close()
        if self.plotter:
            self.plotter.close_all()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 启动 IsaacSim
    print("\n" + "=" * 70)
    print("AGIBOT G1 Isaac Lab Action Replay")
    print("=" * 70)
    
    try:
        from isaaclab.app import AppLauncher
    except ImportError as e:
        print(f"[Error] 无法导入 IsaacLab: {e}")
        print("请确保 IsaacLab 已正确安装并配置路径")
        sys.exit(1)
    
    # 创建 AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    try:
        # 加载配置
        config_path = Path(args.config)
        if config_path.exists():
            print(f"[Main] 加载配置: {config_path}")
            config = ReplayConfig.from_yaml(str(config_path))
        else:
            print(f"[Main] 配置文件不存在，使用默认配置")
            config = ReplayConfig.default()
        
        # 覆盖命令行参数
        config.episode_index = args.episode
        if args.headless:
            config.headless = True
        if args.no_viz:
            config.viz_enabled = False
        if args.output_dir:
            config.output_dir = args.output_dir
            config.viz_output_dir = str(Path(args.output_dir) / "plots")

        # 创建控制器
        controller = ReplayController(config, simulation_app)
        
        # 初始化
        controller.initialize()
        
        # 运行
        controller.run()
        
        print("\n" + "=" * 70)
        print("[Main] 回放完成!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n[Main] 用户中断")
    except Exception as e:
        print(f"\n[Main] 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        if 'controller' in locals():
            controller.close()
        simulation_app.close()


if __name__ == "__main__":
    main()