"""
*** 这个脚本还没有完成 ***

AGIBOT G1motion 数据回放验证脚本

功能:
1. 验证回放轨迹与原始数据的一致性
2. 可视化对比
3. 生成验证报告

使用方法:
    python verify_replay.py --episode 0
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib未安装,将跳过可视化")

from data_loader import AgibotDataLoader
from joint_mapping import G1JointMapping


def compute_trajectory_error(
    original: np.ndarray, 
    replayed: np.ndarray
) -> Dict[str, float]:
    """
    计算轨迹误差指标
    
    Args:
        original: 原始轨迹 (N, D)
        replayed: 回放轨迹 (N, D)
        
    Returns:
        误差指标字典
    """
    if original.shape != replayed.shape:
        raise ValueError(f"形状不匹配: {original.shape} vs {replayed.shape}")
    
    # 逐帧误差
    frame_errors = np.linalg.norm(original - replayed, axis=1)
    
    # 逐关节误差
    joint_errors = np.abs(original - replayed).mean(axis=0)
    
    return {
        "mae": float(np.mean(np.abs(original - replayed))),  # 平均绝对误差
        "rmse": float(np.sqrt(np.mean((original - replayed) ** 2))),  # 均方根误差
        "max_error": float(np.max(np.abs(original - replayed))),  # 最大误差
        "frame_error_mean": float(np.mean(frame_errors)),
        "frame_error_max": float(np.max(frame_errors)),
        "per_joint_mae": joint_errors.tolist(),
    }


def plot_trajectory_comparison(
    timestamps: np.ndarray,
    original: np.ndarray,
    replayed: np.ndarray,
    joint_names: List[str],
    save_path: str = None
):
    """
    绘制轨迹对比图
    
    Args:
        timestamps: 时间戳 (N,)
        original: 原始轨迹 (N, D)
        replayed: 回放轨迹 (N, D) 或 None
        joint_names: 关节名称列表
        save_path: 保存路径
    """
    if not HAS_MATPLOTLIB:
        print("跳过可视化: matplotlib未安装")
        return
    
    n_joints = original.shape[1]
    n_cols = 2
    n_rows = (n_joints + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 2 * n_rows))
    axes = axes.flatten()
    
    for i in range(n_joints):
        ax = axes[i]
        ax.plot(timestamps, original[:, i], 'b-', label='Original', alpha=0.8)
        if replayed is not None:
            ax.plot(timestamps, replayed[:, i], 'r--', label='Replayed', alpha=0.8)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.set_title(joint_names[i] if i < len(joint_names) else f'Joint {i}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(n_joints, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_joint_statistics(
    loader: AgibotDataLoader,
    episode_index: int,
    save_path: str = None
):
    """
    绘制关节统计信息
    """
    if not HAS_MATPLOTLIB:
        return
    
    timestamps, joint_traj = loader.get_joint_trajectory(episode_index)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. 关节角度分布
    ax = axes[0, 0]
    ax.boxplot(joint_traj, labels=[f'J{i}' for i in range(14)])
    ax.set_xlabel('Joint Index')
    ax.set_ylabel('Position (rad)')
    ax.set_title('Joint Position Distribution')
    ax.grid(True, alpha=0.3)
    
    # 2. 关节速度
    ax = axes[0, 1]
    dt = np.diff(timestamps)
    velocities = np.diff(joint_traj, axis=0) / dt[:, np.newaxis]
    ax.boxplot(velocities, labels=[f'J{i}' for i in range(14)])
    ax.set_xlabel('Joint Index')
    ax.set_ylabel('Velocity (rad/s)')
    ax.set_title('Joint Velocity Distribution')
    ax.grid(True, alpha=0.3)
    
    # 3. 轨迹时序 (左臂)
    ax = axes[1, 0]
    for i in range(7):
        ax.plot(timestamps, joint_traj[:, i], label=f'L{i+1}', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (rad)')
    ax.set_title('Left Arm Trajectory')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. 轨迹时序 (右臂)
    ax = axes[1, 1]
    for i in range(7, 14):
        ax.plot(timestamps, joint_traj[:, i], label=f'R{i-6}', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (rad)')
    ax.set_title('Right Arm Trajectory')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"统计图已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def verify_data_integrity(loader: AgibotDataLoader) -> Dict:
    """
    验证数据完整性
    """
    results = {
        "total_episodes": loader.info.total_episodes,
        "total_frames": loader.info.total_frames,
        "fps": loader.info.fps,
        "episodes_checked": 0,
        "frames_checked": 0,
        "errors": [],
    }
    
    # 检查每个episode
    for ep_info in loader.episodes:
        try:
            frames = loader.load_episode(ep_info.episode_index)
            
            # 验证帧数
            if len(frames) != ep_info.length:
                results["errors"].append(
                    f"Episode {ep_info.episode_index}: 帧数不匹配 "
                    f"(expected {ep_info.length}, got {len(frames)})"
                )
            
            # 验证数据维度
            for i, frame in enumerate(frames[:3]):  # 只检查前3帧
                if frame.observation_state.shape[0] != 94:
                    results["errors"].append(
                        f"Episode {ep_info.episode_index}, Frame {i}: "
                        f"state维度错误 ({frame.observation_state.shape[0]} != 94)"
                    )
                if frame.action.shape[0] != 36:
                    results["errors"].append(
                        f"Episode {ep_info.episode_index}, Frame {i}: "
                        f"action维度错误 ({frame.action.shape[0]} != 36)"
                    )
            
            results["episodes_checked"] += 1
            results["frames_checked"] += len(frames)
            
        except Exception as e:
            results["errors"].append(f"Episode {ep_info.episode_index}: 加载失败 - {e}")
    
    return results


def generate_verification_report(
    loader: AgibotDataLoader,
    episode_index: int,
    output_dir: str
) -> str:
    """
    生成完整的验证报告
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_lines = [
        "# AGIBOT G1motion 数据验证报告",
        "",
        f"生成时间: {__import__('datetime').datetime.now().isoformat()}",
        "",
        "## 1. 数据集信息",
        "",
        f"- 机器人类型: {loader.info.robot_type}",
        f"- 总Episodes: {loader.info.total_episodes}",
        f"- 总帧数: {loader.info.total_frames}",
        f"- 采集帧率: {loader.info.fps} FPS",
        "",
    ]
    
    # 数据完整性检查
    report_lines.extend([
        "## 2. 数据完整性检查",
        "",
    ])
    
    integrity = verify_data_integrity(loader)
    report_lines.append(f"- 已检查Episodes: {integrity['episodes_checked']}")
    report_lines.append(f"- 已检查帧数: {integrity['frames_checked']}")
    
    if integrity["errors"]:
        report_lines.append(f"- 发现错误: {len(integrity['errors'])}")
        for err in integrity["errors"][:10]:  # 最多显示10条
            report_lines.append(f"  - {err}")
    else:
        report_lines.append("- ✅ 未发现错误")
    
    report_lines.append("")
    
    # Episode详情
    report_lines.extend([
        f"## 3. Episode {episode_index} 详情",
        "",
    ])
    
    timestamps, joint_traj = loader.get_joint_trajectory(episode_index)
    ep_info = loader.get_episode_info(episode_index)
    
    report_lines.extend([
        f"- 帧数: {len(timestamps)}",
        f"- 时长: {timestamps[-1]:.2f} 秒",
        f"- 任务: {ep_info.tasks}",
        "",
        "### 关节角度统计 (弧度)",
        "",
        "| 关节 | 最小值 | 最大值 | 均值 | 标准差 |",
        "|------|--------|--------|------|--------|",
    ])
    
    mapping = G1JointMapping()
    joint_names = mapping.get_arm_joints()
    
    for i, name in enumerate(joint_names):
        col = joint_traj[:, i]
        report_lines.append(
            f"| {name} | {col.min():.4f} | {col.max():.4f} | "
            f"{col.mean():.4f} | {col.std():.4f} |"
        )
    
    # 保存报告
    report_path = output_path / "verification_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    # 生成可视化
    if HAS_MATPLOTLIB:
        plot_joint_statistics(loader, episode_index, 
                             str(output_path / "joint_statistics.png"))
        
        plot_trajectory_comparison(
            timestamps, joint_traj, None, joint_names,
            str(output_path / "trajectory_plot.png")
        )
    
    print(f"\n验证报告已生成: {report_path}")
    return str(report_path)


def main():
    parser = argparse.ArgumentParser(description="验证回放数据")
    parser.add_argument("--episode", type=int, default=0, help="Episode序号")
    parser.add_argument("--data_path", type=str,
                        default="/home/jianing/Desktop/yuntian/agibot/Bus_table_example/H3_example",
                        help="数据集路径")
    parser.add_argument("--output_dir", type=str,
                        default="/home/jianing/Desktop/yuntian/agibot/replay_isaaclab/verification",
                        help="输出目录")
    parser.add_argument("--check_all", action="store_true", 
                        help="检查所有episodes")
    
    args = parser.parse_args()
    
    print("="*60)
    print("AGIBOT G1motion 数据验证工具")
    print("="*60)
    
    # 加载数据
    loader = AgibotDataLoader(args.data_path)
    loader.load_meta()
    
    # 生成报告
    generate_verification_report(loader, args.episode, args.output_dir)
    
    print("\n验证完成!")


if __name__ == "__main__":
    main()
