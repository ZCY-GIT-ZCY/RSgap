"""
AGIBOT G1 数据可视化模块
========================
功能:
1. 生成关节位置/速度/力矩对比图
2. 支持批量导出 PNG/PDF
3. 计算跟踪误差统计

图表类型:
- 位置对比图: Real (观测) vs Target (目标) vs Sim (仿真)
- 速度对比图: Real vs Sim
- 力矩对比图: Sim (仅仿真)
- 误差统计图: RMSE, 最大误差等
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import pickle


# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class PlotConfig:
    """绘图配置"""
    figsize: Tuple[int, int] = (14, 10)
    dpi: int = 150
    
    # 颜色方案
    color_real: str = "#2196F3"      # 蓝色 - 真实数据
    color_target: str = "#FF9800"    # 橙色 - 目标数据
    color_sim: str = "#4CAF50"       # 绿色 - 仿真数据
    color_error: str = "#F44336"     # 红色 - 误差
    
    # 线条样式
    linewidth: float = 1.5
    alpha_secondary: float = 0.7
    
    # 标签
    label_real: str = "Real (Observation)"
    label_target: str = "Target (Action)"
    label_sim: str = "Sim (Replay)"


# =============================================================================
# 误差计算
# =============================================================================

@dataclass
class ErrorMetrics:
    """误差指标"""
    rmse: float          # 均方根误差
    mae: float           # 平均绝对误差
    max_error: float     # 最大误差
    std: float           # 标准差
    
    @classmethod
    def compute(cls, predicted: np.ndarray, actual: np.ndarray) -> "ErrorMetrics":
        """计算误差指标"""
        error = predicted - actual
        return cls(
            rmse=np.sqrt(np.mean(error ** 2)),
            mae=np.mean(np.abs(error)),
            max_error=np.max(np.abs(error)),
            std=np.std(error),
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "max_error": self.max_error,
            "std": self.std,
        }


# =============================================================================
# 关节绘图器
# =============================================================================

class JointPlotter:
    """
    关节数据绘图器
    
    用法:
        plotter = JointPlotter(config)
        plotter.plot_single_joint(...)
        plotter.plot_all_joints(...)
        plotter.save_all(output_dir)
    """
    
    # 16维关节名称
    JOINT_NAMES = [
        "head_yaw", "head_pitch",
        "arm_l_j1", "arm_l_j2", "arm_l_j3", "arm_l_j4",
        "arm_l_j5", "arm_l_j6", "arm_l_j7",
        "arm_r_j1", "arm_r_j2", "arm_r_j3", "arm_r_j4",
        "arm_r_j5", "arm_r_j6", "arm_r_j7",
    ]
    
    # 关节分组 (16维: 腰部已固定)
    JOINT_GROUPS = {
        "head": (0, 2),
        "arm_l": (2, 9),
        "arm_r": (9, 16),
    }
    
    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()
        self.figures: Dict[str, Figure] = {}
        self.error_metrics: Dict[str, ErrorMetrics] = {}
    
    def plot_single_joint(
        self,
        joint_idx: int,
        timestamps: np.ndarray,
        real_pos: np.ndarray,
        target_pos: np.ndarray,
        sim_pos: np.ndarray,
        sim_vel: Optional[np.ndarray] = None,
        sim_torque: Optional[np.ndarray] = None,
        real_vel: Optional[np.ndarray] = None,
        use_degrees: bool = True,
    ) -> Figure:
        """
        绘制单个关节的完整对比图
        
        Args:
            joint_idx: 关节索引 (0-17)
            timestamps: 时间戳 (N,)
            real_pos: 真实位置 (N,)
            target_pos: 目标位置 (N,)
            sim_pos: 仿真位置 (N,)
            sim_vel: 仿真速度 (N,) 可选
            sim_torque: 仿真力矩 (N,) 可选
            real_vel: 真实速度 (N,) 可选
            use_degrees: 是否转换为角度单位
            
        Returns:
            Figure 对象
        """
        cfg = self.config
        joint_name = self.JOINT_NAMES[joint_idx] if joint_idx < len(self.JOINT_NAMES) else f"joint_{joint_idx}"
        
        # 单位转换
        scale = np.degrees(1.0) if use_degrees else 1.0
        unit = "deg" if use_degrees else "rad"
        vel_unit = "deg/s" if use_degrees else "rad/s"
        
        # 确定子图数量
        n_rows = 1
        if sim_vel is not None:
            n_rows += 1
        if sim_torque is not None:
            n_rows += 1
        
        fig, axes = plt.subplots(n_rows, 1, figsize=cfg.figsize, sharex=True)
        if n_rows == 1:
            axes = [axes]
        
        fig.suptitle(f"Joint: {joint_name}", fontsize=14, fontweight='bold')
        
        # ---- 位置对比图 ----
        ax_pos = axes[0]
        ax_pos.plot(timestamps, real_pos * scale, 
                   color=cfg.color_real, linewidth=cfg.linewidth,
                   label=cfg.label_real)
        ax_pos.plot(timestamps, target_pos * scale,
                   color=cfg.color_target, linewidth=cfg.linewidth,
                   alpha=cfg.alpha_secondary, linestyle='--',
                   label=cfg.label_target)
        ax_pos.plot(timestamps, sim_pos * scale,
                   color=cfg.color_sim, linewidth=cfg.linewidth,
                   alpha=cfg.alpha_secondary,
                   label=cfg.label_sim)
        
        ax_pos.set_ylabel(f"Position ({unit})")
        ax_pos.legend(loc='upper right')
        ax_pos.grid(True, alpha=0.3)
        ax_pos.set_title("Position Comparison")
        
        # 计算误差指标
        metrics_sim = ErrorMetrics.compute(sim_pos * scale, real_pos * scale)
        metrics_target = ErrorMetrics.compute(target_pos * scale, real_pos * scale)
        
        self.error_metrics[f"{joint_name}_sim"] = metrics_sim
        self.error_metrics[f"{joint_name}_target"] = metrics_target
        
        # 显示误差统计
        ax_pos.text(0.02, 0.98, 
                   f"RMSE (Sim-Real): {metrics_sim.rmse:.3f} {unit}\n"
                   f"RMSE (Target-Real): {metrics_target.rmse:.3f} {unit}\n"
                   f"Max Error (Sim): {metrics_sim.max_error:.3f} {unit}",
                   transform=ax_pos.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ---- 速度对比图 ----
        row = 1
        if sim_vel is not None:
            ax_vel = axes[row]
            ax_vel.plot(timestamps, sim_vel * scale,
                       color=cfg.color_sim, linewidth=cfg.linewidth,
                       label="Sim Velocity")
            if real_vel is not None:
                ax_vel.plot(timestamps, real_vel * scale,
                           color=cfg.color_real, linewidth=cfg.linewidth,
                           alpha=cfg.alpha_secondary,
                           label="Real Velocity")
            ax_vel.set_ylabel(f"Velocity ({vel_unit})")
            ax_vel.legend(loc='upper right')
            ax_vel.grid(True, alpha=0.3)
            ax_vel.set_title("Velocity")
            row += 1
        
        # ---- 力矩图 ----
        if sim_torque is not None:
            ax_torque = axes[row]
            ax_torque.plot(timestamps, sim_torque,
                          color=cfg.color_sim, linewidth=cfg.linewidth,
                          label="Sim Torque")
            ax_torque.set_ylabel("Torque (Nm)")
            ax_torque.legend(loc='upper right')
            ax_torque.grid(True, alpha=0.3)
            ax_torque.set_title("Applied Torque")
        
        axes[-1].set_xlabel("Time (s)")
        
        plt.tight_layout()
        
        self.figures[joint_name] = fig
        return fig
    
    def plot_all_joints(
        self,
        timestamps: np.ndarray,
        real_pos: np.ndarray,
        target_pos: np.ndarray,
        sim_pos: np.ndarray,
        sim_vel: Optional[np.ndarray] = None,
        sim_torque: Optional[np.ndarray] = None,
        use_degrees: bool = True,
        joints_to_plot: Optional[List[int]] = None,
    ) -> Dict[str, Figure]:
        """
        绘制所有关节
        
        Args:
            timestamps: (N,)
            real_pos: (N, 16)
            target_pos: (N, 16)
            sim_pos: (N, 16)
            sim_vel: (N, 16) 可选
            sim_torque: (N, 16) 可选
            use_degrees: 是否使用角度单位
            joints_to_plot: 要绘制的关节索引列表，None 表示全部
            
        Returns:
            关节名 -> Figure 的字典
        """
        if joints_to_plot is None:
            joints_to_plot = list(range(real_pos.shape[1]))
        
        for joint_idx in joints_to_plot:
            self.plot_single_joint(
                joint_idx=joint_idx,
                timestamps=timestamps,
                real_pos=real_pos[:, joint_idx],
                target_pos=target_pos[:, joint_idx],
                sim_pos=sim_pos[:, joint_idx],
                sim_vel=sim_vel[:, joint_idx] if sim_vel is not None else None,
                sim_torque=sim_torque[:, joint_idx] if sim_torque is not None else None,
                use_degrees=use_degrees,
            )
        
        return self.figures
    
    def plot_group_overview(
        self,
        group_name: str,
        timestamps: np.ndarray,
        real_pos: np.ndarray,
        target_pos: np.ndarray,
        sim_pos: np.ndarray,
        use_degrees: bool = True,
    ) -> Figure:
        """
        绘制关节组概览图 (多关节在一个图上)
        
        Args:
            group_name: 组名 ("waist", "head", "arm_l", "arm_r")
            其他参数同上
        """
        if group_name not in self.JOINT_GROUPS:
            raise ValueError(f"未知关节组: {group_name}")
        
        start_idx, end_idx = self.JOINT_GROUPS[group_name]
        n_joints = end_idx - start_idx
        
        scale = np.degrees(1.0) if use_degrees else 1.0
        unit = "deg" if use_degrees else "rad"
        
        fig, axes = plt.subplots(n_joints, 1, figsize=(14, 3 * n_joints), sharex=True)
        if n_joints == 1:
            axes = [axes]
        
        fig.suptitle(f"Joint Group: {group_name}", fontsize=14, fontweight='bold')
        
        for i, joint_idx in enumerate(range(start_idx, end_idx)):
            ax = axes[i]
            joint_name = self.JOINT_NAMES[joint_idx]
            
            ax.plot(timestamps, real_pos[:, joint_idx] * scale,
                   color=self.config.color_real, linewidth=1.2, label="Real")
            ax.plot(timestamps, target_pos[:, joint_idx] * scale,
                   color=self.config.color_target, linewidth=1.2, 
                   alpha=0.7, linestyle='--', label="Target")
            ax.plot(timestamps, sim_pos[:, joint_idx] * scale,
                   color=self.config.color_sim, linewidth=1.2,
                   alpha=0.7, label="Sim")
            
            ax.set_ylabel(f"{joint_name}\n({unit})")
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(loc='upper right', ncol=3)
        
        axes[-1].set_xlabel("Time (s)")
        
        plt.tight_layout()
        
        self.figures[f"group_{group_name}"] = fig
        return fig
    
    def plot_error_summary(
        self,
        timestamps: np.ndarray,
        real_pos: np.ndarray,
        sim_pos: np.ndarray,
        use_degrees: bool = True,
    ) -> Figure:
        """
        绘制误差汇总图
        
        显示所有关节的跟踪误差统计
        """
        scale = np.degrees(1.0) if use_degrees else 1.0
        unit = "deg" if use_degrees else "rad"
        
        n_joints = real_pos.shape[1]
        
        # 计算每个关节的误差
        rmse_values = []
        max_errors = []
        
        for j in range(n_joints):
            error = (sim_pos[:, j] - real_pos[:, j]) * scale
            rmse_values.append(np.sqrt(np.mean(error ** 2)))
            max_errors.append(np.max(np.abs(error)))
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        x = np.arange(n_joints)
        
        # RMSE 条形图
        ax1 = axes[0]
        bars1 = ax1.bar(x, rmse_values, color=self.config.color_sim, alpha=0.8)
        ax1.set_ylabel(f"RMSE ({unit})")
        ax1.set_title("RMSE per Joint")
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.JOINT_NAMES[:n_joints], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars1, rmse_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 最大误差条形图
        ax2 = axes[1]
        bars2 = ax2.bar(x, max_errors, color=self.config.color_error, alpha=0.8)
        ax2.set_ylabel(f"Max Error ({unit})")
        ax2.set_title("Maximum Error per Joint")
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.JOINT_NAMES[:n_joints], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars2, max_errors):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        self.figures["error_summary"] = fig
        return fig
    
    def save_all(
        self,
        output_dir: Union[str, Path],
        format: str = "png",
        dpi: Optional[int] = None,
    ):
        """
        保存所有图表
        
        Args:
            output_dir: 输出目录
            format: 图片格式 (png, pdf, svg)
            dpi: DPI 设置
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dpi = dpi or self.config.dpi
        
        for name, fig in self.figures.items():
            filename = output_dir / f"{name}.{format}"
            fig.savefig(filename, format=format, dpi=dpi, bbox_inches='tight')
            print(f"[Plotter] 保存: {filename}")
        
        # 保存误差统计
        if self.error_metrics:
            metrics_file = output_dir / "error_metrics.pkl"
            metrics_dict = {k: v.to_dict() for k, v in self.error_metrics.items()}
            with open(metrics_file, 'wb') as f:
                pickle.dump(metrics_dict, f)
            print(f"[Plotter] 保存误差指标: {metrics_file}")
    
    def close_all(self):
        """关闭所有图表"""
        for fig in self.figures.values():
            plt.close(fig)
        self.figures.clear()
    
    def plot_gripper_compare(
        self,
        timestamps: np.ndarray,
        real_gripper_l: np.ndarray,
        target_gripper_l: np.ndarray,
        real_gripper_r: np.ndarray,
        target_gripper_r: np.ndarray,
        sim_gripper_l: np.ndarray = None,
        sim_gripper_r: np.ndarray = None,
        output_dir: str = "outputs/plots",
        format: str = "png",
        dpi: int = 150,
    ):
        """绘制左右夹爪的开合程度对比图"""
        import matplotlib.pyplot as plt
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for side, real, target, sim in [
            ("left", real_gripper_l, target_gripper_l, sim_gripper_l),
            ("right", real_gripper_r, target_gripper_r, sim_gripper_r),
        ]:
            plt.figure(figsize=(12, 4))
            plt.plot(timestamps, real, label="Real (Obs)", color="#2196F3")
            plt.plot(timestamps, target, label="Target (Action)", color="#FF9800", linestyle="--")
            if sim is not None:
                plt.plot(timestamps, sim, label="Sim", color="#4CAF50", linestyle=":")
            plt.title(f"Gripper {side.capitalize()} Opening (0=Open, 120=Closed)")
            plt.xlabel("Time (s)")
            plt.ylabel("Gripper Value")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            fname = output_dir / f"gripper_{side}.{format}"
            plt.savefig(fname, format=format, dpi=dpi)
            plt.close()
            print(f"[Plotter] 保存: {fname}")


# =============================================================================
# 便捷函数
# =============================================================================

def quick_plot(
    timestamps: np.ndarray,
    real_pos: np.ndarray,
    target_pos: np.ndarray,
    sim_pos: np.ndarray,
    output_dir: str = "outputs/plots",
    **kwargs
):
    """
    快速生成对比图
    
    Args:
        timestamps: (N,)
        real_pos: (N, 16)
        target_pos: (N, 16)
        sim_pos: (N, 16)
        output_dir: 输出目录
        **kwargs: 传递给 plot_all_joints
    """
    plotter = JointPlotter()
    
    # 绘制所有关节
    plotter.plot_all_joints(timestamps, real_pos, target_pos, sim_pos, **kwargs)
    
    # 绘制组概览 (腰部已固定，不再绘制)
    for group in ["head", "arm_l", "arm_r"]:
        plotter.plot_group_overview(group, timestamps, real_pos, target_pos, sim_pos)
    
    # 绘制误差汇总
    plotter.plot_error_summary(timestamps, real_pos, sim_pos)
    
    # 保存
    plotter.save_all(output_dir)
    plotter.close_all()
    
    print(f"\n[Plotter] 完成! 图表保存至: {output_dir}")


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(42)
    n_frames = 300
    n_joints = 16
    
    timestamps = np.linspace(0, 10, n_frames)
    
    # 模拟轨迹
    real_pos = np.sin(timestamps[:, None] * np.linspace(0.5, 2, n_joints)) * 0.5
    target_pos = real_pos + np.random.randn(n_frames, n_joints) * 0.01
    sim_pos = real_pos + np.random.randn(n_frames, n_joints) * 0.05 + 0.02
    
    # 绘图测试
    plotter = JointPlotter()
    
    # 单关节
    plotter.plot_single_joint(
        joint_idx=4,
        timestamps=timestamps,
        real_pos=real_pos[:, 4],
        target_pos=target_pos[:, 4],
        sim_pos=sim_pos[:, 4],
        sim_vel=np.gradient(sim_pos[:, 4], timestamps),
    )
    
    # 误差汇总
    plotter.plot_error_summary(timestamps, real_pos, sim_pos)
    
    plt.show()
