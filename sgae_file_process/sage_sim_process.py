"""
sage_sim_process.py
处理sim机器人数据
输入数据：control.csv, state_motor.csv, joint_list.txt, （无 event.csv）
输出数据：压缩 NPZ 文件，包含joint_sequence, sim_dof_positions, sim_dof_positions_cmd, sim_dof_velocities, sim_dof_torques, motion_name

"""

import argparse
import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd
from pathlib import Path
from typing import List

SKIP_JOINT_NAMES = ["left_hand_joint", "right_hand_joint"]

class SimDataProcessor:
    """Process simulated robot log data."""
    def __init__(self, motion_dir: str):
        self.file_path = motion_dir.replace("\\", "/").rstrip("/")
        self.joint_list = self._load_joint_list()
        self.data = self._load_robot_data(["control", "state_motor"])

    def _load_joint_list(self):
        joint_list_path = Path(self.file_path) / "joint_list.txt"
        if joint_list_path.is_file():
            with open(joint_list_path, "r", encoding="utf-8") as file:
                return tuple(line.strip().split("/")[-1] for line in file.readlines() if line.strip())
        raise FileNotFoundError(f"No joint_list.txt under {self.file_path}")

    def _load_robot_data(self, data_types):
        robot_data = {}
        initial_time = float("inf")

        for data_type in data_types:
            csv_path = f"{self.file_path}/{data_type}.csv"
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV not found: {csv_path}")

            data = pd.read_csv(csv_path)
            initial_time = min(initial_time, float(data["timestamp"].iloc[0]))
            robot_data[data_type] = data

        for data in robot_data.values():
            data["time_since_zero"] = data["timestamp"] - initial_time

        dof_command = self._process_dof_data(robot_data["control"], self.joint_list, ["positions"])
        dof_state = self._process_dof_data(
            robot_data["state_motor"], self.joint_list, ["positions", "velocities", "torques"]
        )
        return {"dof_command": dof_command, "dof_state": dof_state}

    @property
    def dof_command(self):
        return self.data["dof_command"]

    @property
    def dof_state(self):
        return self.data["dof_state"]

    def _preprocess_string(self, value):
        pattern = r"\b([a-zA-Z_][a-zA-Z_0-9]*)\b"
        return re.sub(pattern, r"'\1'", value)

    def _safe_str_to_list(self, value):
        try:
            return ast.literal_eval(self._preprocess_string(value))
        except (ValueError, SyntaxError):
            return []

    def _process_dof_data(self, df, joint_list, keys=["positions"]):
        df_copy = df.copy()
        for k in keys:
            df_copy[k] = df_copy[k].apply(self._safe_str_to_list)
            key_df = pd.DataFrame(index=df.index)
            for i, j in enumerate(joint_list):
                key_df[f"{k}_{j}"] = df_copy[k].apply(lambda x, idx=i: x[idx] if idx < len(x) else 0.0)
            df_copy = pd.concat([df_copy.drop(k, axis=1), key_df], axis=1)
        return df_copy

    @property
    def joints(self):
        return self.joint_list


def discover_sim_motion_dirs(root_dir: str) -> List[str]:
    """Recursively find simulated motion directories (event.csv not required)."""
    need_files = {"control.csv", "state_motor.csv", "joint_list.txt"}
    motion_dirs = []
    root_dir = os.path.abspath(root_dir)

    for dirpath, _, filenames in os.walk(root_dir):
        if need_files.issubset(set(filenames)):
            motion_dirs.append(dirpath)

    motion_dirs.sort()
    print(f"Found {len(motion_dirs)} simulated motion directories (root: {root_dir})")
    return motion_dirs


def interp_to(target_t: np.ndarray, src_t: np.ndarray, src_v: np.ndarray) -> np.ndarray:
    """Linear interpolation."""
    if len(src_t) < 2:
        return np.full_like(target_t, src_v[0] if len(src_v) > 0 else 0.0)
    return np.interp(target_t, src_t, src_v)


def get_relative_output_path(motion_dir: str, root_dir: str) -> str:
    rel_path = os.path.relpath(motion_dir, root_dir)
    return rel_path.replace("\\", "/")


def process_sim_data(
    root_dir: str,
    output_dir: str = "output_files",
    plots_dir: str = "plots",
    if_plot: bool = True,
    dt: float = 0.02,
):
    """
    处理仿真机器人数据 (无 event.csv，使用全部数据)
    Args:
        root_dir: 数据根目录
        output_dir: NPZ 输出目录
        plots_dir: 图片输出目录
        if_plot: 是否生成图片
        dt: 重采样时间间隔 (默认 0.02s = 50Hz)
    """
    motion_dirs = discover_sim_motion_dirs(root_dir)
    if not motion_dirs:
        print("No valid simulated motion directory found.")
        return

    skip_joints = set(SKIP_JOINT_NAMES)
    root_dir = os.path.abspath(root_dir)

    for idx, motion_dir in enumerate(motion_dirs, start=1):
        rel_path = get_relative_output_path(motion_dir, root_dir)
        print(f"\n[{idx}/{len(motion_dirs)}] Processing: {rel_path}")

        try:
            proc = SimDataProcessor(motion_dir)
            df_state = proc.dof_state
            df_cmd = proc.dof_command

            t_state = df_state["time_since_zero"].to_numpy(dtype=float)
            t_cmd = df_cmd["time_since_zero"].to_numpy(dtype=float)

            if len(t_state) < 2:
                print("  Skip: too few samples")
                continue

            t_max = t_state[-1]
            timestamp = np.arange(0, t_max, dt, dtype=float)
            if len(timestamp) == 0:
                print("  Skip: time series too short")
                continue

            joints = [j for j in proc.joints if j not in skip_joints] or list(proc.joints)
            pos_list, pos_cmd_list, vel_list, tau_list = [], [], [], []

            for jn in joints:
                key_pos, key_vel, key_tau = f"positions_{jn}", f"velocities_{jn}", f"torques_{jn}"
                if key_pos not in df_state.columns:
                    continue

                pos_state = interp_to(timestamp, t_state, df_state[key_pos].to_numpy())
                pos_cmd = interp_to(timestamp, t_cmd, df_cmd[f"positions_{jn}"].to_numpy())
                vel_state = interp_to(timestamp, t_state, df_state[key_vel].to_numpy())
                tau_state = interp_to(timestamp, t_state, df_state[key_tau].to_numpy())

                pos_list.append(pos_state)
                pos_cmd_list.append(pos_cmd)
                vel_list.append(vel_state)
                tau_list.append(tau_state)

                if if_plot:
                    joint_plot_dir = os.path.join(plots_dir, rel_path, jn)
                    os.makedirs(joint_plot_dir, exist_ok=True)

                    plt.figure(figsize=(9, 3))
                    plt.plot(timestamp, pos_state, label="position")
                    plt.plot(timestamp, pos_cmd, "--", alpha=0.8, label="position_cmd")
                    plt.title(f"{jn} - position vs cmd (50Hz)")
                    plt.xlabel("time (s)")
                    plt.ylabel("rad")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(joint_plot_dir, "positions.png"))
                    plt.close()

                    plt.figure(figsize=(9, 3))
                    plt.plot(timestamp, vel_state, color="tab:blue")
                    plt.title(f"{jn} - velocity (50Hz)")
                    plt.xlabel("time (s)")
                    plt.ylabel("rad/s")
                    plt.tight_layout()
                    plt.savefig(os.path.join(joint_plot_dir, "velocity.png"))
                    plt.close()

                    plt.figure(figsize=(9, 3))
                    plt.plot(timestamp, tau_state, color="tab:orange")
                    plt.title(f"{jn} - torque (50Hz)")
                    plt.xlabel("time (s)")
                    plt.ylabel("Nm")
                    plt.tight_layout()
                    plt.savefig(os.path.join(joint_plot_dir, "torque.png"))
                    plt.close()

            if not pos_list:
                print("  Skip: no valid joint data")
                continue

            # Save NPZ
            npz_out_dir = os.path.join(output_dir, rel_path)
            os.makedirs(npz_out_dir, exist_ok=True)
            npz_path = os.path.join(npz_out_dir, "motor_all_joints.npz")

            np.savez_compressed(
                npz_path,
                sim_dof_positions=np.stack(pos_list, axis=0),
                sim_dof_positions_cmd=np.stack(pos_cmd_list, axis=0),
                sim_dof_velocities=np.stack(vel_list, axis=0),
                sim_dof_torques=np.stack(tau_list, axis=0),
                joint_sequence=np.array(joints, dtype=object),
                motion_name=os.path.basename(motion_dir),
            )
            print(f"  Saved: {npz_path}")

        except Exception as e:
            print(f"  Failed: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="处理真实机器人数据 (递归查找)")
    p.add_argument("--root", type=str, required=True, help="数据根目录")
    p.add_argument("--out", type=str, default="output_files", help="NPZ 输出目录")
    p.add_argument("--plots", type=str, default="plots", help="图片输出目录")
    p.add_argument("--no-plot", action="store_true", help="禁用绘图")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_sim_data(
        root_dir=args.root,
        output_dir=args.out,
        plots_dir=args.plots,
        if_plot=not args.no_plot,
    )