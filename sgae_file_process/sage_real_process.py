"""
sage_real_process.py
处理真实机器人数据
输入数据：control.csv, state_motor.csv, joint_list.txt, event.csv
输出数据：压缩 NPZ 文件，包含joint_sequence, real_dof_positions, real_dof_positions_cmd, real_dof_velocities, real_dof_torques, sim_dof_positions

"""

import argparse
import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from pathlib import Path
import re
import pandas as pd
from typing import List, Tuple

SKIP_JOINT_NAMES = [] # 处理时跳过关节


class RobotDataProcessor:
    """Read and parse CSV logs under a single motion directory."""

    def __init__(self, motion_dir: str, is_simulation: bool = False):
        self.file_path = motion_dir.replace("\\", "/").rstrip("/")
        self.is_simulation = is_simulation
        self.use_radians = True
        self.use_seconds = is_simulation  # Sim uses seconds; real uses microseconds

        self.joint_list = self._load_joint_list()
        self.data = self._load_robot_data(["control", "state_motor"])

    def _load_joint_list(self):
        """Load joint list from joint_list.txt under the motion directory."""
        joint_list_path = Path(self.file_path) / "joint_list.txt"
        if joint_list_path.is_file():
            with open(joint_list_path, "r", encoding="utf-8") as file:
                return tuple(line.strip().split("/")[-1] for line in file.readlines() if line.strip())
        raise FileNotFoundError(f"No joint_list.txt under {self.file_path}")

    def _load_robot_data(self, data_types):
        """Load CSVs and expand list-like columns into per-joint columns."""
        robot_data = {}
        initial_time = float("inf")

        for data_type in data_types:
            csv_path = f"{self.file_path}/{data_type}.csv"
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV not found: {csv_path}")

            data = pd.read_csv(csv_path)
            if not self.use_seconds:
                data["timestamp"] = data["timestamp"] / 1e6

            initial_time = min(initial_time, float(data["timestamp"].iloc[0]))
            robot_data[data_type] = data

        for data in robot_data.values():
            data["time_since_zero"] = data["timestamp"] - initial_time
            data["time_since_last"] = data["timestamp"].diff()

        dof_command = self._process_dof_data(robot_data["control"], self.joint_list, ["positions"])
        dof_state = self._process_dof_data(
            robot_data["state_motor"], self.joint_list, ["positions", "velocities", "torques"]
        )

        return {"raw_data": robot_data, "dof_command": dof_command, "dof_state": dof_state}

    @property
    def real_event(self):
        event_path = f"{self.file_path}/event.csv"
        if not os.path.exists(event_path):
            return {}
        event = pd.read_csv(event_path)
        return dict(zip(event["event"], event["timestamp"] / 1e6))

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
            preprocessed_value = self._preprocess_string(value)
            return ast.literal_eval(preprocessed_value)
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


def discover_motion_dirs(root_dir: str, need_event: bool = True) -> List[str]:
    need_files = {"control.csv", "state_motor.csv", "joint_list.txt"}
    if need_event:
        need_files.add("event.csv")

    motion_dirs = []
    root_dir = os.path.abspath(root_dir)

    for dirpath, _, filenames in os.walk(root_dir):
        files = set(filenames)
        if need_files.issubset(files):
            motion_dirs.append(dirpath)

    motion_dirs.sort()
    print(f"Found {len(motion_dirs)} motion directories (root: {root_dir})")
    return motion_dirs


def clip_by_events(df: pd.DataFrame, start_t: float, end_t: float) -> pd.DataFrame:
    """Clip data to [start_t, end_t] based on event timestamps."""
    df_adjusted = df.copy()
    for col in ("timestamp", "time_since_zero"):
        if col in df_adjusted.columns:
            df_adjusted[col] = df_adjusted[col] - start_t

    df_adjusted = df_adjusted[df_adjusted["time_since_zero"] >= 0]
    max_t = end_t - start_t
    df_adjusted = df_adjusted[df_adjusted["time_since_zero"] <= max_t]
    return df_adjusted.reset_index(drop=True)


def resample_series(df: pd.DataFrame, key: str, timestamp: np.ndarray) -> np.ndarray:
    """Linearly interpolate df[key] onto the target timestamp array."""
    return np.interp(timestamp, df["time_since_zero"].to_numpy(), df[key].to_numpy())


def get_relative_output_path(motion_dir: str, root_dir: str) -> str:
    """Get a normalized path of motion_dir relative to root_dir."""
    rel_path = os.path.relpath(motion_dir, root_dir)
    return rel_path.replace("\\", "/")


def process_real_data(
    root_dir: str,
    output_dir: str = "output_files",
    plots_dir: str = "plots",
    if_plot: bool = True,
    dt: float = 0.02,
):
    """
    Process real robot logs.

    Args:
        root_dir: 数据根目录，递归查找所有动作
        output_dir: NPZ 输出目录
        plots_dir: 图片输出目录
        if_plot: 是否生成图片
        dt: 重采样时间间隔 (默认 0.02s = 50Hz)
    """
    motion_dirs = discover_motion_dirs(root_dir, need_event=True)
    if not motion_dirs:
        print("No valid motion directory found. Please check your data layout.")
        return

    skip_joints = set(SKIP_JOINT_NAMES)
    root_dir = os.path.abspath(root_dir)

    for idx, motion_dir in enumerate(motion_dirs, start=1):
        rel_path = get_relative_output_path(motion_dir, root_dir)
        print(f"\n[{idx}/{len(motion_dirs)}] Processing: {rel_path}")

        try:
            proc = RobotDataProcessor(motion_dir, is_simulation=False)
            ev = proc.real_event

            if "MOTION_START" not in ev or "DISABLE" not in ev:
                print("  Skip: missing MOTION_START or DISABLE event")
                continue

            start_t, end_t = float(ev["MOTION_START"]), float(ev["DISABLE"])
            df_state = clip_by_events(proc.dof_state, start_t, end_t)
            df_cmd = clip_by_events(proc.dof_command, start_t, end_t)

            if len(df_state) < 2:
                print("  Skip: too few samples")
                continue

            t_max = float(df_state["time_since_zero"].max())
            timestamp = np.arange(dt, t_max, dt, dtype=float)
            if len(timestamp) == 0:
                print("  Skip: time series too short")
                continue

            joints = [j for j in proc.joints if j not in skip_joints] or list(proc.joints)
            pos_list, pos_cmd_list, vel_list, tau_list = [], [], [], []

            for jn in joints:
                key_pos, key_vel, key_tau = f"positions_{jn}", f"velocities_{jn}", f"torques_{jn}"
                if key_pos not in df_state.columns:
                    continue

                pos_state = resample_series(df_state, key_pos, timestamp)
                pos_cmd = resample_series(df_cmd, f"positions_{jn}", timestamp)
                vel_state = resample_series(df_state, key_vel, timestamp)
                tau_state = resample_series(df_state, key_tau, timestamp)

                pos_list.append(pos_state)
                pos_cmd_list.append(pos_cmd)
                vel_list.append(vel_state)
                tau_list.append(tau_state)

                if if_plot:
                    joint_plot_dir = os.path.join(plots_dir, rel_path, jn)
                    os.makedirs(joint_plot_dir, exist_ok=True)

                    # Position
                    plt.figure(figsize=(9, 3))
                    plt.plot(timestamp, pos_state, label="position")
                    plt.plot(timestamp, pos_cmd, "--", alpha=0.8, label="position_cmd")
                    plt.title(f"{jn} - position vs cmd")
                    plt.xlabel("time (s)"); plt.ylabel("rad")
                    plt.legend(); plt.tight_layout()
                    plt.savefig(os.path.join(joint_plot_dir, "positions.png"))
                    plt.close()

                    # Velocity
                    plt.figure(figsize=(9, 3))
                    plt.plot(timestamp, vel_state, color="tab:blue")
                    plt.title(f"{jn} - velocity")
                    plt.xlabel("time (s)"); plt.ylabel("rad/s")
                    plt.tight_layout()
                    plt.savefig(os.path.join(joint_plot_dir, "velocity.png"))
                    plt.close()

                    # Torque
                    plt.figure(figsize=(9, 3))
                    plt.plot(timestamp, tau_state, color="tab:orange")
                    plt.title(f"{jn} - torque")
                    plt.xlabel("time (s)"); plt.ylabel("Nm")
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
                real_dof_positions=np.stack(pos_list, axis=0),
                real_dof_positions_cmd=np.stack(pos_cmd_list, axis=0),
                real_dof_velocities=np.stack(vel_list, axis=0),
                real_dof_torques=np.stack(tau_list, axis=0),
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
    process_real_data(
        root_dir=args.root,
        output_dir=args.out,
        plots_dir=args.plots,
        if_plot=not args.no_plot,
    )