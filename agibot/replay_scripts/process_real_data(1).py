import argparse
import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from pathlib import Path
from scipy import signal
from scipy.spatial import distance
import re
import pandas as pd
import shutil

HAND_JOINT_NAMES = ["left_hand_joint", "right_hand_joint"]  # Will filter


class RobotDataProcessor:
    """Main class for processing robot data"""

    def __init__(self, robot_name, motion_source, motion_name, is_simulation=True, file_root=None):
        self.robot_name = robot_name
        self.motion_source = motion_source
        self.motion_name = motion_name
        self.file_path = f"{file_root}/{motion_source}/{motion_name}"
        self.is_simulation = is_simulation

        # Set data format
        if is_simulation:
            self.use_radians = True
            self.use_seconds = True
        else:
            self.use_radians = True
            self.use_seconds = False

        # Load all the joints from the humanoid config file
        self.joint_config = self._load_joint_config()
        # Load the joints of interest for the motion
        self.joint_list = self._load_joint_list()
        self.data = self._load_robot_data(["control", "state_motor"])

    def _load_joint_config(self):
        """Load joint list from yaml configuration file"""
        base_dir = Path(__file__).parent / "configs"
        primary = base_dir / f"{self.robot_name}_joints.yaml"
        fallback = base_dir / "h1_2_joints.yaml"  # Use existing general configuration

        config_path = primary if primary.is_file() else fallback
        if not config_path.is_file():
            raise FileNotFoundError(f"Joint config not found: {primary} nor fallback: {fallback}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config or "joints" not in config:
            raise ValueError(f"Invalid joints config: {config_path}")

        return tuple(config["joints"])

    def _load_joint_list(self):
        """Load joint list from file or default configuration and apply mask if provided"""
        joint_list_path = Path(f"{self.file_path}/joint_list.txt")

        if joint_list_path.is_file():
            with open(joint_list_path) as file:
                return tuple(line.strip().split("/")[-1] for line in file.readlines())
        else:
            return tuple(joint.strip().split("/")[-1] for joint in self.joint_config)

    def _load_robot_data(self, data_types):
        """Load and process robot data from files"""
        robot_data = {}
        initial_time = float("inf")

        # Read data for each type
        for data_type in data_types:
            data = pd.read_csv(f"{self.file_path}/{data_type}.csv")

            # Convert time units if needed
            if not self.use_seconds:
                data["timestamp"] = data["timestamp"] / 1e6

            initial_time = min(initial_time, data["timestamp"][0])
            robot_data[data_type] = data

        # Calculate relative timestamps
        for data in robot_data.values():
            data["time_since_zero"] = data["timestamp"] - initial_time
            data["time_since_last"] = data["timestamp"].diff()

        # Process DOF data
        dof_command = self._process_dof_data(robot_data["control"], self.joint_list, ["positions"], self.use_radians)
        dof_state = self._process_dof_data(
            robot_data["state_motor"], self.joint_list, ["positions", "velocities", "torques"], self.use_radians
        )

        return {"raw_data": robot_data, "dof_command": dof_command, "dof_state": dof_state}

    @property
    def real_event(self):
        event = pd.read_csv(f"{self.file_path}/event.csv")
        return dict(zip(event["event"], event["timestamp"] / 1e6))

    @property
    def raw_data(self):
        return self.data["raw_data"]

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
            return "Invalid list string"

    def _safe_deg2rad(self, x, i):
        if pd.api.types.is_number(x[i]):
            return np.deg2rad(x[i])
        return x[i]

    def _process_dof_data(self, df, joint_list, keys=["positions"], is_rad=False):
        df_copy = df.copy()

        for k in keys:
            df_copy[k] = df_copy[k].apply(self._safe_str_to_list)

            key_df = pd.DataFrame(index=df.index)
            for i, j in enumerate(joint_list):
                if not is_rad and k in ["positions", "velocities"]:
                    key_df[f"{k}_{j}"] = df_copy[k].apply(lambda x: self._safe_deg2rad(x, i))
                else:
                    key_df[f"{k}_{j}"] = df_copy[k].apply(lambda x: x[i])

            df_copy = pd.concat([df_copy.drop(k, axis=1), key_df], axis=1)
        return df_copy

    @property
    def joints(self):
        return self.joint_list


class SkipMotionError(Exception):
    """Exception for marking motions that need to be skipped"""
    pass


class RobotDataComparator:
    """
    Compare and visualize real robot data (this task only uses real data)
    """

    def __init__(self, robot_name, motion_source, motion_names, valid_joints_file, result_folder: str, sample_dt: int):
        """
        Args:
            robot_name (str): 子目录名（如 'act0_25'）
            motion_source (str): 同 robot_name，形成路径 motions/<robot>/<motion>
            motion_names (str): "*" 或子串过滤
            valid_joints_file (str|None): 可选的关节掩码文件
            result_folder (str): 数据根目录（motions）
            sample_dt (int|None): 可选兜底 dt（本实现按频率自动设定）
        """

        self._robot_name = robot_name
        self._motion_source = motion_source
        self._result_folder = result_folder
        self._valid_joints_list = self._load_valid_joints(valid_joints_file, self._robot_name)
        self._valid_joints_list = [path.split("/")[-1] for path in self._valid_joints_list]

        self._use_all_joints = (len(self._valid_joints_list) == 0)

        # Search robot directory, return motion relative paths and frequencies
        self._motion_names, self._motion_freqs = self._process_motion_names(
            motion_names, self._result_folder, self._robot_name, self._motion_source
        )
        if len(self._motion_names) == 0:
            raise ValueError("Can't get any motion_name")

        self._sample_dt = sample_dt

        # Lazy loading cache: load data for each motion on demand
        self._motion_cache: dict[str, RobotDataProcessor] = {}

    def _get_real_data(self, motion_name: str) -> RobotDataProcessor:
        """Load and cache data processor for a motion on demand"""
        if motion_name not in self._motion_cache:
            self._motion_cache[motion_name] = RobotDataProcessor(
                self._robot_name,
                self._robot_name,          # motions/<robot>/<motion_name>
                motion_name,
                is_simulation=False,
                file_root=f"{self._result_folder}",
            )
        return self._motion_cache[motion_name]

    def _get_joints_for_motion(self, motion_name):
        sim_data = self._get_real_data(motion_name)

        if self._use_all_joints:
            selected_joints = sim_data.joints
            print(f"Using all {len(selected_joints)} joints for motion {motion_name}")
        else:
            sim_joints = sim_data.joints
            selected_joints = list(set(self._valid_joints_list).intersection(set(sim_joints)))
            if not selected_joints:
                selected_joints = sim_joints
                print(f"Using all {len(selected_joints)} joints for motion {motion_name} (mask had no common joints)")
            else:
                print(f"Using {len(selected_joints)} joints from mask intersection for motion {motion_name}")

        return [j for j in selected_joints if j not in HAND_JOINT_NAMES]

    def _load_valid_joints(self, valid_joints_file=None, robot_name=None):
        """
        Load valid joints file
        """
        if valid_joints_file and os.path.exists(valid_joints_file):
            with open(valid_joints_file, "r") as file:
                return [line.strip() for line in file.readlines()]

        default_valid_joints_file = f"configs/{robot_name}_valid_joints.txt"
        if os.path.exists(default_valid_joints_file):
            with open(default_valid_joints_file, "r") as file:
                return [line.strip() for line in file.readlines()]

        print(f"Warning: No joints mask file found for '{robot_name}'. Using all joints.")
        return []

    def _process_motion_names(self, motion_names_arg, result_folder, robot_name, motion_source):
        """
        Recursively search all motion folders under motions/<robot_name> (directories containing csv),
        and identify 50Hz/100Hz frequency from directory names.
        Returns:
            motion_names: List of relative paths like "<motion_dir>/<timestamp_dir>"
            motion_freqs: Dict with motion_name as key and 50 or 100 as value
        """
        base_dir = os.path.join(result_folder, robot_name)
        if not os.path.exists(base_dir):
            raise ValueError(f"Real results directory does not exist: {base_dir}")

        need_files = {"control.csv", "event.csv", "state_motor.csv"}
        motion_names = []
        motion_freqs = {}

        # Simple substring filter (can be "*" or specific substring)
        filter_token = None if (motion_names_arg is None or motion_names_arg == "*") else motion_names_arg.strip()

        hz_pattern = re.compile(r"(\d+)\s*Hz", re.IGNORECASE)

        for dirpath, dirnames, filenames in os.walk(base_dir):
            files = set(filenames)
            if need_files.issubset(files):
                rel = os.path.relpath(dirpath, base_dir).replace("\\", "/")
                if filter_token and filter_token not in rel:
                    continue

                # Look for 50Hz/100Hz in path segments (assumed to be always found)
                freq = None
                for part in rel.split("/"):
                    m = hz_pattern.search(part)
                    if m:
                        cand = int(m.group(1))
                        if cand in (50, 100):
                            freq = cand
                            break
                if freq is None:
                    raise ValueError(f"Frequency not found in path: {rel}")

                motion_names.append(rel)
                motion_freqs[rel] = freq

        print(f"Found {len(motion_names)} motions for robot '{robot_name}'. All with detected frequency.")
        return motion_names, motion_freqs

    def adjust_real_data_timing(self, dataframe, start_delay, end_time):
        """Adjust timestamps of real robot data"""
        df_adjusted = dataframe.copy()

        # Adjust timestamps
        for time_col in ["timestamp", "time_since_zero"]:
            df_adjusted[time_col] -= start_delay

        if time_col in df_adjusted.columns:
            mask = (df_adjusted[time_col] <= 0)
            if mask.any():
                first_non_positive_index = mask.idxmax()
                df_adjusted = df_adjusted.iloc[first_non_positive_index:].reset_index(drop=True)

        # Filter time range
        df_filtered = df_adjusted[df_adjusted["time_since_zero"] >= 0]
        max_time = end_time - start_delay

        return df_filtered[df_filtered["time_since_zero"] <= max_time]

    def _resample_waveform(self, df: pd.DataFrame, key: str, timestamp: np.ndarray) -> pd.DataFrame:
        """Resample a waveform to a new timestamp array using linear interpolation"""
        return pd.DataFrame({"timestamp": timestamp, "value": np.interp(timestamp, df["time_since_zero"], df[key])})

    def align_data(self, motion_name, joint_name, data_type="positions"):
        """Align real data for comparison

        Args:
            motion_name: Name of the motion
            joint_name: Name of the joint
            data_type: Type of data to align (positions, velocities, torques)

        Returns:
            tuple: (aligned_real[, aligned_real_cmd]) DataFrames with aligned timestamps
        """
        real_data = self._get_real_data(motion_name)
        ev = real_data.real_event  # Event name dictionary (e.g., contains 'MOTION_START', 'DISABLE')
        # Strictly require key events to exist; otherwise throw skip exception
        if "MOTION_START" not in ev or "DISABLE" not in ev:
            missing = []
            if "MOTION_START" not in ev:
                missing.append("MOTION_START")
            if "DISABLE" not in ev:
                missing.append("DISABLE")
            raise SkipMotionError(f"Missing events: {', '.join(missing)}")
        real_delay = ev["MOTION_START"]
        real_end = ev["DISABLE"]

        real_df = self.adjust_real_data_timing(real_data.dof_state, real_delay, real_end)
        real_cmd = self.adjust_real_data_timing(real_data.dof_command, real_delay, real_end)

        key = f"{data_type}_{joint_name}"

        # Use sampling interval selected by frequency parsed from this motion
        dt = 1.0 / float(self._motion_freqs[motion_name])
        max_time = real_df["time_since_zero"].max()
        timestamp = np.arange(dt, max_time, dt)

        aligned_real = self._resample_waveform(real_df, key, timestamp)
        if data_type == "positions":
            aligned_real_cmd = self._resample_waveform(real_cmd, key, timestamp)
            return aligned_real, aligned_real_cmd
        else:
            return aligned_real

    def analyze_all_data(self, output_dir, if_plot=True):
        """Output npz by motion and generate separate images for each joint (images saved to root plots/ directory)"""
        # Create output root directory
        if os.path.exists(output_dir):
            if not os.path.isdir(output_dir):
                raise ValueError(f"The provided output_dir '{output_dir}' exists but is not a directory.")
        else:
            os.makedirs(output_dir, exist_ok=True)

        # Progress bar helper
        total_motions = len(self._motion_names)
        print(f"Detected {total_motions} motions in total.")

        def _print_progress(i: int, n: int, width: int = 30):
            i = max(0, min(i, n))
            filled = int(width * (i / n)) if n else width
            bar = "█" * filled + "-" * (width - filled)
            print(f"\rProcessing progress [{bar}] {i}/{n}", end="", flush=True)

        _print_progress(0, total_motions)

        # Target joints (all joints in joint_dict)
        target_joints = [
            "left_shoulder_pitch_joint", "left_shoulder_yaw_joint", "left_shoulder_roll_joint",
            "left_elbow_joint", "right_shoulder_pitch_joint", "right_shoulder_yaw_joint",
            "right_shoulder_roll_joint", "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint"
        ]

        # Process each motion separately
        for idx, motion_name in enumerate(self._motion_names, start=1):
            print(f"\n[{idx}/{total_motions}] Processing motion: {motion_name}")
            freq = self._motion_freqs[motion_name]
            safe_motion = motion_name.replace("/", "_")

            # npz output directory: output_files/<robot>/<motion_frequency>/
            motion_out_dir = os.path.join(output_dir, f"{safe_motion}_{freq}Hz")
            os.makedirs(motion_out_dir, exist_ok=True)

            # Image output directory: plots/<robot>/<motion_frequency>/ (parallel to output_files)
            plot_out_dir = os.path.join("plots", self._robot_name, f"{safe_motion}_{freq}Hz")
            os.makedirs(plot_out_dir, exist_ok=True)

            try:
                # Filter: only keep target joints that actually exist in the data (load on demand)
                joints_available = self._get_real_data(motion_name).joints
                joints = [j for j in target_joints if j in joints_available]
                if not joints:
                    print(f"  Warning: no target joints found in motion {motion_name}, skip.")
                    raise SkipMotionError("no target joints available")

                # Collect all three types of data for all joints
                pos_list, pos_cmd_list, vel_list, tau_list = [], [], [], []
                for joint_name in joints:
                    # positions + command
                    aligned_real_pos, aligned_real_pos_cmd = self.align_data(motion_name, joint_name, "positions")
                    pos_vals = aligned_real_pos.iloc[:, 1].to_numpy()
                    pos_cmd_vals = aligned_real_pos_cmd.iloc[:, 1].to_numpy()

                    # velocities
                    aligned_real_vel = self.align_data(motion_name, joint_name, "velocities")
                    vel_vals = aligned_real_vel.iloc[:, 1].to_numpy()

                    # torques
                    aligned_real_tau = self.align_data(motion_name, joint_name, "torques")
                    tau_vals = aligned_real_tau.iloc[:, 1].to_numpy()

                    pos_list.append(pos_vals)
                    pos_cmd_list.append(pos_cmd_vals)
                    vel_list.append(vel_vals)
                    tau_list.append(tau_vals)

                # After aligning time axis, can directly stack
                real_dof_positions = np.stack(pos_list, axis=0)
                real_dof_positions_cmd = np.stack(pos_cmd_list, axis=0)
                real_dof_velocities = np.stack(vel_list, axis=0)
                real_dof_torques = np.stack(tau_list, axis=0)
                joint_sequence = np.array(joints, dtype=object)

                # Generate separate images for each joint (save to plots/<robot>/<motion_frequency>/)
                if if_plot:
                    x = np.arange(real_dof_positions.shape[1])
                    for i, jn in enumerate(joints):
                        fig, axs = plt.subplots(3, 1, figsize=(10, 10))

                        axs[0].plot(x, real_dof_positions[i], label='position')
                        axs[0].plot(x, real_dof_positions_cmd[i], label='position command', linestyle='--', alpha=0.7)
                        axs[0].set_title(f'{jn} - position (solid) & position command (dashed)')
                        axs[0].legend()
                        axs[0].set_xlabel('Index'); axs[0].set_ylabel('Value')

                        axs[1].plot(x, real_dof_velocities[i], color='tab:blue', label='velocity')
                        axs[1].set_title(f'{jn} - velocity'); axs[1].set_xlabel('Index'); axs[1].set_ylabel('Value')

                        axs[2].plot(x, real_dof_torques[i], color='tab:blue', label='torque')
                        axs[2].set_title(f'{jn} - torque'); axs[2].set_xlabel('Index'); axs[2].set_ylabel('Value')

                        plt.tight_layout()
                        plt.savefig(os.path.join(plot_out_dir, f"{jn}.png"))
                        plt.close()

                # Save npz for this motion (only under output_files)
                npz_file_path = os.path.join(motion_out_dir, "motor_all_joints.npz")
                np.savez_compressed(
                    npz_file_path,
                    real_dof_positions=real_dof_positions,
                    real_dof_positions_cmd=real_dof_positions_cmd,
                    real_dof_velocities=real_dof_velocities,
                    real_dof_torques=real_dof_torques,
                    joint_sequence=joint_sequence,
                    motion_name=motion_name,
                    frequency=freq,
                )
                print(f"Saved npz to {npz_file_path}")
            except SkipMotionError as e:
                print(f"Skip motion due to missing events: {e}")
                # Delete created directories for this motion (npz/plots)
                if os.path.isdir(motion_out_dir):
                    shutil.rmtree(motion_out_dir, ignore_errors=True)
                if os.path.isdir(plot_out_dir):
                    shutil.rmtree(plot_out_dir, ignore_errors=True)
                _print_progress(idx, total_motions)
                continue

            # Update progress bar
            _print_progress(idx, total_motions)

        print()  # New line, end progress bar line


if __name__ == "__main__":
    # Batch processing mode: traverse all subdirectories under motions (each is a robot data directory)
    motions_root = "motions"
    robots = [d for d in os.listdir(motions_root) if os.path.isdir(os.path.join(motions_root, d))]

    for robot_name in robots:
        print(f"== Processing robot: {robot_name} ==")
        data_comparator = RobotDataComparator(
            robot_name=robot_name,      # Subdirectory name
            motion_source=robot_name,   # Make path motions/<robot>/<motion>
            motion_names="*",           # All motions under this robot
            valid_joints_file=None,     # Don't use external mask
            result_folder=motions_root, # Data root directory
            sample_dt=None,             # Auto resample each motion with 50/100Hz
        )
        # Output: output_files/<robot>/<motion_name_frequency>/
        output_root = Path(f"output_files/{robot_name}")
        data_comparator.analyze_all_data(output_root, if_plot=True)