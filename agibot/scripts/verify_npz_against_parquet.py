"""
Verify converted .npz against source parquet data and GapONet format.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_utils import DataLoader, frames_to_arrays, JointNameMapper


REQUIRED_KEYS = [
    "joint_sequence",
    "payloads",
]


def parse_episode_indices(episode_arg: str, total_episodes: int) -> List[int]:
    arg = episode_arg.strip().lower()
    if arg == "all":
        return list(range(total_episodes))

    indices: List[int] = []
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                start, end = end, start
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    return sorted({i for i in indices if 0 <= i < total_episodes})


def compute_velocity(positions: np.ndarray, timestamps: np.ndarray, fallback_dt: float) -> np.ndarray:
    if positions.shape[0] < 2:
        return np.zeros_like(positions, dtype=np.float32)
    timestamps = timestamps.astype(np.float64)
    if timestamps.shape[0] != positions.shape[0]:
        raise ValueError("timestamps length does not match positions length")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        return np.gradient(positions, fallback_dt, axis=0).astype(np.float32)
    return np.gradient(positions, timestamps, axis=0).astype(np.float32)


def check_gaponet_format(data: np.lib.npyio.NpzFile, expected_dofs: int) -> List[str]:
    errors: List[str] = []
    for key in REQUIRED_KEYS:
        if key not in data.files:
            errors.append(f"missing key: {key}")

    has_object = all(
        key in data.files
        for key in [
            "real_dof_positions",
            "real_dof_velocities",
            "real_dof_positions_cmd",
            "real_dof_torques",
        ]
    )
    has_dense = all(
        key in data.files
        for key in [
            "real_dof_positions_padded",
            "real_dof_velocities_padded",
            "real_dof_positions_cmd_padded",
            "real_dof_torques_padded",
            "motion_len",
        ]
    )
    if not has_object and not has_dense:
        errors.append("missing motion arrays: provide object or dense padded keys")

    if "joint_sequence" in data.files:
        joint_seq = np.array(data["joint_sequence"], dtype=object)
        if joint_seq.shape[0] != expected_dofs:
            errors.append(f"joint_sequence length mismatch: {joint_seq.shape[0]} != {expected_dofs}")

    if "payloads" in data.files:
        payloads = np.array(data["payloads"])
        if payloads.ndim != 1:
            errors.append("payloads must be 1D array")

    if has_object:
        for key in [
            "real_dof_positions",
            "real_dof_velocities",
            "real_dof_positions_cmd",
            "real_dof_torques",
        ]:
            arr = np.array(data[key], dtype=object)
            if arr.ndim != 1:
                errors.append(f"{key} should be an object array of motions")
                continue
            for idx, motion in enumerate(arr):
                if not isinstance(motion, np.ndarray):
                    errors.append(f"{key}[{idx}] is not ndarray")
                    continue
                if motion.ndim != 2 or motion.shape[1] != expected_dofs:
                    errors.append(f"{key}[{idx}] shape invalid: {motion.shape}")
    if has_dense:
        for key in [
            "real_dof_positions_padded",
            "real_dof_velocities_padded",
            "real_dof_positions_cmd_padded",
            "real_dof_torques_padded",
        ]:
            arr = np.array(data[key])
            if arr.ndim != 3 or arr.shape[2] != expected_dofs:
                errors.append(f"{key} should be [N,T,D], got {arr.shape}")
        motion_len = np.array(data["motion_len"])
        if motion_len.ndim != 1:
            errors.append("motion_len must be 1D array")

    return errors


def get_motion_arrays(data: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (real, vel, target, torque, motion_len) arrays."""
    if "real_dof_positions_padded" in data.files:
        real = np.array(data["real_dof_positions_padded"])
        vel = np.array(data["real_dof_velocities_padded"])
        target = np.array(data["real_dof_positions_cmd_padded"])
        torque = np.array(data["real_dof_torques_padded"])
        motion_len = np.array(data["motion_len"])
        return real, vel, target, torque, motion_len

    motions_real = np.array(data["real_dof_positions"], dtype=object)
    motions_vel = np.array(data["real_dof_velocities"], dtype=object)
    motions_target = np.array(data["real_dof_positions_cmd"], dtype=object)
    motions_torque = np.array(data["real_dof_torques"], dtype=object)
    motion_len = np.array([m.shape[0] for m in motions_real], dtype=np.int64)
    return motions_real, motions_vel, motions_target, motions_torque, motion_len


def compare_episode(
    ep_index: int,
    arrays: dict,
    npz_entry: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fallback_dt: float,
    atol: float,
) -> List[str]:
    errors: List[str] = []
    real_pos = arrays["real_joint_pos"].astype(np.float32)
    target_pos = arrays["target_joint_pos"].astype(np.float32)
    timestamps = arrays["timestamps"].astype(np.float64)

    npz_real, npz_vel, npz_target, npz_torque = npz_entry
    if npz_real.shape != real_pos.shape:
        errors.append(f"Episode {ep_index}: real shape mismatch {npz_real.shape} vs {real_pos.shape}")
        return errors
    if npz_target.shape != target_pos.shape:
        errors.append(f"Episode {ep_index}: target shape mismatch {npz_target.shape} vs {target_pos.shape}")
        return errors

    if not np.allclose(npz_real, real_pos, atol=atol, rtol=0):
        max_diff = np.max(np.abs(npz_real - real_pos))
        errors.append(f"Episode {ep_index}: real max diff {max_diff:.6g} > {atol}")
    if not np.allclose(npz_target, target_pos, atol=atol, rtol=0):
        max_diff = np.max(np.abs(npz_target - target_pos))
        errors.append(f"Episode {ep_index}: target max diff {max_diff:.6g} > {atol}")

    recomputed_vel = compute_velocity(real_pos, timestamps, fallback_dt=fallback_dt)
    if npz_vel.shape != recomputed_vel.shape:
        errors.append(f"Episode {ep_index}: vel shape mismatch {npz_vel.shape} vs {recomputed_vel.shape}")
    elif not np.allclose(npz_vel, recomputed_vel, atol=atol, rtol=0):
        max_diff = np.max(np.abs(npz_vel - recomputed_vel))
        errors.append(f"Episode {ep_index}: vel max diff {max_diff:.6g} > {atol}")

    if npz_torque.shape != real_pos.shape:
        errors.append(f"Episode {ep_index}: torque shape mismatch {npz_torque.shape} vs {real_pos.shape}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify .npz against parquet dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset root, e.g. /home/user/agibot/data/H3_example")
    parser.add_argument("--npz", type=str, required=True, help="Path to motion_agibot.npz, e.g. /home/user/agibot/data/H3_example/motion_agibot.npz")
    parser.add_argument("--episodes", type=str, default="all", help="Episode indices used to build the npz")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for numeric compare")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    npz_path = Path(args.npz)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ path not found: {npz_path}")

    loader = DataLoader(str(dataset_path))
    episode_indices = parse_episode_indices(args.episodes, loader.get_episode_count())
    if not episode_indices:
        raise ValueError(f"No valid episodes parsed from '{args.episodes}'")

    joint_names = JointNameMapper.get_joint_names()
    num_dofs = len(joint_names)

    data = np.load(npz_path, allow_pickle=True)
    format_errors = check_gaponet_format(data, expected_dofs=num_dofs)
    if format_errors:
        print("[Format] FAILED")
        for err in format_errors:
            print(f"  - {err}")
        return 1
    print("[Format] OK")

    motions_real, motions_vel, motions_target, motions_torque, motion_len = get_motion_arrays(data)

    if len(motions_real) < len(episode_indices):
        print(f"[Compare] NPZ motions count {len(motions_real)} < episodes {len(episode_indices)}")
        return 1

    errors: List[str] = []
    for idx, ep in enumerate(episode_indices):
        frames = loader.load_episode(ep)
        arrays = frames_to_arrays(frames)
        if motions_real.ndim == 3:
            length = int(motion_len[idx])
            npz_entry = (
                motions_real[idx, :length, :],
                motions_vel[idx, :length, :],
                motions_target[idx, :length, :],
                motions_torque[idx, :length, :],
            )
        else:
            npz_entry = (motions_real[idx], motions_vel[idx], motions_target[idx], motions_torque[idx])
        errors.extend(
            compare_episode(
                ep_index=ep,
                arrays=arrays,
                npz_entry=npz_entry,
                fallback_dt=1.0 / loader.config.fps,
                atol=args.atol,
            )
        )

    if errors:
        print("[Compare] FAILED")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("[Compare] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
