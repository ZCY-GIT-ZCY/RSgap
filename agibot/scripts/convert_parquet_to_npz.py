"""
Convert AGIBOT parquet episodes into GapONet motion .npz.

Outputs keys expected by GapONet motion loaders:
- real_dof_positions
- real_dof_velocities
- real_dof_positions_cmd
- real_dof_torques
- joint_sequence
- payloads
- joint_names (optional but included)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

# Ensure local imports work when executed from different cwd.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse AGIBOT data extraction logic to avoid mismatches.
from data_utils import DataLoader, frames_to_arrays, JointNameMapper


def parse_episode_indices(episode_arg: str, total_episodes: int) -> List[int]:
    """Parse episode index string like 'all', '0,1,2', '0-10'."""
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

    # Remove duplicates and out-of-range values
    indices = sorted({i for i in indices if 0 <= i < total_episodes})
    return indices


def compute_velocity(positions: np.ndarray, timestamps: np.ndarray, fallback_dt: float) -> np.ndarray:
    """Compute joint velocities by time gradient with fallback for bad timestamps."""
    if positions.shape[0] < 2:
        return np.zeros_like(positions, dtype=np.float32)

    timestamps = timestamps.astype(np.float64)
    if timestamps.shape[0] != positions.shape[0]:
        raise ValueError("timestamps length does not match positions length")
    if not np.all(np.isfinite(timestamps)):
        return np.gradient(positions, fallback_dt, axis=0).astype(np.float32)
    if np.any(np.diff(timestamps) <= 0):
        return np.gradient(positions, fallback_dt, axis=0).astype(np.float32)
    return np.gradient(positions, timestamps, axis=0).astype(np.float32)


def to_object_array(list_of_arrays: List[np.ndarray]) -> np.ndarray:
    """Store variable-length motions as object arrays for npz (allow_pickle=True)."""
    return np.array(list_of_arrays, dtype=object)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert AGIBOT parquet dataset to GapONet .npz")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset root, e.g. agibot/data/H3_example",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="Episode indices: 'all', '0,1,2', or '0-10'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npz path (default: <dataset>/motion_agibot.npz)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    loader = DataLoader(str(dataset_path))
    total_episodes = loader.get_episode_count()
    episode_indices = parse_episode_indices(args.episodes, total_episodes)
    if not episode_indices:
        raise ValueError(f"No valid episodes parsed from '{args.episodes}'")

    joint_names = JointNameMapper.get_joint_names()
    num_dofs = len(joint_names)

    real_positions_list: List[np.ndarray] = []
    target_positions_list: List[np.ndarray] = []
    velocities_list: List[np.ndarray] = []
    torques_list: List[np.ndarray] = []

    print(f"[Info] Dataset: {dataset_path}")
    print(f"[Info] Episodes: {episode_indices}")
    print(f"[Info] DOFs: {num_dofs}")

    for ep in episode_indices:
        frames = loader.load_episode(ep)
        if not frames:
            print(f"[Skip] Episode {ep}: empty")
            continue

        arrays = frames_to_arrays(frames)
        timestamps = arrays["timestamps"].astype(np.float64)
        real_pos = arrays["real_joint_pos"].astype(np.float32)
        target_pos = arrays["target_joint_pos"].astype(np.float32)

        if real_pos.shape[1] != num_dofs or target_pos.shape[1] != num_dofs:
            raise ValueError(
                f"Episode {ep} DOF mismatch: real={real_pos.shape}, target={target_pos.shape}, expected {num_dofs}"
            )

        vel = compute_velocity(real_pos, timestamps, fallback_dt=1.0 / loader.config.fps)
        torque = np.zeros_like(real_pos, dtype=np.float32)

        # Basic sanity checks
        if not np.isfinite(real_pos).all() or not np.isfinite(target_pos).all():
            raise ValueError(f"Episode {ep} contains NaN in positions")

        real_positions_list.append(real_pos)
        target_positions_list.append(target_pos)
        velocities_list.append(vel)
        torques_list.append(torque)

        print(f"[OK] Episode {ep}: frames={real_pos.shape[0]}")

    if not real_positions_list:
        raise RuntimeError("No valid episodes converted. Nothing to save.")

    payloads = np.zeros(len(real_positions_list), dtype=np.float32)

    output_path = Path(args.output) if args.output else dataset_path / "motion_agibot.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        real_dof_positions=to_object_array(real_positions_list),
        real_dof_velocities=to_object_array(velocities_list),
        real_dof_positions_cmd=to_object_array(target_positions_list),
        real_dof_torques=to_object_array(torques_list),
        joint_sequence=np.array(joint_names, dtype=object),
        payloads=payloads,
        joint_names=np.array(joint_names, dtype=object),
    )

    print(f"[Saved] {output_path}")
    print("[Check] Keys:", np.load(output_path, allow_pickle=True).files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
