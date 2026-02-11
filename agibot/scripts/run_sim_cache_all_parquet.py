"""
Run IsaacLab simulation for parquet episodes and cache sim trajectories.

This script:
1) Scans parquet episodes under <dataset>/data/chunk-*/episode_*.parquet
2) Replays corresponding motions in IsaacLab with zero delta-action
3) Saves per-episode sim/real trajectories and error metrics
4) Writes a summary CSV and an optional merged NPZ cache

Notes:
- The motion file must be generated from the same episode ordering.
- Default ordering follows sorted parquet paths (same as convert_parquet_to_npz --episodes all).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import AppLauncher


def _scan_episode_paths(data_root: Path) -> List[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data path not found: {data_root}")
    paths = sorted(data_root.rglob("episode_*.parquet"))
    return [p for p in paths if p.is_file()]


def _parse_episode_id(path: Path) -> int:
    match = re.search(r"episode_(\d+)\.parquet$", path.name)
    if not match:
        raise ValueError(f"Invalid episode parquet name: {path.name}")
    return int(match.group(1))


def _parse_episode_arg(episode_arg: str, available_ids: List[int]) -> List[int]:
    arg = episode_arg.strip().lower()
    if arg == "all":
        return sorted(available_ids)

    available = set(available_ids)
    result: List[int] = []
    for part in [p.strip() for p in arg.split(",") if p.strip()]:
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a)
            hi = int(b)
            if hi < lo:
                lo, hi = hi, lo
            for idx in range(lo, hi + 1):
                if idx in available:
                    result.append(idx)
        else:
            idx = int(part)
            if idx in available:
                result.append(idx)
    return sorted(set(result))


def _sync_to_motion(env_unwrapped, motion_index: int, time_index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    device = env_unwrapped.device
    motion_indices = torch.full((env_unwrapped.num_envs,), motion_index, dtype=torch.long, device=device)
    time_indices = torch.full((env_unwrapped.num_envs,), time_index, dtype=torch.long, device=device)
    motion_pos = env_unwrapped._motion_loader.dof_positions[motion_indices, time_indices]
    motion_vel = torch.zeros_like(motion_pos)
    env_unwrapped.robot.write_joint_state_to_sim(
        motion_pos, motion_vel, joint_ids=env_unwrapped.motion_joint_ids
    )
    env_unwrapped.robot.set_joint_position_target(
        motion_pos, joint_ids=env_unwrapped.motion_joint_ids
    )
    env_unwrapped._raw_step_simulator()
    env_unwrapped.motion_indices[:] = motion_indices
    env_unwrapped.time_indices[:] = time_indices
    env_unwrapped.last_delta_action[:] = 0
    if hasattr(env_unwrapped, "delta_action"):
        env_unwrapped.delta_action[:] = 0
    if hasattr(env_unwrapped, "model_history"):
        env_unwrapped.model_history[:] = 0
    return motion_indices, time_indices


def _run_one_episode(env_unwrapped, motion_index: int) -> Tuple[np.ndarray, np.ndarray]:
    device = env_unwrapped.device
    num_actions = env_unwrapped.cfg.action_space
    zero_action = torch.zeros((env_unwrapped.num_envs, num_actions), device=device)

    motion_indices, time_indices = _sync_to_motion(env_unwrapped, motion_index, time_index=0)
    motion_len = int(env_unwrapped._motion_loader.motion_len[motion_index].item())

    sim_traj: List[np.ndarray] = []
    real_traj: List[np.ndarray] = []
    for _ in range(motion_len):
        time_indices_step = time_indices.clone()
        _, _, dones, _ = env_unwrapped.step_operator(
            zero_action, motion_coords=(motion_indices, time_indices_step)
        )
        sim_pos = env_unwrapped.robot.data.joint_pos[:, env_unwrapped.motion_joint_ids].detach().cpu().numpy()
        real_pos = env_unwrapped._motion_loader.dof_positions[
            motion_indices, time_indices_step
        ].detach().cpu().numpy()
        sim_traj.append(sim_pos[0].astype(np.float32))
        real_traj.append(real_pos[0].astype(np.float32))
        motion_indices = env_unwrapped.motion_indices.clone()
        time_indices = env_unwrapped.time_indices.clone()
        if bool(torch.any(dones)):
            break

    if not sim_traj:
        return np.zeros((0, env_unwrapped.num_dofs), dtype=np.float32), np.zeros((0, env_unwrapped.num_dofs), dtype=np.float32)
    return np.stack(sim_traj, axis=0), np.stack(real_traj, axis=0)


def _save_episode_npz(
    out_file: Path,
    episode_id: int,
    motion_index: int,
    sim_arr: np.ndarray,
    real_arr: np.ndarray,
    peak_err: float,
    mean_err: float,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_file,
        episode_id=np.int64(episode_id),
        motion_index=np.int64(motion_index),
        sim_dof_positions=sim_arr.astype(np.float32),
        real_dof_positions=real_arr.astype(np.float32),
        abs_err=np.abs(sim_arr - real_arr).astype(np.float32),
        peak_abs_err_rad=np.float32(peak_err),
        mean_abs_err_rad=np.float32(mean_err),
    )


def _write_summary_csv(rows: List[Dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode_id",
        "motion_index",
        "status",
        "num_frames",
        "peak_abs_err_rad",
        "mean_abs_err_rad",
        "peak_abs_err_deg",
        "mean_abs_err_deg",
        "episode_file",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _merge_episode_npz(episode_files: List[Path], out_file: Path) -> None:
    episode_ids: List[int] = []
    motion_indices: List[int] = []
    sim_list: List[np.ndarray] = []
    real_list: List[np.ndarray] = []
    peak_list: List[float] = []
    mean_list: List[float] = []
    motion_len: List[int] = []

    for f in episode_files:
        data = np.load(f, allow_pickle=True)
        episode_ids.append(int(data["episode_id"]))
        motion_indices.append(int(data["motion_index"]))
        sim = np.asarray(data["sim_dof_positions"], dtype=np.float32)
        real = np.asarray(data["real_dof_positions"], dtype=np.float32)
        sim_list.append(sim)
        real_list.append(real)
        peak_list.append(float(np.asarray(data["peak_abs_err_rad"]).item()))
        mean_list.append(float(np.asarray(data["mean_abs_err_rad"]).item()))
        motion_len.append(int(sim.shape[0]))

    if not sim_list:
        return
    max_len = max(motion_len)
    dof = sim_list[0].shape[1]
    sim_padded = np.zeros((len(sim_list), max_len, dof), dtype=np.float32)
    real_padded = np.zeros((len(real_list), max_len, dof), dtype=np.float32)
    for i, (sim, real) in enumerate(zip(sim_list, real_list)):
        sim_padded[i, : sim.shape[0], :] = sim
        real_padded[i, : real.shape[0], :] = real

    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_file,
        episode_indices=np.asarray(episode_ids, dtype=np.int64),
        motion_indices=np.asarray(motion_indices, dtype=np.int64),
        sim_dof_positions=np.asarray(sim_list, dtype=object),
        real_dof_positions=np.asarray(real_list, dtype=object),
        sim_dof_positions_padded=sim_padded,
        real_dof_positions_padded=real_padded,
        motion_len=np.asarray(motion_len, dtype=np.int64),
        peak_abs_err_rad=np.asarray(peak_list, dtype=np.float32),
        mean_abs_err_rad=np.asarray(mean_list, dtype=np.float32),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all parquet episodes in sim and cache trajectories.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset root, e.g. D:/Research/Part_Agi/agibot/data/H3_example",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default=None,
        help="Motion npz used by environment. Default: <dataset>/motion_agibot.npz",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="Episode indices: all, 0,1,2, or 0-100",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Humanoid-AGIBOT-Delta-Action",
        help="Gym task id.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <dataset>/sim_cache",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-episode cache files.",
    )
    parser.add_argument(
        "--save-merged-npz",
        action="store_true",
        help="Also save merged cache npz with object and padded arrays.",
    )
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    data_root = dataset_path / "data"
    episode_paths = _scan_episode_paths(data_root)
    if not episode_paths:
        raise RuntimeError(f"No episode parquet found under: {data_root}")

    episode_id_to_path = {_parse_episode_id(p): p for p in episode_paths}
    selected_episode_ids = _parse_episode_arg(args.episodes, sorted(episode_id_to_path.keys()))
    if not selected_episode_ids:
        raise RuntimeError(f"No episodes selected by --episodes={args.episodes}")

    # Motion order must follow sorted parquet paths.
    sorted_ids = sorted(episode_id_to_path.keys())
    episode_id_to_motion_index = {ep_id: i for i, ep_id in enumerate(sorted_ids)}

    motion_file = Path(args.motion_file).resolve() if args.motion_file else (dataset_path / "motion_agibot.npz").resolve()
    if not motion_file.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (dataset_path / "sim_cache").resolve()
    episodes_out_dir = output_dir / "episodes"
    summary_csv_path = output_dir / "episode_metrics.csv"

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Ensure task package import works from repo layout.
    # NOTE: keep imports after AppLauncher, consistent with training scripts.
    repo_root = Path(__file__).resolve().parents[2]
    sim2real_src = repo_root / "gaponet" / "source" / "sim2real"
    sim2real_assets_src = repo_root / "gaponet" / "source" / "sim2real_assets"
    for p in (sim2real_src, sim2real_assets_src):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import sim2real.tasks.humanoid_agibot  # noqa: F401
    from sim2real.tasks.humanoid_agibot.humanoid_agibot_env_cfg import HumanoidOperatorEnvCfg

    env_cfg = HumanoidOperatorEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.sim.device = args.device
    # Use play mode so MotionLoader does not skip first 10 motions.
    env_cfg.mode = "play"
    env_cfg.train_motion_file = str(motion_file)
    env_cfg.test_motion_file = str(motion_file)
    env_cfg.add_noise = False
    env_cfg.online_gap_filter = False

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env_unwrapped = env.unwrapped
    motion_count = int(env_unwrapped._motion_loader.motion_num)
    if motion_count != len(sorted_ids):
        raise RuntimeError(
            "Motion/parquet count mismatch. "
            f"motion_count={motion_count}, parquet_count={len(sorted_ids)}. "
            "Please ensure motion_agibot.npz was generated from the same parquet set and ordering."
        )
    print(f"[Info] Dataset: {dataset_path}")
    print(f"[Info] Motion file: {motion_file}")
    print(f"[Info] Episodes selected: {len(selected_episode_ids)}")
    print(f"[Info] Output dir: {output_dir}")

    rows: List[Dict[str, object]] = []
    saved_episode_files: List[Path] = []

    # Load existing summary for resume.
    existing_done: Dict[int, Dict[str, object]] = {}
    if summary_csv_path.exists() and not args.overwrite:
        with open(summary_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                ep_id = int(r["episode_id"])
                existing_done[ep_id] = r
        print(f"[Info] Resume mode: found {len(existing_done)} rows in {summary_csv_path}")

    total = len(selected_episode_ids)
    processed = 0
    for i, episode_id in enumerate(selected_episode_ids, start=1):
        motion_index = episode_id_to_motion_index[episode_id]
        episode_file = episode_id_to_path[episode_id]
        per_episode_out = episodes_out_dir / f"episode_{episode_id:06d}.npz"

        if not args.overwrite and per_episode_out.exists() and episode_id in existing_done:
            row = dict(existing_done[episode_id])
            row["episode_id"] = int(row["episode_id"])
            row["motion_index"] = int(row["motion_index"])
            row["num_frames"] = int(row["num_frames"])
            row["peak_abs_err_rad"] = float(row["peak_abs_err_rad"])
            row["mean_abs_err_rad"] = float(row["mean_abs_err_rad"])
            row["peak_abs_err_deg"] = float(row["peak_abs_err_deg"])
            row["mean_abs_err_deg"] = float(row["mean_abs_err_deg"])
            rows.append(row)
            saved_episode_files.append(per_episode_out)
            print(f"[Skip] {i}/{total} episode={episode_id} cached")
            continue

        sim_arr, real_arr = _run_one_episode(env_unwrapped, motion_index)
        if sim_arr.shape[0] == 0:
            row = {
                "episode_id": episode_id,
                "motion_index": motion_index,
                "status": "failed_empty",
                "num_frames": 0,
                "peak_abs_err_rad": float("nan"),
                "mean_abs_err_rad": float("nan"),
                "peak_abs_err_deg": float("nan"),
                "mean_abs_err_deg": float("nan"),
                "episode_file": str(episode_file),
            }
            rows.append(row)
            print(f"[Warn] {i}/{total} episode={episode_id} empty rollout")
            continue

        err = np.abs(sim_arr - real_arr)
        peak_err = float(np.max(err))
        mean_err = float(np.mean(err))
        _save_episode_npz(per_episode_out, episode_id, motion_index, sim_arr, real_arr, peak_err, mean_err)
        saved_episode_files.append(per_episode_out)

        row = {
            "episode_id": episode_id,
            "motion_index": motion_index,
            "status": "ok",
            "num_frames": int(sim_arr.shape[0]),
            "peak_abs_err_rad": peak_err,
            "mean_abs_err_rad": mean_err,
            "peak_abs_err_deg": float(np.rad2deg(peak_err)),
            "mean_abs_err_deg": float(np.rad2deg(mean_err)),
            "episode_file": str(episode_file),
        }
        rows.append(row)
        processed += 1
        print(
            f"[OK] {i}/{total} episode={episode_id} motion_index={motion_index} "
            f"frames={sim_arr.shape[0]} peak={peak_err:.6f}rad mean={mean_err:.6f}rad"
        )

        # Periodically flush summary for resume safety.
        if processed % 20 == 0:
            _write_summary_csv(rows, summary_csv_path)
            print(f"[Info] Flush summary: {summary_csv_path} (rows={len(rows)})")

    _write_summary_csv(rows, summary_csv_path)
    print(f"[Saved] Summary CSV: {summary_csv_path}")

    if args.save_merged_npz:
        merged_out = output_dir / "sim_cache_all.npz"
        _merge_episode_npz(saved_episode_files, merged_out)
        print(f"[Saved] Merged NPZ: {merged_out}")

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

