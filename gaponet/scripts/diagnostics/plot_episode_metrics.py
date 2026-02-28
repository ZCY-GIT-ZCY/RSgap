"""Plot per-episode sim-real trajectories from cached npz files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main() -> int:
    parser = argparse.ArgumentParser(description="Plot per-episode sim-real trajectories from cached npz files")
    parser.add_argument(
        "--episodes-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "agibot" / "data" / "H3_example" / "sim_cache" / "episodes"),
        help="Directory containing per-episode npz files (from run_sim_cache_all_parquet).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "agibot" / "data" / "H3_example" / "sim_cache" / "plot"),
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="If >0, only plot the first N episodes after sorting by filename.",
    )
    parser.add_argument(
        "--plot-topk",
        type=int,
        default=5,
        help="Plot top-K joints by mean abs error (use --plot-all-joints to plot all).",
    )
    parser.add_argument(
        "--plot-all-joints",
        action="store_true",
        help="Plot all joints for each episode.",
    )
    args = parser.parse_args()

    episodes_dir = Path(args.episodes_dir).resolve()
    if not episodes_dir.exists():
        raise FileNotFoundError(f"episodes dir not found: {episodes_dir}")

    episode_files = sorted(episodes_dir.rglob("*.npz"))
    if not episode_files:
        raise RuntimeError(f"No episode npz found under: {episodes_dir}")

    if args.max_episodes and args.max_episodes > 0:
        episode_files = episode_files[: args.max_episodes]

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep_file in episode_files:
        data = np.load(ep_file, allow_pickle=True)
        if "sim_dof_positions" not in data or "real_dof_positions" not in data:
            print(f"[WARN] Missing sim/real arrays in {ep_file.name}; skip")
            continue

        sim = np.asarray(data["sim_dof_positions"], dtype=np.float32)
        real = np.asarray(data["real_dof_positions"], dtype=np.float32)
        min_len = min(sim.shape[0], real.shape[0])
        if min_len == 0:
            print(f"[WARN] Empty episode: {ep_file.name}")
            continue
        sim = sim[:min_len]
        real = real[:min_len]

        steps = np.arange(min_len)
        sim_deg = np.degrees(sim)
        real_deg = np.degrees(real)

        abs_err = np.abs(sim_deg - real_deg)
        per_joint_mean = np.mean(abs_err, axis=0)

        if args.plot_all_joints:
            joint_idx = np.arange(sim_deg.shape[1])
        else:
            topk = min(args.plot_topk, sim_deg.shape[1])
            joint_idx = np.argsort(-per_joint_mean)[:topk]

        ep_dir = out_dir / ep_file.stem
        ep_dir.mkdir(parents=True, exist_ok=True)

        for j in joint_idx:
            sim_line = sim_deg[:, j]
            real_line = real_deg[:, j]
            err_line = np.abs(sim_line - real_line)
            mean_err = float(np.mean(err_line))

            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
            ax0.plot(steps, sim_line, label="sim")
            ax0.plot(steps, real_line, label="real")
            ax0.set_title(f"{ep_file.stem} - joint {j}")
            ax0.text(
                0.01,
                0.95,
                f"mean |err|: {mean_err:.3f} deg",
                transform=ax0.transAxes,
                va="top",
                ha="left",
            )
            ax0.set_ylabel("position (deg)")
            ax0.legend()
            ax0.grid(True, alpha=0.3)

            ax1.plot(steps, err_line, label="|error|")
            ax1.set_xlabel("step")
            ax1.set_ylabel("|error| (deg)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(ep_dir / f"joint_{j:02d}.png")
            plt.close(fig)

        print(f"[INFO] Saved: {ep_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
