"""Compute per-joint mean sim-real error across all cached episodes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np


def _iter_episode_files(episodes_dir: Path) -> List[Path]:
    return sorted(episodes_dir.rglob("*.npz"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute per-joint mean error from sim cache npz files.")
    parser.add_argument(
        "--episodes-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "H3_example" / "sim_cache" / "episodes"),
        help="Directory containing per-episode npz files.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "H3_example" / "sim_cache" / "mean_joint_error.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    episodes_dir = Path(args.episodes_dir).resolve()
    if not episodes_dir.exists():
        raise FileNotFoundError(f"episodes dir not found: {episodes_dir}")

    files = _iter_episode_files(episodes_dir)
    if not files:
        raise RuntimeError(f"No episode npz files found under: {episodes_dir}")

    err_chunks: List[np.ndarray] = []
    episode_peak_list: List[np.ndarray] = []

    for f in files:
        data = np.load(f, allow_pickle=True)
        if "sim_dof_positions" not in data or "real_dof_positions" not in data:
            continue
        sim = np.asarray(data["sim_dof_positions"], dtype=np.float32)
        real = np.asarray(data["real_dof_positions"], dtype=np.float32)
        min_len = min(sim.shape[0], real.shape[0])
        if min_len == 0:
            continue
        sim = sim[:min_len]
        real = real[:min_len]
        err = np.abs(sim - real)
        err_chunks.append(err)
        episode_peak_list.append(np.max(err, axis=0))

    if not err_chunks:
        raise RuntimeError("No valid sim/real data found in episodes.")

    all_err = np.concatenate(err_chunks, axis=0)
    mean_err = np.mean(all_err, axis=0)
    median_err = np.median(all_err, axis=0)
    var_err = np.var(all_err, axis=0)
    episode_peak = np.max(np.stack(episode_peak_list, axis=0), axis=0)

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "joint_index",
            "mean_abs_err_rad",
            "median_abs_err_rad",
            "var_abs_err_rad",
            "episode_peak_abs_err_rad",
            "mean_abs_err_deg",
            "median_abs_err_deg",
            "var_abs_err_deg",
            "episode_peak_abs_err_deg",
        ])
        for idx in range(mean_err.shape[0]):
            writer.writerow([
                idx,
                float(mean_err[idx]),
                float(median_err[idx]),
                float(var_err[idx]),
                float(episode_peak[idx]),
                float(np.degrees(mean_err[idx])),
                float(np.degrees(median_err[idx])),
                float(np.degrees(var_err[idx])),
                float(np.degrees(episode_peak[idx])),
            ])

    print(f"[INFO] Saved: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
