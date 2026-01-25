"""Quick alignment check between motion data and sim joints.

Runs a short rollout using zero delta-action and reports joint position errors.
This script is agibot-specific and does not affect other robots.
"""

from __future__ import annotations

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import torch
import importlib.util
from pathlib import Path
import types

from isaaclab.app import AppLauncher
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _median_dt(t: np.ndarray) -> float:
    if t.size < 3:
        return 0.0
    d = np.diff(t.astype(np.float64))
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    return float(np.median(d))


def _zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < eps:
        return x * 0.0
    return (x - mu) / sd


def _finite_mask(*arrs: np.ndarray) -> np.ndarray:
    m = np.ones(arrs[0].shape[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = _finite_mask(x, y)
    if m.sum() < 3:
        return float("nan")
    x0 = x[m]
    y0 = y[m]
    sx = np.std(x0)
    sy = np.std(y0)
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(x0, y0)[0, 1])


def _linear_regression_standardized(y: np.ndarray, xcols: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    names = list(xcols.keys())
    X_list = []
    for n in names:
        X_list.append(_zscore(xcols[n]))
    X = np.stack(X_list, axis=1).astype(np.float64)
    y0 = _zscore(y).astype(np.float64)

    m = np.isfinite(y0) & np.all(np.isfinite(X), axis=1)
    if int(m.sum()) < (len(names) + 3):
        return float("nan"), {n: float("nan") for n in names}

    X = X[m]
    y0 = y0[m]
    X1 = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)
    beta, *_ = np.linalg.lstsq(X1, y0, rcond=None)
    yhat = X1 @ beta
    ss_res = float(np.sum((y0 - yhat) ** 2))
    ss_tot = float(np.sum((y0 - np.mean(y0)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    coef = {names[i]: float(beta[i + 1]) for i in range(len(names))}
    return r2, coef


def _interp_to_target_timestamps(
    source_t: np.ndarray, source_data: np.ndarray, target_t: np.ndarray
) -> np.ndarray:
    if source_data.ndim == 1:
        return np.interp(target_t, source_t, source_data)
    result = np.zeros((len(target_t), source_data.shape[1]), dtype=np.float32)
    for i in range(source_data.shape[1]):
        result[:, i] = np.interp(target_t, source_t, source_data[:, i])
    return result


def _read_parquet_signals(parquet_path: Path) -> dict[str, np.ndarray]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "缺少 pyarrow，无法读取 parquet。请先安装：pip install pyarrow"
        ) from exc

    table = pq.read_table(str(parquet_path), columns=["timestamp", "observation.state"])
    timestamps = np.asarray(table["timestamp"].to_numpy(zero_copy_only=False), dtype=np.float64)
    try:
        state_col = table["observation.state"]
        state = np.asarray(state_col.to_pylist(), dtype=np.float32)
        if state.ndim != 2 or state.shape[1] != 94:
            import pandas as pd  # type: ignore
            df = table.to_pandas()
            state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    except Exception:
        import pandas as pd  # type: ignore
        df = table.to_pandas()
        state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)

    end_wrench = state[:, 2:14].astype(np.float64)
    end_velocity = state[:, 20:32].astype(np.float64)
    joint_position = state[:, 54:68].astype(np.float64)
    joint_current = state[:, 68:82].astype(np.float64)

    return {
        "timestamps": timestamps,
        "end_wrench": end_wrench,
        "end_velocity": end_velocity,
        "joint_position": joint_position,
        "joint_current": joint_current,
    }


def _write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False, encoding="utf-8")
    except Exception:
        keys = list(rows[0].keys())
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check motion alignment for humanoid_agibot")
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Humanoid-AGIBOT-Delta-Action",
        help="Gym task id to run.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of envs to create.")
    parser.add_argument("--num-steps", type=int, default=200, help="Number of steps to run.")
    parser.add_argument("--motion-index", type=int, default=0, help="Fixed motion index to use.")
    parser.add_argument("--time-index", type=int, default=0, help="Fixed start time index to use.")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup steps to skip from stats.")
    parser.add_argument(
        "--sync-to-motion",
        action="store_true",
        help="Before rollout, set robot joint states to motion real positions at the start frame.",
    )
    parser.add_argument("--plot", action="store_true", help="Save plots for top error joints.")
    parser.add_argument("--plot-topk", type=int, default=5, help="Top-K joints to plot.")
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="logs/diagnostics",
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--plot-episode",
        type=int,
        default=None,
        help="If set, override motion-index and plot the whole episode.",
    )
    parser.add_argument(
        "--plot-all-joints",
        action="store_true",
        help="Plot all joints (useful with --plot-episode).",
    )
    parser.add_argument(
        "--parquet-root",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "agibot" / "data" / "H3_example"),
        help="AGIBOT dataset root for parquet signals (e.g., agibot/data/H3_example).",
    )
    parser.add_argument(
        "--resample-fps",
        type=float,
        default=50.0,
        help="Target fps used to resample parquet when building the motion npz.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save sim/real/error time series and aligned parquet signals to CSV.",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default="logs/diagnostics",
        help="Directory to save CSV outputs.",
    )
    # AppLauncher args (e.g., --headless, --device)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch Isaac Sim (required for carb/omni dependencies).
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Ensure task registration without importing sim2real.__init__ (avoids omni.ext).
    tasks_init = Path(__file__).resolve().parents[2] / "source" / "sim2real" / "sim2real" / "tasks" / "humanoid_agibot" / "__init__.py"
    if not tasks_init.is_file():
        raise RuntimeError(f"Task module not found: {tasks_init}")
    # Create lightweight parent packages to bypass sim2real/__init__.py.
    if "sim2real" not in sys.modules:
        sys.modules["sim2real"] = types.ModuleType("sim2real")
    if "sim2real.tasks" not in sys.modules:
        sys.modules["sim2real.tasks"] = types.ModuleType("sim2real.tasks")
    spec = importlib.util.spec_from_file_location("sim2real.tasks.humanoid_agibot", tasks_init)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load task module: {tasks_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sim2real.tasks.humanoid_agibot"] = module
    spec.loader.exec_module(module)

    # Build env config overrides.
    # Load env config class without importing sim2real.__init__.
    env_cfg_path = Path(__file__).resolve().parents[2] / "source" / "sim2real" / "sim2real" / "tasks" / "humanoid_agibot" / "humanoid_agibot_env_cfg.py"
    if not env_cfg_path.is_file():
        raise RuntimeError(f"Env cfg module not found: {env_cfg_path}")
    env_cfg_mod_name = "sim2real.tasks.humanoid_agibot.humanoid_agibot_env_cfg"
    env_cfg_spec = importlib.util.spec_from_file_location(env_cfg_mod_name, env_cfg_path)
    if env_cfg_spec is None or env_cfg_spec.loader is None:
        raise RuntimeError(f"Failed to load env cfg module: {env_cfg_path}")
    env_cfg_module = importlib.util.module_from_spec(env_cfg_spec)
    sys.modules[env_cfg_mod_name] = env_cfg_module
    env_cfg_spec.loader.exec_module(env_cfg_module)
    env_cfg = env_cfg_module.HumanoidOperatorEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)

    env_unwrapped = env.unwrapped
    num_actions = env_unwrapped.cfg.action_space
    device = env_unwrapped.device

    # Initialize motion indices.
    motion_index = args.motion_index if args.plot_episode is None else args.plot_episode
    motion_indices = torch.full((args.num_envs,), motion_index, dtype=torch.long, device=device)
    time_indices = torch.full((args.num_envs,), args.time_index, dtype=torch.long, device=device)

    # Optionally sync robot joints to the motion pose at the start frame.
    if args.sync_to_motion:
        motion_pos = env_unwrapped._motion_loader.dof_positions[motion_indices, time_indices]
        motion_vel = torch.zeros_like(motion_pos)
        env_unwrapped.robot.write_joint_state_to_sim(
            motion_pos, motion_vel, joint_ids=env_unwrapped.motion_joint_ids
        )
        env_unwrapped.robot.set_joint_position_target(
            motion_pos, joint_ids=env_unwrapped.motion_joint_ids
        )
        env_unwrapped._raw_step_simulator()

    # Zero delta-action to directly follow motion targets.
    zero_action = torch.zeros((args.num_envs, num_actions), device=device)

    joint_err_means = []
    joint_err_maxs = []
    sim_traj = []
    real_traj = []
    time_idx_hist: list[int] = []

    per_joint_max = []
    # If plotting full episode, override num_steps to the motion length.
    if args.plot_episode is not None:
        motion_len = int(env_unwrapped._motion_loader.motion_len[motion_index].item())
        args.num_steps = motion_len

    for step in range(args.num_steps):
        time_indices_step = time_indices.clone()
        _, _, dones, info = env_unwrapped.step_operator(
            zero_action, motion_coords=(motion_indices, time_indices_step)
        )
        # Collect trajectories for env 0 (full range for csv/regression).
        sim_pos = env_unwrapped.robot.data.joint_pos[:, env_unwrapped.motion_joint_ids].detach().cpu().numpy()
        real_pos = env_unwrapped._motion_loader.dof_positions[
            motion_indices, time_indices_step
        ].detach().cpu().numpy()
        # Compute aligned error in degrees to match plotting units.
        joint_pos_diff_deg = np.degrees(np.abs(sim_pos - real_pos))
        if step >= args.warmup_steps:
            joint_err_means.append(float(np.mean(joint_pos_diff_deg)))
            joint_err_maxs.append(float(np.max(joint_pos_diff_deg)))
            per_joint_max.append(np.max(joint_pos_diff_deg, axis=0))
        sim_traj.append(sim_pos[0])
        real_traj.append(real_pos[0])
        time_idx_hist.append(int(time_indices_step[0].item()))

        # advance time indices for the next step (mirror env internal logic)
        time_indices = time_indices + 1
        if bool(torch.any(dones)):
            break

    print("Alignment check results:")
    print(f"  steps: {len(joint_err_means)} (warmup skipped: {args.warmup_steps})")
    if joint_err_means:
        print(f"  mean joint error (deg): {np.mean(joint_err_means):.4f}")
        print(f"  max joint error (deg): {np.max(joint_err_maxs):.4f}")
        per_joint_max = np.max(np.stack(per_joint_max, axis=0), axis=0)
        dof_names = env_unwrapped._motion_loader.dof_names
        top_idx = np.argsort(-per_joint_max)[:5]
        print("  top max-error joints (deg):")
        for idx in top_idx:
            print(f"    {dof_names[idx]}: {per_joint_max[idx]:.4f}")
            if args.plot and sim_traj:
                plot_dir = Path(args.plot_dir)
                if args.plot_episode is not None:
                    plot_dir = plot_dir / f"episode_{motion_index:06d}"
                plot_dir.mkdir(parents=True, exist_ok=True)
                sim_arr = np.stack(sim_traj, axis=0)
                real_arr = np.stack(real_traj, axis=0)
                warmup = max(0, int(args.warmup_steps))
                if warmup < sim_arr.shape[0]:
                    sim_arr = sim_arr[warmup:]
                    real_arr = real_arr[warmup:]
                if args.plot_all_joints:
                    top_idx = np.arange(sim_arr.shape[1])
                else:
                    top_idx = np.argsort(-per_joint_max)[: args.plot_topk]
                steps = np.arange(sim_arr.shape[0])
                sim_arr_deg = np.degrees(sim_arr)
                real_arr_deg = np.degrees(real_arr)
                for idx in top_idx:
                    sim_line = sim_arr_deg[:, idx]
                    real_line = real_arr_deg[:, idx]
                    err_line = sim_line - real_line
                    abs_err_line = np.abs(err_line)
                    mean_err = float(np.mean(abs_err_line))

                    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
                    ax0.plot(steps, sim_line, label="sim")
                    ax0.plot(steps, real_line, label="real")
                    ax0.set_title(f"{dof_names[idx]} (deg)")
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

                    ax1.plot(steps, abs_err_line, label="|error|")
                    ax1.set_xlabel("step")
                    ax1.set_ylabel("|error| (deg)")
                    ax1.legend()

                    out_path = plot_dir / f"{dof_names[idx]}_traj.png"
                    fig.tight_layout()
                    fig.savefig(out_path)
                    plt.close(fig)
                print(f"  plots saved to: {plot_dir}")
    else:
        print("  no stats collected (all steps were warmup)")

    if args.save_csv and sim_traj:
        csv_dir = Path(args.csv_dir)
        if args.plot_episode is not None:
            csv_dir = csv_dir / f"episode_{motion_index:06d}"
        csv_dir.mkdir(parents=True, exist_ok=True)

        sim_arr = np.stack(sim_traj, axis=0)
        real_arr = np.stack(real_traj, axis=0)
        sim_arr_deg = np.degrees(sim_arr)
        real_arr_deg = np.degrees(real_arr)
        err_arr = sim_arr_deg - real_arr_deg
        abs_err_arr = np.abs(err_arr)

        warmup = max(0, int(args.warmup_steps))
        if warmup >= sim_arr.shape[0]:
            raise RuntimeError(f"warmup_steps={warmup} >= total_steps={sim_arr.shape[0]}")
        valid_slice = slice(warmup, None)
        sim_arr = sim_arr[valid_slice]
        real_arr = real_arr[valid_slice]
        sim_arr_deg = sim_arr_deg[valid_slice]
        real_arr_deg = real_arr_deg[valid_slice]
        err_arr = err_arr[valid_slice]
        abs_err_arr = abs_err_arr[valid_slice]
        time_idx_hist = time_idx_hist[valid_slice]

        dof_names = env_unwrapped._motion_loader.dof_names
        motion_len = int(env_unwrapped._motion_loader.motion_len[motion_index].item())
        parquet_path = Path(args.parquet_root) / "data" / f"chunk-{motion_index // 1000:03d}" / f"episode_{motion_index:06d}.parquet"
        parquet = _read_parquet_signals(parquet_path)

        t_src = parquet["timestamps"].astype(np.float64)
        if t_src.size < 2:
            raise RuntimeError(f"parquet timestamps too short: {parquet_path}")
        t0 = float(t_src[0])
        dt = 1.0 / float(args.resample_fps)
        target_ts_full = t0 + np.arange(motion_len, dtype=np.float64) * dt
        target_ts = target_ts_full[np.array(time_idx_hist, dtype=np.int64)]

        # Ensure monotonic source timestamps
        order = np.argsort(t_src)
        t_src_sorted = t_src[order]
        unique_t, unique_idx = np.unique(t_src_sorted, return_index=True)

        end_wrench = _interp_to_target_timestamps(unique_t, parquet["end_wrench"][order][unique_idx], target_ts)
        end_velocity = _interp_to_target_timestamps(unique_t, parquet["end_velocity"][order][unique_idx], target_ts)
        joint_position = _interp_to_target_timestamps(unique_t, parquet["joint_position"][order][unique_idx], target_ts)
        joint_current = _interp_to_target_timestamps(unique_t, parquet["joint_current"][order][unique_idx], target_ts)

        # Build per-joint current aligned to dof order (head/gripper -> NaN)
        current_full = np.full((joint_current.shape[0], sim_arr.shape[1]), np.nan, dtype=np.float64)
        current_full[:, 2:16] = joint_current

        # Per-frame signals (aligned, warmup removed)
        frame_rows: list[dict[str, object]] = []
        for i, (ts, ti) in enumerate(zip(target_ts, time_idx_hist)):
            row: dict[str, object] = {
                "motion_index": motion_index,
                "time_index": int(ti),
                "timestamp": float(ts),
            }
            for k in range(end_wrench.shape[1]):
                row[f"end_wrench_{k:02d}"] = float(end_wrench[i, k])
            for k in range(end_velocity.shape[1]):
                row[f"end_velocity_{k:02d}"] = float(end_velocity[i, k])
            for k in range(joint_position.shape[1]):
                row[f"joint_position_{k:02d}"] = float(joint_position[i, k])
            for k in range(joint_current.shape[1]):
                row[f"joint_current_{k:02d}"] = float(joint_current[i, k])
            frame_rows.append(row)
        _write_csv(frame_rows, csv_dir / f"episode_{motion_index:06d}_frame_signals.csv")

        # Velocity / acceleration from real (aligned to target_ts)
        v = np.gradient(real_arr_deg, dt, axis=0)
        a = np.gradient(v, dt, axis=0)
        v = np.abs(v)
        a = np.abs(a)

        # Per-joint time series rows (warmup removed)
        series_rows: list[dict[str, object]] = []
        for i, (ts, ti) in enumerate(zip(target_ts, time_idx_hist)):
            for j in range(sim_arr.shape[1]):
                series_rows.append(
                    {
                        "motion_index": motion_index,
                        "time_index": int(ti),
                        "timestamp": float(ts),
                        "joint_index": int(j),
                        "joint_name": str(dof_names[j]),
                        "sim_pos": float(sim_arr_deg[i, j]),
                        "real_pos": float(real_arr_deg[i, j]),
                        "err": float(err_arr[i, j]),
                        "abs_err": float(abs_err_arr[i, j]),
                        "real_vel_abs": float(v[i, j]),
                        "real_acc_abs": float(a[i, j]),
                        "current_abs": float(current_full[i, j]) if np.isfinite(current_full[i, j]) else float("nan"),
                    }
                )
        _write_csv(series_rows, csv_dir / f"episode_{motion_index:06d}_per_joint_timeseries.csv")

        # Per-joint regression summary (warmup removed)
        reg_rows: list[dict[str, object]] = []
        for j in range(sim_arr.shape[1]):
            err_j = abs_err_arr[:, j]
            v_j = v[:, j]
            a_j = a[:, j]
            cur_j = current_full[:, j] if np.isfinite(current_full[:, j]).any() else None

            corr_v = _corr(err_j, v_j)
            corr_a = _corr(err_j, a_j)
            corr_c = _corr(err_j, cur_j) if cur_j is not None else float("nan")

            xcols = {"vel_abs": v_j, "acc_abs": a_j}
            if cur_j is not None:
                xcols["current_abs"] = cur_j
            r2, coef = _linear_regression_standardized(err_j, xcols=xcols)

            reg_rows.append(
                {
                    "motion_index": motion_index,
                    "joint_index": int(j),
                    "joint_name": str(dof_names[j]),
                    "err_abs_mean": float(np.nanmean(err_j)),
                    "err_abs_p95": float(np.nanpercentile(err_j, 95)),
                    "corr_err_vel_abs": float(corr_v),
                    "corr_err_acc_abs": float(corr_a),
                    "corr_err_current_abs": float(corr_c),
                    "reg_r2": float(r2),
                    "reg_beta_vel_abs": float(coef.get("vel_abs", float("nan"))),
                    "reg_beta_acc_abs": float(coef.get("acc_abs", float("nan"))),
                    "reg_beta_current_abs": float(coef.get("current_abs", float("nan"))),
                }
            )
        _write_csv(reg_rows, csv_dir / f"episode_{motion_index:06d}_regression.csv")
        print(f"  csv saved to: {csv_dir}")

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
