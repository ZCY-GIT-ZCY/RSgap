"""
Per-joint correlation/regression between sim-real error and joint signals.

Inputs:
1) aligned pkl from replay_action.py:
   agibot/outputs/replay_results/episode_XXXXXX_aligned.pkl
   - timestamps
   - real_joint_pos / sim_joint_pos
   - (optional) joint_names
2) parquet from dataset:
   agibot/data/<DATASET>/data/chunk-???/episode_XXXXXX.parquet
   - observation.state (94-dim), joint current at indices [68:82]

Output:
- summary.csv: one row per (episode, joint)

Notes:
- Uses absolute joint error |sim-real| as target by default.
- Current is only available for 14 arm joints (head/gripper have NaN current).
- This script is offline; no IsaacSim needed.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _parse_episode_arg(arg: str, available: List[int]) -> List[int]:
    s = arg.strip().lower()
    if s == "all":
        return sorted(available)
    out: List[int] = []
    for part in [x.strip() for x in s.split(",") if x.strip()]:
        if "-" in part:
            a, b = part.split("-", 1)
            i0, i1 = int(a), int(b)
            if i1 < i0:
                i0, i1 = i1, i0
            out.extend(list(range(i0, i1 + 1)))
        else:
            out.append(int(part))
    avail_set = set(available)
    return [e for e in sorted(set(out)) if e in avail_set]


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


@dataclass
class RegressionResult:
    r2: float
    coef: Dict[str, float]


def _linear_regression_standardized(y: np.ndarray, xcols: Dict[str, np.ndarray]) -> RegressionResult:
    """
    Simple OLS: y ~ 1 + x1 + x2 + ...
    Returns standardized coefficients and R^2.
    """
    names = list(xcols.keys())
    X_list = []
    for n in names:
        X_list.append(_zscore(xcols[n]))
    X = np.stack(X_list, axis=1).astype(np.float64)
    y0 = _zscore(y).astype(np.float64)

    m = np.isfinite(y0) & np.all(np.isfinite(X), axis=1)
    if int(m.sum()) < (len(names) + 3):
        return RegressionResult(r2=float("nan"), coef={n: float("nan") for n in names})

    X = X[m]
    y0 = y0[m]

    X1 = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)
    beta, *_ = np.linalg.lstsq(X1, y0, rcond=None)
    yhat = X1 @ beta
    ss_res = float(np.sum((y0 - yhat) ** 2))
    ss_tot = float(np.sum((y0 - np.mean(y0)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    coef = {names[i]: float(beta[i + 1]) for i in range(len(names))}
    return RegressionResult(r2=r2, coef=coef)


def _episode_to_parquet_path(dataset_root: Path, episode_idx: int) -> Path:
    chunk_size = 1000
    chunk_idx = episode_idx // chunk_size
    return dataset_root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"


def _load_aligned_pkl(pkl_path: Path) -> Dict[str, np.ndarray]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"aligned.pkl 不是 dict: {pkl_path}")
    return data


def _read_parquet_joint_current(parquet_path: Path) -> np.ndarray:
    """
    Read joint current (14-dim) from observation.state[68:82].
    """
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "缺少 pyarrow，无法读取 parquet。请先安装：pip install pyarrow"
        ) from e

    table = pq.read_table(str(parquet_path), columns=["observation.state"])
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

    current = state[:, 68:82].astype(np.float64)
    return current


def _compute_joint_vel_acc(q: np.ndarray, t: np.ndarray, fallback_dt: float) -> tuple[np.ndarray, np.ndarray]:
    if q.shape[0] < 2:
        zeros = np.zeros_like(q, dtype=np.float64)
        return zeros, zeros
    t = t.astype(np.float64)
    if t.shape[0] != q.shape[0] or not np.all(np.isfinite(t)):
        dt = fallback_dt
        v = np.gradient(q, dt, axis=0)
        a = np.gradient(v, dt, axis=0)
        return v, a
    if np.any(np.diff(t) <= 0):
        dt = fallback_dt
        v = np.gradient(q, dt, axis=0)
        a = np.gradient(v, dt, axis=0)
        return v, a
    v = np.gradient(q, t, axis=0)
    a = np.gradient(v, t, axis=0)
    return v, a


def _write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
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
    parser = argparse.ArgumentParser(description="Per-joint correlation of sim-real error vs joint signals.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "H3_example"),
        help="AGIBOT dataset root, e.g. agibot/data/H3_example",
    )
    parser.add_argument(
        "--aligned-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "replay_results"),
        help="Directory containing episode_XXXXXX_aligned.pkl",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="Episode indices: 'all', '0,1,2', or '0-10' (filtered by aligned pkl existence).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "joint_correlation"),
        help="Output directory for summary.csv",
    )
    parser.add_argument(
        "--signed-error",
        action="store_true",
        help="Use signed error (sim-real) as target instead of |sim-real|.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    aligned_dir = Path(args.aligned_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    aligned_files = sorted(aligned_dir.glob("episode_*_aligned.pkl"))
    if not aligned_files:
        raise FileNotFoundError(f"找不到 aligned pkl：{aligned_dir}")

    avail: List[int] = []
    pkl_by_ep: Dict[int, Path] = {}
    for p in aligned_files:
        stem = p.stem
        try:
            ep_str = stem.split("_")[1]
            ep = int(ep_str)
        except Exception:
            continue
        avail.append(ep)
        pkl_by_ep[ep] = p
    avail = sorted(set(avail))
    episodes = _parse_episode_arg(args.episodes, avail)
    if not episodes:
        raise ValueError(f"episodes 解析后为空。可用 episodes: {avail[:20]} ... 共{len(avail)}")

    rows: List[Dict[str, object]] = []

    for ep in episodes:
        pkl_path = pkl_by_ep[ep]
        aligned = _load_aligned_pkl(pkl_path)

        t = np.asarray(aligned.get("timestamps"), dtype=np.float64)
        q_real = np.asarray(aligned.get("real_joint_pos"), dtype=np.float64)
        q_sim = np.asarray(aligned.get("sim_joint_pos"), dtype=np.float64)

        if t.ndim != 1 or q_real.ndim != 2 or q_sim.ndim != 2:
            raise ValueError(f"aligned 数据维度异常：{pkl_path}")

        d = min(q_real.shape[1], q_sim.shape[1])
        q_real = q_real[:, :d]
        q_sim = q_sim[:, :d]
        n = min(len(t), q_real.shape[0], q_sim.shape[0])
        t = t[:n]
        q_real = q_real[:n]
        q_sim = q_sim[:n]

        # joint names fallback
        joint_names = aligned.get("joint_names")
        if joint_names is None or len(joint_names) < d:
            try:
                from data_utils import JointNameMapper
                joint_names = JointNameMapper.get_joint_names()
            except Exception:
                joint_names = [f"joint_{i:02d}" for i in range(d)]
        joint_names = list(joint_names)[:d]

        # error target
        err = q_sim - q_real
        err_target = err if args.signed_error else np.abs(err)

        # velocity / acceleration from real joint position
        dt = _median_dt(t) or (1.0 / 30.0)
        v, a = _compute_joint_vel_acc(q_real, t, fallback_dt=dt)
        v = np.abs(v)
        a = np.abs(a)

        # current from parquet (14 joints: left+right arms)
        parquet_path = _episode_to_parquet_path(dataset_root, ep)
        if not parquet_path.exists():
            raise FileNotFoundError(f"parquet 不存在：{parquet_path}")
        current = _read_parquet_joint_current(parquet_path)
        current = np.abs(current)

        # align length with current
        n2 = min(n, current.shape[0])
        t = t[:n2]
        err_target = err_target[:n2]
        v = v[:n2]
        a = a[:n2]
        current = current[:n2]

        # current corresponds to arm joints at indices [2:16] in 18D layout
        current_offset = 2
        current_count = current.shape[1]  # 14

        for j in range(d):
            err_j = err_target[:, j]
            v_j = v[:, j]
            a_j = a[:, j]

            cur_j: Optional[np.ndarray] = None
            cur_idx = j - current_offset
            if 0 <= cur_idx < current_count:
                cur_j = current[:, cur_idx]

            corr_v = _corr(err_j, v_j)
            corr_a = _corr(err_j, a_j)
            corr_c = _corr(err_j, cur_j) if cur_j is not None else float("nan")

            xcols = {
                "vel_abs": v_j,
                "acc_abs": a_j,
            }
            if cur_j is not None:
                xcols["current_abs"] = cur_j
            reg = _linear_regression_standardized(err_j, xcols=xcols)

            row: Dict[str, object] = {
                "episode": ep,
                "joint_index": j,
                "joint_name": joint_names[j],
                "n": int(n2),
                "err_abs_mean": float(np.nanmean(err_j)),
                "err_abs_p95": float(np.nanpercentile(err_j, 95)),
                "corr_err_vel_abs": float(corr_v),
                "corr_err_acc_abs": float(corr_a),
                "corr_err_current_abs": float(corr_c),
                "reg_r2": float(reg.r2),
                "reg_beta_vel_abs": float(reg.coef.get("vel_abs", float("nan"))),
                "reg_beta_acc_abs": float(reg.coef.get("acc_abs", float("nan"))),
                "reg_beta_current_abs": float(reg.coef.get("current_abs", float("nan"))),
            }
            rows.append(row)

    _write_csv(rows, outdir / "summary.csv")
    print(f"[Done] summary saved to: {outdir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
