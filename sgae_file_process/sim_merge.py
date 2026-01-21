"""
Robot Motion Data Merger (Name-aligned, no freq/payload)

- Scans a root directory for .npz files and merges them into a single dataset.
- No frequency classification, no payload extraction.
- Uses joint names to align dimensions:
  - Pick a canonical joint sequence (the longest and most frequent among samples).
  - For each sample, map its channels to the canonical order by joint name.
  - If a sample has no joint names and its D != target D, it will be skipped.
- If all samples already share the same D and names, no expansion/reorder is needed.

Output:
- merged_all.npz at the output directory
- Arrays saved as object arrays (each item is a (T, D_target) ndarray)
"""

import os
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

def find_npz_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".npz"):
                files.append(Path(dirpath) / f)
    return files

def normalize_joint_sequence(js_any: Any) -> Optional[List[str]]:
    if js_any is None:
        return None
    try:
        if isinstance(js_any, np.ndarray):
            if js_any.ndim == 0:
                v = js_any.item()
                if isinstance(v, (list, tuple, np.ndarray)):
                    return [str(x) for x in list(v)]
                elif isinstance(v, (str, np.str_)):
                    return [str(v)]
                else:
                    return None
            if js_any.ndim == 1 and (js_any.dtype == object or np.issubdtype(js_any.dtype, np.str_)):
                out = [str(x) for x in js_any.tolist()]
                return out if all(isinstance(x, str) for x in out) else None
            out = list(js_any.tolist())
            return [str(x) for x in out] if all(isinstance(x, (str, np.str_)) for x in out) else None
        if isinstance(js_any, (list, tuple)):
            out = [str(x) for x in js_any]
            return out if all(isinstance(x, str) for x in out) else None
        if isinstance(js_any, (str, np.str_)):
            return [str(js_any)]
    except Exception:
        return None
    return None

def to_time_first(a: np.ndarray, D_guess: Optional[int] = None) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {a.shape}")
    if D_guess is not None:
        if a.shape[1] == D_guess:
            return a
        if a.shape[0] == D_guess:
            return a.T
    # Heuristic: if first dimension is smaller (usually channels), then transpose
    return a.T if a.shape[0] <= a.shape[1] else a

def to_object_array(seq: List[Any]) -> np.ndarray:
    arr = np.empty(len(seq), dtype=object)
    for i, v in enumerate(seq):
        arr[i] = v
    return arr

def build_canonical_joint_names(candidates: List[List[str]], fallback_D: int) -> List[str]:
    if not candidates:
        print(f"No joint_sequence found in any file; using placeholder names of length {fallback_D}.")
        return [f"joint_{i}" for i in range(fallback_D)]
    # Pick the longest sequences first
    max_len = max(len(c) for c in candidates)
    long_ones = [tuple(c) for c in candidates if len(c) == max_len]
    # Among them, choose the most frequent tuple
    cnt: Dict[Tuple[str, ...], int] = {}
    for c in long_ones:
        cnt[c] = cnt.get(c, 0) + 1
    canon = list(max(cnt.items(), key=lambda x: x[1])[0])
    print(f"Canonical joint_sequence picked. Length={len(canon)}")
    return canon

def remap_series_to_target(series: np.ndarray, src_names: Optional[List[str]], target_names: List[str]) -> Optional[np.ndarray]:
    """
    Map a (T, D_src) series to (T, D_tgt) by joint name.
    - If src_names is None and D_src == D_tgt: keep as-is.
    - If src_names provided: fill zeros then copy matching columns by name.
    - If src_names is None and D_src != D_tgt: return None (cannot map).
    """
    T, D_src = series.shape
    D_tgt = len(target_names)
    if src_names is None:
        if D_src == D_tgt:
            return series
        return None

    # Build name -> index for source and target
    name_to_src = {n: i for i, n in enumerate(src_names)}
    out = np.zeros((T, D_tgt), dtype=series.dtype)
    missing = []
    for j, name in enumerate(target_names):
        if name in name_to_src:
            out[:, j] = series[:, name_to_src[name]]
        else:
            missing.append(name)
    if missing:
        # Only informational; many datasets don't have every joint present
        pass
    return out

def main():
    parser = argparse.ArgumentParser(description="Merge npz files under a root. Assumes all files have the same joint dimension.")
    parser.add_argument("-r", "--root", type=str, default="output_files", help="npz root directory (default: output_files)")
    parser.add_argument("-o", "--out", type=str, default="merged_npz", help="output directory (default: merged_npz)")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"Root directory does not exist: {root}")
        return

    files = find_npz_files(root)
    print(f"Found npz files: {len(files)}")

    # Simplified data collection
    merged_pos, merged_pos_cmd, merged_vel, merged_tor = [], [], [], []
    final_joint_sequence: Optional[List[str]] = None
    final_D: Optional[int] = None

    used = 0
    skipped = 0
    total = len(files)

    for i, f in enumerate(files, start=1):
        try:
            with np.load(f, allow_pickle=True) as npzf:
                req = ["real_dof_positions", "real_dof_positions_cmd", "real_dof_velocities", "real_dof_torques"]
                if not all(k in npzf.files for k in req):
                    skipped += 1
                    print(f"[{i}/{total}] Skip (missing key fields): {f}")
                    continue

                # Grab joint sequence from the first valid file and set the expected dimension
                if final_joint_sequence is None:
                    if "joint_sequence" in npzf.files:
                        final_joint_sequence = normalize_joint_sequence(npzf["joint_sequence"])
                        if final_joint_sequence:
                            final_D = len(final_joint_sequence)
                            print(f"Using joint wwquence from {f} as canonical. Dimension set to {final_D}.")

                pos = to_time_first(npzf["real_dof_positions"], D_guess=final_D)
                pos_cmd = to_time_first(npzf["real_dof_positions_cmd"], D_guess=final_D)
                vel = to_time_first(npzf["real_dof_velocities"], D_guess=final_D)
                tor = to_time_first(npzf["real_dof_torques"], D_guess=final_D)

                # Basic dimension check
                if final_D is not None and pos.shape[1] != final_D:
                     skipped += 1
                     print(f"[{i}/{total}] Skip (dimension mismatch): {f} has D={pos.shape[1]}, expected {final_D}")
                     continue

                merged_pos.append(pos)
                merged_pos_cmd.append(pos_cmd)
                merged_vel.append(vel)
                merged_tor.append(tor)
                used += 1
                print(f"[{i}/{total}] Collected: {f} -> shapes pos={pos.shape}, vel={vel.shape}, tor={tor.shape}")

        except Exception as e:
            skipped += 1
            print(f"[{i}/{total}] Skip (read failed): {f} ({e})")

    if used == 0:
        print("No valid samples to merge.")
        return

    save_path = out_dir / "merged_all.npz"
    
    # Use an empty list if no joint sequence was ever found
    if final_joint_sequence is None:
        final_joint_sequence = []
        print("Warning: No 'joint_sequence' found in any of the npz files.")

    np.savez_compressed(
        save_path,
        real_dof_positions=to_object_array(merged_pos),
        real_dof_positions_cmd=to_object_array(merged_pos_cmd),
        real_dof_velocities=to_object_array(merged_vel),
        real_dof_torques=to_object_array(merged_tor),
        joint_sequence=np.asarray(final_joint_sequence, dtype=object),
    )
    print(f"Saved merged file: {save_path} (samples kept: {used}, total_read: {used}, data columns: {final_D or 'Unknown'})")
    print(f"Complete. Files scanned: {total}, Read ok: {used}, Skipped read: {skipped}")
    
if __name__ == "__main__":
    main()