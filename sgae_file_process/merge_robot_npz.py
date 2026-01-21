"""
Robot Motion Data Merger

This script merges scattered robot motion data files (.npz format) into standardized datasets 
for machine learning and analysis. Key features include:

1. Data Collection & Classification:
   - Recursively searches for .npz files in specified directories
   - Automatically detects sampling frequencies (50Hz/100Hz) from file paths or content
   - Classifies and collects data by frequency

2. Data Standardization:
   - Aligns time axis to (T, D) format (time first, features second)
   - Normalizes joint sequence formats
   - Extracts payload information from filenames (e.g., "5kg")

3. Dimension Expansion:
   - Expands original 10 effective joint data to 27-dimension standard URDF format
   - Only fills actual data for these 10 joints, sets others to zero:
     left/right_shoulder_pitch/roll/yaw_joint, left/right_elbow_joint, left/right_wrist_roll_joint

4. Output Generation:
   - Produces standardized merged files: merged_50Hz.npz and merged_100Hz.npz
   - Output format aligns with test.npz structure for downstream processing

Usage: python col_npz.py -r output_files -o merged_npz
"""

import os
import re
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

def detect_freq_from_path(path: Path) -> Optional[int]:
    """Parse 50Hz/100Hz from various directory levels in the path"""
    hz_pat = re.compile(r"(\d+)\s*Hz", re.IGNORECASE)
    for part in path.parts:
        m = hz_pat.search(part)
        if m:
            try:
                val = int(m.group(1))
                if val in (50, 100):
                    return val
            except ValueError:
                pass
    return None

def find_npz_files(root: Path) -> List[Path]:
    """Recursively search for all .npz files under root directory"""
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".npz"):
                files.append(Path(dirpath) / f)
    return files

# New: Extract payload (kg) as integer (rounded) from string; return None if failed
_KG_PAT = re.compile(r'(\d+(?:\.\d+)?)\s*kg(?=[/_\-\s]|$)', re.IGNORECASE)
def extract_payload_int_kg(name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    s = str(name)
    m = _KG_PAT.search(s)
    if not m:
        return None
    try:
        val = float(m.group(1))
        return int(round(val))
    except Exception:
        return None

# New: Normalize joint_sequence to string list
def normalize_joint_sequence(js_any: Any) -> Optional[List[str]]:
    if js_any is None:
        return None
    try:
        # Common forms: np.ndarray(dtype=object or <U..), list, tuple
        if isinstance(js_any, np.ndarray):
            if js_any.ndim == 0:
                v = js_any.item()
                if isinstance(v, (list, tuple, np.ndarray)):
                    return [str(x) for x in list(v)]
                elif isinstance(v, (str, np.str_)):
                    return [str(v)]
                else:
                    return None
            # 1D string array
            if js_any.ndim == 1 and (js_any.dtype == object or np.issubdtype(js_any.dtype, np.str_)):
                out = [str(x) for x in js_any.tolist()]
                return out if all(isinstance(x, str) for x in out) else None
            # Other forms try tolist
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

# New: Convert any 2D array to (T, D); prioritize judgment based on D_guess
def to_time_first(a: np.ndarray, D_guess: Optional[int] = None) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {a.shape}")
    if D_guess is not None:
        if a.shape[1] == D_guess:
            return a
        if a.shape[0] == D_guess:
            return a.T
    # Heuristic: if first dimension is smaller (channel count usually < time length), consider it (D, T), need transpose
    return a.T if a.shape[0] <= a.shape[1] else a

# Target URDF dimension and effective joint target indices
D_TARGET = 27
TARGET_IDX_BY_NAME = {
    "left_shoulder_pitch_joint": 5,
    "right_shoulder_pitch_joint": 6,
    "left_shoulder_roll_joint": 9,
    "right_shoulder_roll_joint": 10,
    "left_shoulder_yaw_joint": 13,
    "right_shoulder_yaw_joint": 14,
    "left_elbow_joint": 17,
    "right_elbow_joint": 18,
    "left_wrist_roll_joint": 21,
    "right_wrist_roll_joint": 22,
}

def expand_series_to_27(series_list, src_names):
    """
    Expand a group of (T, D_src) sequences to (T, 27).
    Only copy columns from src_names that match TARGET_IDX_BY_NAME to corresponding target indices, set others to 0.
    """
    name_to_src = {name: i for i, name in enumerate(src_names)}
    out = []
    missing = []
    for a in series_list:
        if not isinstance(a, np.ndarray) or a.ndim != 2:
            raise ValueError(f"Expected 2D array, got: {type(a)}, shape={getattr(a, 'shape', None)}")
        T, _ = a.shape
        b = np.zeros((T, D_TARGET), dtype=a.dtype)
        for name, dst_idx in TARGET_IDX_BY_NAME.items():
            if name in name_to_src:
                src_idx = name_to_src[name]
                b[:, dst_idx] = a[:, src_idx]
            else:
                # Record missing names, only count first time
                missing.append(name)
        out.append(b)
    if missing:
        # Only prompt once (deduplicated)
        miss_set = sorted(set(missing))
        print(f"[Warning] Following joints missing in source joint_sequence, will remain 0: {miss_set}")
    return out

def main():
    parser = argparse.ArgumentParser(description="Merge npz files with same frequency under output_files to merged_npz (aligned with test.npz format).")
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

    # Changed to only collect keys needed for test.npz alignment
    buckets: Dict[int, Dict[str, List[Any]]] = {
        50: {"positions": [], "positions_cmd": [], "velocities": [], "torques": [],
             "payloads": [], "joint_candidates": []},
        100: {"positions": [], "positions_cmd": [], "velocities": [], "torques": [],
              "payloads": [], "joint_candidates": []},
    }

    used = 0
    skipped = 0
    total = len(files)
    for i, f in enumerate(files, start=1):
        freq = detect_freq_from_path(f)
        if freq is None:
            # Try to read frequency field from file
            try:
                with np.load(f, allow_pickle=True) as npzf:
                    freq_inside = int(npzf["frequency"]) if "frequency" in npzf.files else None
                    if freq_inside in (50, 100):
                        freq = freq_inside
            except Exception:
                pass

        if freq not in (50, 100):
            skipped += 1
            print(f"[{i}/{total}] Skip (unable to identify frequency): {f}")
            continue

        try:
            with np.load(f, allow_pickle=True) as npzf:
                # Required keys
                req = ["real_dof_positions", "real_dof_positions_cmd", "real_dof_velocities", "real_dof_torques"]
                if not all(k in npzf.files for k in req):
                    skipped += 1
                    print(f"[{i}/{total}] Skip (missing key fields): {f}")
                    continue

                # joint_sequence candidates
                jlist = normalize_joint_sequence(npzf["joint_sequence"]) if "joint_sequence" in npzf.files else None
                if jlist:
                    buckets[freq]["joint_candidates"].append(jlist)

                D_guess = len(jlist) if jlist else None

                # Convert to (T, D)
                pos = to_time_first(npzf["real_dof_positions"], D_guess=D_guess)
                pos_cmd = to_time_first(npzf["real_dof_positions_cmd"], D_guess=D_guess)
                vel = to_time_first(npzf["real_dof_velocities"], D_guess=D_guess)
                tor = to_time_first(npzf["real_dof_torques"], D_guess=D_guess)

                # Parse payloads (from motion_name's Xkg)
                motion_name = str(npzf["motion_name"]) if "motion_name" in npzf.files else None
                payload = extract_payload_int_kg(motion_name)
                if payload is None:
                    # If unable to parse, set to -1 and prompt
                    payload = -1
                    print(f"[{i}/{total}] Warning: Unable to extract payload from motion_name, set to -1: {f}")

                # Append
                buckets[freq]["positions"].append(pos)
                buckets[freq]["positions_cmd"].append(pos_cmd)
                buckets[freq]["velocities"].append(vel)
                buckets[freq]["torques"].append(tor)
                buckets[freq]["payloads"].append(payload)
                used += 1
                print(f"[{i}/{total}] Collected: {f} -> {freq}Hz, shapes pos={pos.shape}, vel={vel.shape}, tor={tor.shape}, payload={payload}")
        except Exception as e:
            skipped += 1
            print(f"[{i}/{total}] Skip (read failed): {f} ({e})")

    # Save merged results for both frequencies (aligned with test.npz)
    for freq in (50, 100):
        data = buckets[freq]
        N = len(data["positions"])
        if N == 0:
            print(f"{freq}Hz has no available samples, skip saving.")
            continue

        def to_object_array(seq: List[Any]) -> np.ndarray:
            arr = np.empty(len(seq), dtype=object)
            for i, v in enumerate(seq):
                arr[i] = v
            return arr

        # Select global joint_sequence: take the most frequent candidate; if no candidates, construct placeholder names based on first sample's D
        canon_js: List[str]
        candidates: List[Tuple[str, ...]] = [tuple(js) for js in data["joint_candidates"] if js]
        if candidates:
            cnt = {}
            for c in candidates:
                cnt[c] = cnt.get(c, 0) + 1
            canon_js = list(max(cnt.items(), key=lambda x: x[1])[0])
        else:
            D = data["positions"][0].shape[1]
            canon_js = [f"joint_{i}" for i in range(D)]
            print(f"{freq}Hz no joint_sequence found, using placeholder names: D={D}")

        # If current data column count is not 27, and we have 10 effective joint names, perform expansion to 27
        cur_D = data["positions"][0].shape[1]
        if cur_D != D_TARGET and len(canon_js) == 10:
            print(f"{freq}Hz performing expansion to {D_TARGET} dimensions (other channels set to 0)...")
            data["positions"]      = expand_series_to_27(data["positions"],      canon_js)
            data["positions_cmd"]  = expand_series_to_27(data["positions_cmd"],  canon_js)
            data["velocities"]     = expand_series_to_27(data["velocities"],     canon_js)
            data["torques"]        = expand_series_to_27(data["torques"],        canon_js)
            final_D = D_TARGET
        else:
            final_D = cur_D

        save_path = out_dir / f"merged_{freq}Hz.npz"
        np.savez_compressed(
            save_path,
            real_dof_positions=to_object_array(data["positions"]),
            real_dof_positions_cmd=to_object_array(data["positions_cmd"]),
            real_dof_velocities=to_object_array(data["velocities"]),
            real_dof_torques=to_object_array(data["torques"]),
            joint_sequence=np.asarray(canon_js, dtype=object),  # Keep 10 effective joint names
            payloads=np.asarray(data["payloads"], dtype=np.int64),
        )
        print(f"Saved merged file (aligned with test.npz + 27-dim): {save_path} (samples: {N}, joint names: {len(canon_js)}, data columns: {final_D})")

    print(f"Complete. Used: {used}, Skipped: {skipped}, Total: {len(files)}")

if __name__ == "__main__":
    main()