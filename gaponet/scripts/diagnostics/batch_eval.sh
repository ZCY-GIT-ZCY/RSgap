#!/usr/bin/env bash
# Batch evaluate GAPONet compensation for episodes START..END
# Usage:
#   bash batch_eval.sh                         # episodes 0-9 (default)
#   bash batch_eval.sh 0 9 /path/to/model.pt

# Activate the gapo conda environment (has isaacsim + isaaclab)
eval "$(conda shell.bash hook)"
conda activate gapo || { echo "[ERROR] Failed to activate conda env 'gapo'"; exit 1; }

START=${1:-0}
END=${2:-9}
CHECKPOINT=${3:-"logs/rsl_rl/humanoid_agibot/2026-03-05_20-37-07-best/model_58500.pt"}

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
RESULT_FILE="$ROOT_DIR/logs/train_result/batch_summary_ep${START}-${END}.txt"

export PYTHONPATH="$ROOT_DIR/gaponet/source/sim2real:$ROOT_DIR/gaponet/source/sim2real_assets:${PYTHONPATH:-}"

echo "========================================="
echo " Batch Evaluation: episodes $START to $END"
echo " Checkpoint: $CHECKPOINT"
echo " Results -> $RESULT_FILE"
echo "========================================="

# clear previous summary file
> "$RESULT_FILE"

for IDX in $(seq "$START" "$END"); do
    echo ""
    echo "----- Episode $IDX -----"
    TMPFILE=$(mktemp)
    PYTHONUNBUFFERED=1 python -u "$ROOT_DIR/gaponet/scripts/diagnostics/evaluate_compensation.py" \
        --task Isaac-Humanoid-AGIBOT-Delta-Action \
        --motion-index "$IDX" \
        --checkpoint "$CHECKPOINT" \
        --headless \
        --full-episode \
        --model-based-sensor false \
        --mode play \
        2>&1 | tee "$TMPFILE"
    EXIT_CODE=${PIPESTATUS[0]}

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "[WARN] Episode $IDX exited with code $EXIT_CODE"
        echo "[SUMMARY] episode=$IDX ERROR: python exited with code $EXIT_CODE" >> "$RESULT_FILE"
        rm -f "$TMPFILE"
        continue
    fi

    # Extract and save the summary line directly from file (avoids large-variable / buffering issues)
    SUMMARY=$(grep "^\[SUMMARY\]" "$TMPFILE")
    if [ -n "$SUMMARY" ]; then
        echo "$SUMMARY" >> "$RESULT_FILE"
    else
        echo "[SUMMARY] episode=$IDX ERROR: no summary line found" >> "$RESULT_FILE"
    fi
    rm -f "$TMPFILE"
done

echo ""
echo "========================================="
echo " FINAL RESULTS (episodes $START-$END)"
echo "========================================="
cat "$RESULT_FILE"

echo ""
echo "--- Averages ---"
python3 - <<'PYEOF'
import re, sys

result_file = sys.argv[1] if len(sys.argv) > 1 else None

import os
# Find the result file from env
for f in os.listdir("logs/train_result"):
    if f.startswith("batch_summary"):
        result_file = f"logs/train_result/{f}"
        break

lines = open(result_file).readlines()
baselines, comps = [], []
for line in lines:
    m = re.search(r"baseline=([\d.]+)deg.*compensated=([\d.]+)deg", line)
    if m:
        baselines.append(float(m.group(1)))
        comps.append(float(m.group(2)))

if baselines:
    avg_base = sum(baselines) / len(baselines)
    avg_comp = sum(comps) / len(comps)
    impr = (avg_base - avg_comp) / avg_base * 100
    print(f"Episodes evaluated : {len(baselines)}")
    print(f"Mean baseline gap  : {avg_base:.4f} deg")
    print(f"Mean compensated gap: {avg_comp:.4f} deg")
    print(f"Mean improvement   : {impr:.2f}%")
else:
    print("No valid summary lines found.")
PYEOF
