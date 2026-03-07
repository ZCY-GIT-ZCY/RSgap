#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"

# Activate the gapo conda environment (has isaacsim + isaaclab)
eval "$(conda shell.bash hook)"
conda activate gapo

# Keep script directory first so local helper modules can be imported.
export PYTHONPATH="$ROOT_DIR/gaponet/scripts/diagnostics/sim_vis_compare:$ROOT_DIR/sgae_file_process/ignore/sage/sage:${PYTHONPATH:-}"

python "$ROOT_DIR/gaponet/scripts/diagnostics/sim_vis_compare/vis_gaponet_comparison.py" \
    --robot-name agibot \
    --precomputed-file "$ROOT_DIR/logs/precompute_data/nonesensor_43.npz" \
    --group-offset 2 \
    --livestream 0 \
    --max-frames -1  \
    --save-video \
    --video-path "$ROOT_DIR/logs/precompute_data/nonesensor_43.mp4"