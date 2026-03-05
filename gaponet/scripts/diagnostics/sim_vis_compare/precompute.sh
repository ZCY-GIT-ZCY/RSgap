#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
export PYTHONPATH="$ROOT_DIR/gaponet/scripts/reinforcement_learning/rsl_rl:$ROOT_DIR/gaponet/source/sim2real:$ROOT_DIR/gaponet/source/sim2real_assets:${PYTHONPATH:-}"

python "$ROOT_DIR/gaponet/scripts/diagnostics/sim_vis_compare/precompute_delta_actions.py" \
    --task Isaac-Humanoid-AGIBOT-Delta-Action \
    --model "$ROOT_DIR/logs/rsl_rl/humanoid_agibot/2026-03-04_20-08-04-nosensormodel/model_6000.pt" \
    --motion_file "$ROOT_DIR/gaponet/source/sim2real/sim2real/tasks/humanoid_agibot/motions/motion_amass/agibot_g1/motion_agibot.npz" \
    --motion_idx 0 \
    --output_file "$ROOT_DIR/logs/precompute_data/nonesensor.npz"