# python precompute_delta_actions.py \
#     --task Isaac-SO101-Operator-Delta-Action \
#     --model /cpfs/workspace/data/pretrained_models/sim2real/gaponet/logs/rsl_rl/so101_operator/2026-02-02_08-43-39_run_009_nosim_all_v2_1/model_8000.pt \
#     --motion_file /cpfs/workspace/code/sage_lerobot/gaponet/data/so101/batch_processed_all_nosim_v2/test.npz \
#     --motion_idx 0 \
#     --output_file /cpfs/workspace/code/sage_lerobot/gaponet/scripts/sim_vis_gap/precomputed_delta.npz

python precompute_delta_actions.py \
    --task Isaac-SO101-Operator-Delta-Action \
    --model /cpfs/workspace/data/pretrained_models/sim2real/gaponet/logs/rsl_rl/so101_operator/2026-02-09_17-38-24_run_013_nosim_with_random_v1_3/model_51000.pt \
    --motion_file /cpfs/workspace/code/sage_lerobot/gaponet/data/so101/batch_processed_all_nosim_v2/test.npz \
    --motion_idx 0 \
    --output_file /cpfs/workspace/code/sage_lerobot/gaponet/scripts/sim_vis_gap/precomputed_delta_run013_51000step.npz