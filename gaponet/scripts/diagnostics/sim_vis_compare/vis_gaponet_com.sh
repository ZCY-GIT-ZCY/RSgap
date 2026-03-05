# python vis_gaponet_comparison.py \
#     --task Isaac-SO101-Operator-Delta-Action \
#     --model /cpfs/workspace/data/pretrained_models/sim2real/gaponet/logs/rsl_rl/so101_operator/2026-02-02_08-43-39_run_009_nosim_all_v2_1/model_8000.pt \
#     --motion_file /cpfs/workspace/code/sage_lerobot/gaponet/data/so101/batch_processed_all_nosim_v2/test.npz \
#     --motion_idx 0 \
#     --group_offset 0.8  # 两组之间的间距

python vis_gaponet_comparison.py \
    --robot-name so101 \
    --precomputed-file /cpfs/workspace/code/sage_lerobot/gaponet/scripts/sim_vis_gap/precomputed_delta_run013_51000step.npz \
    --group-offset 0.5