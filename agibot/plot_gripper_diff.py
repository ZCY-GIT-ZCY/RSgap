import argparse
import matplotlib.pyplot as plt
import os
import csv
from scripts.data_utils import DataLoader

def plot_and_save(frame_indices, diff_l, diff_r, episode_idx, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(frame_indices, diff_l, label='left_gripper observation[0] - action[0]')
    plt.plot(frame_indices, diff_r, label='right_gripper observation[1] - action[1]')
    plt.xlabel('frame_index')
    plt.ylabel('diff')
    plt.title(f'Gripper Observation-Action Difference (Episode {episode_idx})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'gripper_diff_ep{episode_idx:03d}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"[保存] {save_path}")

def save_table(frame_indices, diff_l, diff_r, episode_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    table_path = os.path.join(output_dir, f'gripper_diff_ep{episode_idx:03d}.csv')
    with open(table_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame_index', 'left_gripper_diff', 'right_gripper_diff'])
        for idx, dl, dr in zip(frame_indices, diff_l, diff_r):
            writer.writerow([idx, dl, dr])
    print(f"[保存] {table_path}")

def main():
    parser = argparse.ArgumentParser(description="夹爪观测-目标差值分析 (批量)")
    parser.add_argument('--dataset', type=str, required=True, help='数据集路径')
    parser.add_argument('--output-dir', type=str, default='output_gripper_diff', help='输出图片文件夹')
    args = parser.parse_args()

    loader = DataLoader(args.dataset)
    episode_count = loader.get_episode_count()
    print(f"[Info] 共 {episode_count} 个 episode，将全部处理...")

    for ep in range(episode_count):
        try:
            frames = loader.load_episode(ep)
        except Exception as e:
            print(f"[跳过] episode {ep}: {e}")
            continue

        frame_indices = [f.frame_index for f in frames]
        obs_l = [f.observation_state[0] for f in frames]
        obs_r = [f.observation_state[1] for f in frames]
        act_l = [f.action[0] for f in frames]
        act_r = [f.action[1] for f in frames]

        diff_l = [o - a for o, a in zip(obs_l, act_l)]
        diff_r = [o - a for o, a in zip(obs_r, act_r)]

        plot_and_save(frame_indices, diff_l, diff_r, ep, args.output_dir)
        save_table(frame_indices, diff_l, diff_r, ep, args.output_dir)

if __name__ == "__main__":
    main()