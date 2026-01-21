import argparse
import numpy as np
import os
from scripts.data_utils import DataLoader

def compute_rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=0))  # 每一维

def compute_corr(a, b):
    # 每一维分别算相关系数
    corrs = []
    for i in range(a.shape[1]):
        if np.std(a[:, i]) < 1e-8 or np.std(b[:, i]) < 1e-8:
            corrs.append(0.0)
        else:
            corrs.append(np.corrcoef(a[:, i], b[:, i])[0, 1])
    return np.array(corrs)

def analyze_episode(frames, max_shift=2):
    # 只看关节：observation[54-67] vs action[16-29]，共14维
    obs = np.stack([f.observation_state[54:68] for f in frames])  # shape (N,14)
    act = np.stack([f.action[16:30] for f in frames])             # shape (N,14)
    n = len(frames)
    results = []
    for shift in range(-max_shift, max_shift+1):
        if shift < 0:
            obs_shifted = obs[-shift:]
            act_shifted = act[:n+shift]
        elif shift > 0:
            obs_shifted = obs[:n-shift]
            act_shifted = act[shift:]
        else:
            obs_shifted = obs
            act_shifted = act
        rmse = compute_rmse(obs_shifted, act_shifted)  # shape (14,)
        corr = compute_corr(obs_shifted, act_shifted)  # shape (14,)
        results.append((shift, rmse, corr))
    return results

def main():
    parser = argparse.ArgumentParser(description="验证 Action-State 关节时序对齐关系")
    parser.add_argument('--dataset', type=str, required=True, help='数据集路径')
    parser.add_argument('--output', type=str, default=None, help='结果保存路径（可选）')
    parser.add_argument('--max-shift', type=int, default=2, help='最大平移帧数')
    args = parser.parse_args()

    loader = DataLoader(args.dataset)
    episode_count = loader.get_episode_count()
    all_results = []

    print(f"[Info] 共 {episode_count} 个 episode，开始分析...")
    for ep in range(episode_count):
        try:
            frames = loader.load_episode(ep)
        except Exception as e:
            print(f"[跳过] episode {ep}: {e}")
            continue
        results = analyze_episode(frames, max_shift=args.max_shift)
        # 统计每个shift的均方根误差均值
        mean_rmses = [rmse.mean() for shift, rmse, corr in results]
        best_shift = np.argmin(mean_rmses) - args.max_shift
        print(f"Episode {ep:03d}:")

        print(f"  最小RMSE均值对应 shift={best_shift:+d}\n")
        all_results.append((ep, results))

    # 可选：保存结果
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            for ep, results in all_results:
                f.write(f"Episode {ep:03d}:\n")
                for shift, rmse, corr in results:
                    f.write(f"  shift={shift:+d}: RMSE均值={rmse.mean():.6f}, 每维RMSE={np.round(rmse,4)}, 每维Corr={np.round(corr,4)}\n")
                mean_rmses = [rmse.mean() for shift, rmse, corr in results]
                best_shift = np.argmin(mean_rmses) - args.max_shift
                f.write(f"  最小RMSE均值对应 shift={best_shift:+d}\n\n")
        print(f"[保存] 结果已写入 {args.output}")

if __name__ == "__main__":
    main()