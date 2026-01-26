"""Visualize sim/real trajectories with and without gaponet compensation.

Loads a fixed motion episode, rolls out baseline sim (zero delta-action) and
compensated sim (policy checkpoint), then plots per-joint curves with errors.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import torch

from isaaclab.app import AppLauncher

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make rsl_rl cli args available
_RSL_RL_PATH = Path(__file__).resolve().parents[1] / "reinforcement_learning" / "rsl_rl"
sys.path.append(str(_RSL_RL_PATH))
import cli_args  # noqa: E402

import sim2real.tasks.humanoid_operator  # noqa: F401
import sim2real.tasks.humanoid_agibot  # noqa: F401
from sim2real.rsl_rl.modules import DeepONetActorCritic  # noqa: E402
import rsl_rl.runners.on_policy_runner as on_policy_runner  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

# Make DeepONetActorCritic discoverable by rsl_rl eval()
on_policy_runner.DeepONetActorCritic = DeepONetActorCritic


def _build_output_dir(base_dir: Path, episode_idx: int) -> Path:
    out_dir = base_dir / f"episode{episode_idx:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _sync_to_motion(env_unwrapped, motion_indices: torch.Tensor, time_indices: torch.Tensor) -> None:
    motion_pos = env_unwrapped._motion_loader.dof_positions[motion_indices, time_indices]
    motion_vel = torch.zeros_like(motion_pos)
    env_unwrapped.robot.write_joint_state_to_sim(
        motion_pos, motion_vel, joint_ids=env_unwrapped.motion_joint_ids
    )
    env_unwrapped.robot.set_joint_position_target(
        motion_pos, joint_ids=env_unwrapped.motion_joint_ids
    )
    env_unwrapped._raw_step_simulator()


def _update_sensor_data_from_sim(env_unwrapped) -> None:
    joint_pos = env_unwrapped.robot.data.joint_pos[:, env_unwrapped.motion_joint_ids]
    joint_vel = env_unwrapped.robot.data.joint_vel[:, env_unwrapped.motion_joint_ids]
    sensor = torch.cat([joint_pos, joint_vel * env_unwrapped.step_dt], dim=1)
    sensor = sensor.view(env_unwrapped.num_envs, env_unwrapped.num_sensor_positions, env_unwrapped.cfg.sensor_dim)
    env_unwrapped.set_sensor_data(sensor)


def _collect_rollout(
    env,
    num_steps: int,
    motion_index: int,
    time_index: int,
    use_policy: bool,
    policy=None,
    sync_to_motion: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    env_unwrapped = env.unwrapped
    device = env_unwrapped.device
    num_actions = env_unwrapped.cfg.action_space

    motion_indices = torch.full((env_unwrapped.num_envs,), motion_index, dtype=torch.long, device=device)
    time_indices = torch.full((env_unwrapped.num_envs,), time_index, dtype=torch.long, device=device)

    if sync_to_motion:
        _sync_to_motion(env_unwrapped, motion_indices, time_indices)

    zero_action = torch.zeros((env_unwrapped.num_envs, num_actions), device=device)

    sim_traj: list[np.ndarray] = []
    real_traj: list[np.ndarray] = []

    for _ in range(num_steps):
        time_indices_step = time_indices.clone()
        if use_policy:
            with torch.inference_mode():
                _update_sensor_data_from_sim(env_unwrapped)
                obs_dict = env_unwrapped.compute_operator_observation()
                obs = torch.cat([obs_dict["branch"], obs_dict["trunk"]], dim=1)
                actions = policy(obs)
        else:
            actions = zero_action

        _, _, dones, _ = env_unwrapped.step_operator(actions, motion_coords=(motion_indices, time_indices_step))

        motion_indices = env_unwrapped.motion_indices.clone()
        time_indices = env_unwrapped.time_indices.clone()

        sim_pos = env_unwrapped.robot.data.joint_pos[:, env_unwrapped.motion_joint_ids].detach().cpu().numpy()
        real_pos = env_unwrapped._motion_loader.dof_positions[
            motion_indices, time_indices_step
        ].detach().cpu().numpy()

        sim_traj.append(sim_pos[0])
        real_traj.append(real_pos[0])

        if bool(torch.any(dones)):
            break

    return np.stack(sim_traj, axis=0), np.stack(real_traj, axis=0)


def _plot_joint_curves(
    out_dir: Path,
    dof_names: list[str],
    sim_arr: np.ndarray,
    sim_comp_arr: np.ndarray,
    real_arr: np.ndarray,
    joint_indices: np.ndarray,
) -> None:
    steps = np.arange(real_arr.shape[0])
    sim_arr_deg = np.degrees(sim_arr)
    sim_comp_arr_deg = np.degrees(sim_comp_arr)
    real_arr_deg = np.degrees(real_arr)

    for idx in joint_indices:
        sim_line = sim_arr_deg[:, idx]
        sim_comp_line = sim_comp_arr_deg[:, idx]
        real_line = real_arr_deg[:, idx]

        err_sim = np.abs(sim_line - real_line)
        err_comp = np.abs(sim_comp_line - real_line)

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax0.plot(steps, sim_line, label="sim")
        ax0.plot(steps, sim_comp_line, label="sim+delta")
        ax0.plot(steps, real_line, label="real")
        ax0.set_title(f"{dof_names[idx]} (deg)")
        ax0.set_ylabel("position (deg)")
        ax0.legend()

        ax1.plot(steps, err_sim, label="|sim-real|")
        ax1.plot(steps, err_comp, label="|sim+delta-real|")
        ax1.set_xlabel("step")
        ax1.set_ylabel("|error| (deg)")
        ax1.legend()

        out_path = out_dir / f"{dof_names[idx]}_traj.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def _load_policy(env, args_cli) -> tuple[OnPolicyRunner, callable, str]:
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            raise RuntimeError("No published pretrained checkpoint available for this task.")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    env_wrapped = RslRlVecEnvWrapper(env)
    ppo_runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    return ppo_runner, policy, resume_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate gaponet compensation on a motion episode.")
    parser.add_argument("--task", type=str, default="Isaac-Humanoid-AGIBOT-Delta-Action", help="Gym task id.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of envs to create.")
    parser.add_argument("--motion-index", type=int, default=0, help="Motion episode index.")
    parser.add_argument("--time-index", type=int, default=0, help="Start time index in the episode.")
    parser.add_argument("--num-steps", type=int, default=200, help="Number of steps to simulate.")
    parser.add_argument("--full-episode", action="store_true", help="Override num-steps to the full episode length.")
    parser.add_argument("--sync-to-motion", action="store_true", help="Sync sim joints to real pose at start.")
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="logs/train_result",
        help="Directory to save plots (episode folder will be created inside).",
    )
    parser.add_argument(
        "--use_pretrained_checkpoint",
        action="store_true",
        help="Use the published pretrained checkpoint (if available).",
    )
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    # append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)

    args_cli = parser.parse_args()

    # Launch Isaac Sim
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # build env for baseline
    env_base = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env_base.unwrapped, DirectMARLEnv):
        env_base = multi_agent_to_single_agent(env_base)

    env_unwrapped = env_base.unwrapped

    # resolve motion length
    if args_cli.full_episode:
        motion_len = int(env_unwrapped._motion_loader.motion_len[args_cli.motion_index].item())
        num_steps = motion_len
    else:
        num_steps = args_cli.num_steps

    # baseline rollout (zero delta)
    sim_arr, real_arr = _collect_rollout(
        env_base,
        num_steps=num_steps,
        motion_index=args_cli.motion_index,
        time_index=args_cli.time_index,
        use_policy=False,
        policy=None,
        sync_to_motion=args_cli.sync_to_motion,
    )

    env_base.close()

    # build env for compensated rollout
    env_comp = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env_comp.unwrapped, DirectMARLEnv):
        env_comp = multi_agent_to_single_agent(env_comp)

    # load policy
    _, policy, resume_path = _load_policy(env_comp, args_cli)

    sim_comp_arr, real_arr_comp = _collect_rollout(
        env_comp,
        num_steps=num_steps,
        motion_index=args_cli.motion_index,
        time_index=args_cli.time_index,
        use_policy=True,
        policy=policy,
        sync_to_motion=args_cli.sync_to_motion,
    )

    env_comp.close()
    simulation_app.close()

    # align lengths
    min_len = min(sim_arr.shape[0], sim_comp_arr.shape[0], real_arr.shape[0], real_arr_comp.shape[0])
    sim_arr = sim_arr[:min_len]
    sim_comp_arr = sim_comp_arr[:min_len]
    real_arr = real_arr[:min_len]

    # plot 18 joints
    dof_names = list(env_unwrapped._motion_loader.dof_names)
    joint_count = min(18, len(dof_names))
    joint_indices = np.arange(joint_count)

    out_dir = _build_output_dir(Path(args_cli.plot_dir), args_cli.motion_index)
    _plot_joint_curves(out_dir, dof_names, sim_arr, sim_comp_arr, real_arr, joint_indices)

    print(f"[INFO] checkpoint: {resume_path}")
    print(f"[INFO] plots saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
