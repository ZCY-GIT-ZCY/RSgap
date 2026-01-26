"""Evaluate GAPOnet compensation performance by comparing Sim, Sim+Delta, and Real trajectories."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import importlib.util
import types

import gymnasium as gym
import matplotlib
import numpy as np
import torch

from isaaclab.app import AppLauncher

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add rsl_rl script directory to path to import cli_args
rsl_rl_script_dir = Path(__file__).resolve().parents[2] / "scripts" / "reinforcement_learning" / "rsl_rl"
sys.path.append(str(rsl_rl_script_dir))

try:
    import cli_args
except ImportError:
    print(f"Warning: Could not import cli_args from {rsl_rl_script_dir}. Some arguments might be missing.")
    cli_args = None


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate GAPOnet compensation performance.")

    # Task and Environment args
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Humanoid-AGIBOT-Delta-Action",
        help="Gym task id to run.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of envs to create.")

    # Motion selection args
    parser.add_argument("--motion-index", type=int, default=0, help="Fixed motion index to use.")
    parser.add_argument("--time-index", type=int, default=0, help="Fixed start time index to use.")
    parser.add_argument("--motion-file", type=str, default=None, help="Override motion file path.")
    parser.add_argument("--num-steps", type=int, default=None, help="Number of steps to simulate.")

    # Simulation args
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )

    # Checkpoint args
    parser.add_argument(
        "--use_pretrained_checkpoint",
        action="store_true",
        help="Use the pre-trained checkpoint from Nucleus.",
    )

    # Plotting args
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="logs/train_result",
        help="Directory to save plots.",
    )

    # Add RSL-RL args
    if cli_args:
        cli_args.add_rsl_rl_args(parser)

    # Add AppLauncher args
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()

    # Launch Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import Isaac Sim dependent modules (MUST be after AppLauncher)
    import rsl_rl.runners.on_policy_runner as on_policy_runner
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    # -------------------------------------------------------------------------
    # Import tasks and env config (similar to check_motion_alignment logic)
    # -------------------------------------------------------------------------
    tasks_init = (
        Path(__file__).resolve().parents[2]
        / "source"
        / "sim2real"
        / "sim2real"
        / "tasks"
        / "humanoid_agibot"
        / "__init__.py"
    )
    if not tasks_init.is_file():
        raise RuntimeError(f"Task module not found: {tasks_init}")

    if "sim2real" not in sys.modules:
        sys.modules["sim2real"] = types.ModuleType("sim2real")
    if "sim2real.tasks" not in sys.modules:
        sys.modules["sim2real.tasks"] = types.ModuleType("sim2real.tasks")

    spec = importlib.util.spec_from_file_location("sim2real.tasks.humanoid_agibot", tasks_init)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load task module: {tasks_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sim2real.tasks.humanoid_agibot"] = module
    spec.loader.exec_module(module)

    # Register DeepONetActorCritic after task modules are ready
    from sim2real.rsl_rl.modules import DeepONetActorCritic

    on_policy_runner.DeepONetActorCritic = DeepONetActorCritic

    # Load Env Cfg
    env_cfg_path = (
        Path(__file__).resolve().parents[2]
        / "source"
        / "sim2real"
        / "sim2real"
        / "tasks"
        / "humanoid_agibot"
        / "humanoid_agibot_env_cfg.py"
    )
    if not env_cfg_path.is_file():
        raise RuntimeError(f"Env cfg module not found: {env_cfg_path}")

    env_cfg_mod_name = "sim2real.tasks.humanoid_agibot.humanoid_agibot_env_cfg"
    env_cfg_spec = importlib.util.spec_from_file_location(env_cfg_mod_name, env_cfg_path)
    if env_cfg_spec is None or env_cfg_spec.loader is None:
        raise RuntimeError(f"Failed to load env cfg module: {env_cfg_path}")
    env_cfg_module = importlib.util.module_from_spec(env_cfg_spec)
    sys.modules[env_cfg_mod_name] = env_cfg_module
    env_cfg_spec.loader.exec_module(env_cfg_module)

    # -------------------------------------------------------------------------
    # Setup Environment Configuration
    # -------------------------------------------------------------------------
    env_cfg = env_cfg_module.HumanoidOperatorEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device

    if args.motion_file:
        env_cfg.train_motion_file = args.motion_file
        env_cfg.test_motion_file = args.motion_file
        print(f"[INFO] Overriding motion file with: {args.motion_file}")

    # -------------------------------------------------------------------------
    # Phase 1: Run Baseline Simulation (No Compensation)
    # -------------------------------------------------------------------------
    env_base = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env_unwrapped = env_base.unwrapped
    device = env_unwrapped.device
    num_actions = env_unwrapped.cfg.action_space

    motion_len = int(env_unwrapped._motion_loader.motion_len[args.motion_index].item())
    num_steps = motion_len if args.num_steps is None else min(args.num_steps, motion_len)

    print(f"[INFO] Environment created. Device: {device}, Num Envs: {args.num_envs}")
    print(f"[INFO] Motion Index: {args.motion_index}, Length: {motion_len}, Steps: {num_steps}")

    motion_indices = torch.full((args.num_envs,), args.motion_index, dtype=torch.long, device=device)
    time_indices = torch.full((args.num_envs,), args.time_index, dtype=torch.long, device=device)

    _sync_to_motion(env_unwrapped, motion_indices, time_indices)

    zero_action = torch.zeros((args.num_envs, num_actions), device=device)
    sim_baseline_traj = []
    real_traj = []

    for _ in range(num_steps):
        time_indices_step = time_indices.clone()
        _, _, dones, _ = env_unwrapped.step_operator(
            zero_action, motion_coords=(motion_indices, time_indices_step)
        )

        sim_pos = env_unwrapped.robot.data.joint_pos[:, env_unwrapped.motion_joint_ids].detach().cpu().numpy()
        real_pos = env_unwrapped._motion_loader.dof_positions[
            motion_indices, time_indices_step
        ].detach().cpu().numpy()

        sim_baseline_traj.append(sim_pos[0])
        real_traj.append(real_pos[0])

        motion_indices = env_unwrapped.motion_indices.clone()
        time_indices = env_unwrapped.time_indices.clone()

        if bool(torch.any(dones)):
            break

    sim_baseline_arr = np.stack(sim_baseline_traj, axis=0)
    real_arr = np.stack(real_traj, axis=0)

    print(f"[INFO] Baseline simulation complete. Steps: {len(sim_baseline_traj)}")
    env_base.close()

    # -------------------------------------------------------------------------
    # Phase 2: Run Compensated Simulation (With GAPOnet)
    # -------------------------------------------------------------------------
    if cli_args:
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    else:
        agent_cfg = RslRlOnPolicyRunnerCfg(experiment_name="default", run_name="default")

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args.task)
    elif args.checkpoint:
        resume_path = retrieve_file_path(args.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    env_comp = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env_comp_unwrapped = env_comp.unwrapped
    env_wrapped = RslRlVecEnvWrapper(env_comp)

    ppo_runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=device)

    motion_indices = torch.full((args.num_envs,), args.motion_index, dtype=torch.long, device=device)
    time_indices = torch.full((args.num_envs,), args.time_index, dtype=torch.long, device=device)

    _sync_to_motion(env_comp_unwrapped, motion_indices, time_indices)
    env_comp_unwrapped.last_delta_action[:] = 0
    if hasattr(env_comp_unwrapped, "model_history"):
        env_comp_unwrapped.model_history[:] = 0

    sim_comp_traj = []

    for _ in range(num_steps):
        time_indices_step = time_indices.clone()
        with torch.inference_mode():
            _update_sensor_data_from_sim(env_comp_unwrapped)
            obs_dict = env_comp_unwrapped.compute_operator_observation()
            obs = torch.cat([obs_dict["branch"], obs_dict["trunk"]], dim=1)
            actions = policy(obs)

        _, _, dones, _ = env_comp_unwrapped.step_operator(
            actions, motion_coords=(motion_indices, time_indices_step)
        )

        motion_indices = env_comp_unwrapped.motion_indices.clone()
        time_indices = env_comp_unwrapped.time_indices.clone()

        sim_pos = env_comp_unwrapped.robot.data.joint_pos[:, env_comp_unwrapped.motion_joint_ids].detach().cpu().numpy()
        sim_comp_traj.append(sim_pos[0])

        if bool(torch.any(dones)):
            break

    sim_comp_arr = np.stack(sim_comp_traj, axis=0)
    print(f"[INFO] Compensated simulation complete. Steps: {len(sim_comp_traj)}")

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    min_len = min(sim_baseline_arr.shape[0], sim_comp_arr.shape[0], real_arr.shape[0])
    sim_baseline_arr = sim_baseline_arr[:min_len]
    sim_comp_arr = sim_comp_arr[:min_len]
    real_arr = real_arr[:min_len]

    steps = np.arange(min_len)

    sim_base_deg = np.degrees(sim_baseline_arr)
    sim_comp_deg = np.degrees(sim_comp_arr)
    real_deg = np.degrees(real_arr)

    dof_names = list(env_comp_unwrapped._motion_loader.dof_names)
    joint_count = min(18, len(dof_names))

    plot_dir = Path(args.plot_dir) / f"episode{args.motion_index:03d}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving plots to: {plot_dir}")

    for idx in range(joint_count):
        name = dof_names[idx]
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax0.plot(steps, real_deg[:, idx], "r-", label="real", linewidth=1.5, alpha=0.8)
        ax0.plot(steps, sim_base_deg[:, idx], "b--", label="sim", linewidth=1.5, alpha=0.8)
        ax0.plot(steps, sim_comp_deg[:, idx], "g-", label="sim+delta", linewidth=1.5, alpha=0.8)
        ax0.set_title(f"Joint: {name}")
        ax0.set_ylabel("position (deg)")
        ax0.legend()
        ax0.grid(True, alpha=0.3)

        err_base = np.abs(sim_base_deg[:, idx] - real_deg[:, idx])
        err_comp = np.abs(sim_comp_deg[:, idx] - real_deg[:, idx])
        ax1.plot(steps, err_base, "b--", label=f"|sim-real| mean: {np.mean(err_base):.3f}")
        ax1.plot(steps, err_comp, "g-", label=f"|sim+delta-real| mean: {np.mean(err_comp):.3f}")
        ax1.set_xlabel("step")
        ax1.set_ylabel("|error| (deg)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_dir / f"{name}_traj_comp.png")
        plt.close(fig)

    print("[INFO] Done.")

    env_comp.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
