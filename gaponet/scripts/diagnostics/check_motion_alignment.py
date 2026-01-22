"""Quick alignment check between motion data and sim joints.

Runs a short rollout using zero delta-action and reports joint position errors.
This script is agibot-specific and does not affect other robots.
"""

from __future__ import annotations

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import torch
import importlib.util
from pathlib import Path
import types

from isaaclab.app import AppLauncher


def main() -> int:
    parser = argparse.ArgumentParser(description="Check motion alignment for humanoid_agibot")
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Humanoid-AGIBOT-Delta-Action",
        help="Gym task id to run.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of envs to create.")
    parser.add_argument("--num-steps", type=int, default=200, help="Number of steps to run.")
    parser.add_argument("--motion-index", type=int, default=0, help="Fixed motion index to use.")
    parser.add_argument("--time-index", type=int, default=0, help="Fixed start time index to use.")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup steps to skip from stats.")
    parser.add_argument(
        "--sync-to-motion",
        action="store_true",
        help="Before rollout, set robot joint states to motion real positions at the start frame.",
    )
    # AppLauncher args (e.g., --headless, --device)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch Isaac Sim (required for carb/omni dependencies).
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Ensure task registration without importing sim2real.__init__ (avoids omni.ext).
    tasks_init = Path(__file__).resolve().parents[2] / "source" / "sim2real" / "sim2real" / "tasks" / "humanoid_agibot" / "__init__.py"
    if not tasks_init.is_file():
        raise RuntimeError(f"Task module not found: {tasks_init}")
    # Create lightweight parent packages to bypass sim2real/__init__.py.
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

    # Build env config overrides.
    # Load env config class without importing sim2real.__init__.
    env_cfg_path = Path(__file__).resolve().parents[2] / "source" / "sim2real" / "sim2real" / "tasks" / "humanoid_agibot" / "humanoid_agibot_env_cfg.py"
    if not env_cfg_path.is_file():
        raise RuntimeError(f"Env cfg module not found: {env_cfg_path}")
    env_cfg_mod_name = "sim2real.tasks.humanoid_agibot.humanoid_agibot_env_cfg"
    env_cfg_spec = importlib.util.spec_from_file_location(env_cfg_mod_name, env_cfg_path)
    if env_cfg_spec is None or env_cfg_spec.loader is None:
        raise RuntimeError(f"Failed to load env cfg module: {env_cfg_path}")
    env_cfg_module = importlib.util.module_from_spec(env_cfg_spec)
    sys.modules[env_cfg_mod_name] = env_cfg_module
    env_cfg_spec.loader.exec_module(env_cfg_module)
    env_cfg = env_cfg_module.HumanoidOperatorEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)

    env_unwrapped = env.unwrapped
    num_actions = env_unwrapped.cfg.action_space
    device = env_unwrapped.device

    # Initialize motion indices.
    motion_indices = torch.full((args.num_envs,), args.motion_index, dtype=torch.long, device=device)
    time_indices = torch.full((args.num_envs,), args.time_index, dtype=torch.long, device=device)

    # Optionally sync robot joints to the motion pose at the start frame.
    if args.sync_to_motion:
        motion_pos = env_unwrapped._motion_loader.dof_positions[motion_indices, time_indices]
        motion_vel = torch.zeros_like(motion_pos)
        env_unwrapped.robot.write_joint_state_to_sim(
            motion_pos, motion_vel, joint_ids=env_unwrapped.motion_joint_ids
        )
        env_unwrapped.robot.set_joint_position_target(
            motion_pos, joint_ids=env_unwrapped.motion_joint_ids
        )
        env_unwrapped._raw_step_simulator()

    # Zero delta-action to directly follow motion targets.
    zero_action = torch.zeros((args.num_envs, num_actions), device=device)

    joint_err_means = []
    joint_err_maxs = []

    per_joint_max = []
    for step in range(args.num_steps):
        _, _, dones, info = env_unwrapped.step_operator(
            zero_action, motion_coords=(motion_indices, time_indices)
        )
        # info["episode"]["joint_pos_diff"] shape: (num_envs, num_dofs)
        joint_pos_diff = info["episode"]["joint_pos_diff"].detach().cpu().numpy()
        if step >= args.warmup_steps:
            joint_err_means.append(float(np.mean(joint_pos_diff)))
            joint_err_maxs.append(float(np.max(joint_pos_diff)))
            per_joint_max.append(np.max(joint_pos_diff, axis=0))

        # advance time indices for the next step (mirror env internal logic)
        time_indices = time_indices + 1
        if bool(torch.any(dones)):
            break

    print("Alignment check results:")
    print(f"  steps: {len(joint_err_means)} (warmup skipped: {args.warmup_steps})")
    if joint_err_means:
        print(f"  mean joint error (deg): {np.mean(joint_err_means):.4f}")
        print(f"  max joint error (deg): {np.max(joint_err_maxs):.4f}")
        per_joint_max = np.max(np.stack(per_joint_max, axis=0), axis=0)
        dof_names = env_unwrapped._motion_loader.dof_names
        top_idx = np.argsort(-per_joint_max)[:5]
        print("  top max-error joints (deg):")
        for idx in top_idx:
            print(f"    {dof_names[idx]}: {per_joint_max[idx]:.4f}")
    else:
        print("  no stats collected (all steps were warmup)")

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
