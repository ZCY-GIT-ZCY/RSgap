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
    env = gym.make(
        args.task,
        cfg={
            "scene": {"num_envs": args.num_envs},
            "sim": {"device": args.device},
        },
        render_mode=None,
    )

    env_unwrapped = env.unwrapped
    num_actions = env_unwrapped.cfg.action_space
    device = env_unwrapped.device

    # Initialize motion indices.
    motion_indices = torch.full((args.num_envs,), args.motion_index, dtype=torch.long, device=device)
    time_indices = torch.full((args.num_envs,), args.time_index, dtype=torch.long, device=device)

    # Zero delta-action to directly follow motion targets.
    zero_action = torch.zeros((args.num_envs, num_actions), device=device)

    joint_err_means = []
    joint_err_maxs = []

    for _ in range(args.num_steps):
        _, _, dones, info = env_unwrapped.step_operator(
            zero_action, motion_coords=(motion_indices, time_indices)
        )
        # info["episode"]["joint_pos_diff"] shape: (num_envs, num_dofs)
        joint_pos_diff = info["episode"]["joint_pos_diff"].detach().cpu().numpy()
        joint_err_means.append(float(np.mean(joint_pos_diff)))
        joint_err_maxs.append(float(np.max(joint_pos_diff)))

        # advance time indices for the next step (mirror env internal logic)
        time_indices = time_indices + 1
        if bool(torch.any(dones)):
            break

    print("Alignment check results:")
    print(f"  steps: {len(joint_err_means)}")
    print(f"  mean joint error (deg): {np.mean(joint_err_means):.4f}")
    print(f"  max joint error (deg): {np.max(joint_err_maxs):.4f}")

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
