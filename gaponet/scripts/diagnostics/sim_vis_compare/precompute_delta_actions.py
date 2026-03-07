"""
precompute GAPONet Delta Actions


Usage:
    python precompute_delta_actions.py \
        --task Isaac-SO101-Operator-Delta-Action \
        --model /path/to/model.pt \
        --motion_file /path/to/test.npz \
        --motion_idx 0 \
        --output_file /path/to/delta_actions.npz
"""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

# Add rsl_rl scripts path for cli_args
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # noqa: E402

# Add argparse arguments
parser = argparse.ArgumentParser(description="Precompute GAPONet delta actions for visualization.")
parser.add_argument("--task", type=str, default="Isaac-SO101-Operator-Delta-Action", help="Name of the task.")
parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint file.")
parser.add_argument("--motion_file", type=str, required=True, help="Path to motion/test data file (.npz).")
parser.add_argument("--motion_idx", type=int, default=0, help="Motion index to process.")
parser.add_argument("--output_file", type=str, default=None, help="Output file path for delta actions.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (usually 1).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

# Append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Run headless for precomputation
args_cli.headless = True

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim2real.rsl_rl.runners import *  # noqa: E402, F401, F403
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# Import task modules explicitly to trigger gym.register()
import sim2real.tasks.humanoid_operator  # noqa: F401, E402
import sim2real.tasks.humanoid_agibot  # noqa: F401, E402


def log_message(message):
    """Format and print log messages with timestamp."""
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[PrecomputeDelta][{current_time}] {message}")


def main():
    """Main function to precompute delta actions."""
    log_message("Starting delta action precomputation...")
    
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )
    
    # Set to play mode
    try:
        env_cfg.mode = "play"
    except Exception:
        pass

    # Extend episode length to avoid timeout before motion ends
    # Default 10s = 500 steps, which truncates longer motions
    env_cfg.episode_length_s = 10000.0

    # Override motion file
    motion_file_path = os.path.abspath(args_cli.motion_file)
    if not os.path.exists(motion_file_path):
        raise FileNotFoundError(f"Motion file not found: {motion_file_path}")
    env_cfg.test_motion_file = motion_file_path
    log_message(f"Using motion file: {motion_file_path}")

    # Parse agent configuration
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Verify model path
    resume_path = os.path.abspath(args_cli.model)
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Model file not found: {resume_path}")
    log_dir = os.path.dirname(resume_path)
    log_message(f"Loading model from: {resume_path}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Create runner and load checkpoint
    runner_class = eval(agent_cfg.class_name)
    ppo_runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # Get the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # NOTE:
    # Keep using runner-provided inference policy.
    # Some architectures define full_forward() that expects a dict-like input
    # (e.g., {"model": ..., "operator": ...}), while env wrappers here provide
    # flattened tensor observations.

    # Get environment info
    unwrapped_env = env.unwrapped
    device = unwrapped_env.device
    motion_loader = unwrapped_env._motion_loader
    motion_idx = args_cli.motion_idx
    
    num_dofs = unwrapped_env.num_dofs if hasattr(unwrapped_env, 'num_dofs') else 6
    
    # Get motion length
    motion_len = motion_loader.motion_len
    if hasattr(motion_len, '__getitem__'):
        total_frames = int(motion_len[motion_idx].item()) if hasattr(motion_len[motion_idx], 'item') else int(motion_len[motion_idx])
    else:
        total_frames = int(motion_len)
    
    log_message(f"Motion {motion_idx}: {total_frames} frames, {num_dofs} DOFs")
    
    # Storage for results
    all_commands = []
    all_delta_actions = []
    all_real_positions = []
    
    # ========== Single Run: Collect delta actions ==========
    # Use step_operator (same as evaluate_compensation.py) to ensure frame count
    # is identical: N frames (0 .. N-1).  env.step() goes through the full
    # gymnasium pipeline (_get_dones increments time_indices unconditionally and
    # may trigger early episode resets), which produces N-1 or fewer frames.
    log_message("Collecting delta actions from GAPONet model...")
    
    with torch.inference_mode():
        # Initialize environment
        env.reset()

        # Pin to the requested motion from the start
        motion_indices = torch.full(
            (unwrapped_env.num_envs,), motion_idx, dtype=torch.long, device=device
        )
        time_indices = torch.zeros(unwrapped_env.num_envs, dtype=torch.long, device=device)
        unwrapped_env.motion_indices[:] = motion_idx
        unwrapped_env.time_indices[:] = 0

        # Reset robot to the first frame of this motion
        joint_pos_init = motion_loader.dof_positions[motion_idx, 0]
        joint_vel_init = motion_loader.dof_velocities[motion_idx, 0]
        unwrapped_env.robot.write_joint_state_to_sim(
            joint_pos_init.unsqueeze(0),
            joint_vel_init.unsqueeze(0),
            joint_ids=unwrapped_env.motion_joint_ids,
        )

        # Initialise history / delta buffers (mirrors evaluate_compensation.py)
        unwrapped_env.last_delta_action[:] = 0
        if hasattr(unwrapped_env, 'model_history'):
            unwrapped_env.model_history[:] = 0
        prev_joint_pos = unwrapped_env.robot.data.joint_pos[
            :, unwrapped_env.motion_joint_ids
        ].clone()
        prev_joint_vel = unwrapped_env.robot.data.joint_vel[
            :, unwrapped_env.motion_joint_ids
        ].clone()

        for step in range(total_frames):
            time_indices_step = time_indices.clone()
            t = time_indices_step[0].item()

            # Ground-truth data from motion loader (independent of sim state)
            command  = motion_loader.dof_target_pos[motion_idx, t].cpu().numpy()
            real_pos = motion_loader.dof_positions[motion_idx, t].cpu().numpy()

            # ---- Build sensor (model_based_sensor=False) ----
            joint_pos_now = unwrapped_env.robot.data.joint_pos[
                :, unwrapped_env.motion_joint_ids
            ]
            joint_vel_now = unwrapped_env.robot.data.joint_vel[
                :, unwrapped_env.motion_joint_ids
            ]
            sensor = torch.cat(
                [joint_pos_now, joint_vel_now * unwrapped_env.step_dt], dim=1
            )
            if unwrapped_env.cfg.delta_sensor_value:
                prev = torch.cat(
                    [prev_joint_pos, prev_joint_vel * unwrapped_env.step_dt], dim=1
                )
                sensor = sensor - prev
            sensor = sensor.view(
                unwrapped_env.num_envs,
                unwrapped_env.num_sensor_positions,
                unwrapped_env.cfg.sensor_dim,
            )
            unwrapped_env.set_sensor_data(sensor)

            # ---- Compute observation and query policy ----
            obs_dict = unwrapped_env.compute_operator_observation()
            obs_tensor = torch.cat([obs_dict["branch"], obs_dict["trunk"]], dim=1)
            actions = policy(obs_tensor)
            delta_action = actions[0].cpu().numpy()

            # Store data
            all_commands.append(command.copy())
            all_delta_actions.append(delta_action.copy())
            all_real_positions.append(real_pos.copy())

            # ---- Advance simulation via step_operator (mirrors evaluate_compensation.py) ----
            _, _, dones, _ = unwrapped_env.step_operator(
                actions, motion_coords=(motion_indices, time_indices_step)
            )

            # Update prev state for delta-sensor computation on next step
            prev_joint_pos = unwrapped_env.robot.data.joint_pos[
                :, unwrapped_env.motion_joint_ids
            ].clone()
            prev_joint_vel = unwrapped_env.robot.data.joint_vel[
                :, unwrapped_env.motion_joint_ids
            ].clone()

            # Track time externally (step_operator updates unwrapped_env.time_indices)
            motion_indices = unwrapped_env.motion_indices.clone()
            time_indices   = unwrapped_env.time_indices.clone()

            if step % 100 == 0:
                log_message(f"  Frame {step}/{total_frames}")

            if bool(torch.any(dones)):
                log_message(f"  Motion done at frame {step}, stopping collection.")
                break
    
    log_message(f"Collected {len(all_delta_actions)} frames")
    
    # Convert to numpy arrays
    data = {
        'motion_idx': motion_idx,
        'num_frames': len(all_delta_actions),
        'num_dofs': num_dofs,
        'dof_names': np.array(motion_loader.dof_names, dtype=object),
        'commands': np.array(all_commands, dtype=np.float32),
        'delta_actions': np.array(all_delta_actions, dtype=np.float32),
        'real_positions': np.array(all_real_positions, dtype=np.float32),
        'model_path': resume_path,
        'motion_file': motion_file_path,
    }
    
    # Determine output path
    if args_cli.output_file:
        output_path = args_cli.output_file
    else:
        output_dir = os.path.dirname(motion_file_path)
        output_path = os.path.join(output_dir, f"precomputed_delta_motion{motion_idx}.npz")
    
    # Save
    np.savez(output_path, **data)
    log_message(f"Saved precomputed data to: {output_path}")
    
    # Print summary
    log_message("")
    log_message("=" * 60)
    log_message("SUMMARY")
    log_message("=" * 60)
    log_message(f"Total frames: {len(all_delta_actions)}")
    log_message(f"Delta action range: [{np.min(all_delta_actions):.4f}, {np.max(all_delta_actions):.4f}] rad")
    log_message(f"Mean |delta|: {np.abs(all_delta_actions).mean():.4f} rad ({np.rad2deg(np.abs(all_delta_actions).mean()):.2f}°)")
    
    # Per-joint delta statistics
    delta_arr = np.array(all_delta_actions)
    log_message("Per-joint mean |delta|:")
    for j in range(num_dofs):
        mean_delta = np.abs(delta_arr[:, j]).mean()
        log_message(f"  Joint {j}: {np.rad2deg(mean_delta):.2f}°")
    
    # Close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
