"""
Evaluate GAPOnet compensation performance by comparing Sim, Sim+Delta, and Real trajectories.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import importlib.util
import types

import gymnasium as gym
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

# Add rsl_rl script directory to path to import cli_args
rsl_rl_script_dir = Path(__file__).resolve().parents[2] / "scripts" / "reinforcement_learning" / "rsl_rl"
sys.path.append(str(rsl_rl_script_dir))

# Try importing cli_args, if it fails, we might need to adjust path or mock it
try:
    import cli_args
except ImportError:
    print(f"Warning: Could not import cli_args from {rsl_rl_script_dir}. Some arguments might be missing.")
    cli_args = None

    # Add AppLauncher args
    AppLauncher.add_app_launcher_args(parser)
    
    args = parser.parse_args()

    # Launch Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # -------------------------------------------------------------------------
    # Import Isaac Sim dependent modules (MUST be after AppLauncher)
    # -------------------------------------------------------------------------
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    # import isaaclab_tasks
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    # -------------------------------------------------------------------------
    # Import tasks and env config (similar to check_motion_alignment logic to be safe)
    # -------------------------------------------------------------------------
    # Ensure task registration without importing sim2real.__init__ (avoids omni.ext).
    tasks_init = Path(__file__).resolve().parents[2] / "source" / "sim2real" / "sim2real" / "tasks" / "humanoid_agibot" / "__init__.py"
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

    # Load Env Cfg
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
    
    # -------------------------------------------------------------------------
    # Setup Environment Configuration
    # -------------------------------------------------------------------------
    env_cfg = env_cfg_module.HumanoidOperatorEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    
    # Override motion file if provided
    if args.motion_file:
        env_cfg.train_motion_file = args.motion_file
        env_cfg.test_motion_file = args.motion_file
        print(f"[INFO] Overriding motion file with: {args.motion_file}")

    # Create Environment
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env_unwrapped = env.unwrapped
    device = env_unwrapped.device
    num_actions = env_unwrapped.cfg.action_space
    
    print(f"[INFO] Environment created. Device: {device}, Num Envs: {args.num_envs}")
    
    # Get Motion Length
    motion_len = int(env_unwrapped._motion_loader.motion_len[args.motion_index].item())
    print(f"[INFO] Motion Index: {args.motion_index}, Length: {motion_len}")

    # =========================================================================
    # Phase 1: Run Baseline Simulation (No Compensation)
    # =========================================================================
    print("\n[INFO] Starting Phase 1: Baseline Simulation (No Compensation)...")
    
    # Reset to specific motion
    motion_indices = torch.full((args.num_envs,), args.motion_index, dtype=torch.long, device=device)
    time_indices = torch.full((args.num_envs,), args.time_index, dtype=torch.long, device=device)
    
    # Sync robot to start
    motion_pos = env_unwrapped._motion_loader.dof_positions[motion_indices, time_indices]
    motion_vel = torch.zeros_like(motion_pos)
    env_unwrapped.robot.write_joint_state_to_sim(
        motion_pos, motion_vel, joint_ids=env_unwrapped.motion_joint_ids
    )
    env_unwrapped.robot.set_joint_position_target(
        motion_pos, joint_ids=env_unwrapped.motion_joint_ids
    )
    env_unwrapped._raw_step_simulator()
    
    zero_action = torch.zeros((args.num_envs, num_actions), device=device)
    
    sim_baseline_traj = []
    real_traj = []
    
    for step in range(motion_len):
        time_indices_step = time_indices.clone()
        
        # Step with zero action (no compensation)
        _, _, dones, _ = env_unwrapped.step_operator(
            zero_action, motion_coords=(motion_indices, time_indices_step)
        )
        
        # Collect data (use pre-step state logic from check_motion_alignment)
        # We need to capture the state that corresponds to the target we just tried to reach
        # But check_motion_alignment captures current state vs target at time_indices_step
        
        sim_pos = env_unwrapped.robot.data.joint_pos[:, env_unwrapped.motion_joint_ids].detach().cpu().numpy()
        real_pos = env_unwrapped._motion_loader.dof_positions[
            motion_indices, time_indices_step
        ].detach().cpu().numpy()
        
        sim_baseline_traj.append(sim_pos[0])
        real_traj.append(real_pos[0])
        
        # Update indices for next step logic inside step_operator implies it increments internally?
        # check_motion_alignment re-reads motion_indices/time_indices from env after step_operator
        motion_indices = env_unwrapped.motion_indices.clone()
        time_indices = env_unwrapped.time_indices.clone()

        if bool(torch.any(dones)):
            break
            
    sim_baseline_arr = np.stack(sim_baseline_traj, axis=0)
    real_arr = np.stack(real_traj, axis=0)
    
    print(f"[INFO] Baseline simulation complete. Steps: {len(sim_baseline_traj)}")

    # =========================================================================
    # Phase 2: Run Compensated Simulation (With GAPOnet)
    # =========================================================================
    print("\n[INFO] Starting Phase 2: Compensated Simulation (With GAPOnet)...")
    
    # Load Agent Config
    if cli_args:
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    else:
        # Fallback default if cli_args fails
        agent_cfg = RslRlOnPolicyRunnerCfg(experiment_name="default", run_name="default")

    # Determine Checkpoint Path
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    if args.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args.task)
    elif args.checkpoint:
        resume_path = retrieve_file_path(args.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # Wrap env for RSL-RL
    # Note: We need to use a wrapper that matches what training used, usually RslRlVecEnvWrapper
    # However, RslRlVecEnvWrapper might interfere with our manual control of motion indices if not careful.
    # But OnPolicyRunner expects a wrapped env to get observations in correct dict format.
    # Let's wrap it, but we need to be careful about reset().
    
    # Re-create env wrapped
    # Actually, we can reuse the underlying env, just wrap it for the runner
    # But RSL-RL runner .load() needs an env to initialize buffers? No, it just loads weights.
    # But OnPolicyRunner init does use env properties.
    
    env_wrapped = RslRlVecEnvWrapper(env)
    
    ppo_runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=device)
    
    # Reset again for Phase 2
    motion_indices = torch.full((args.num_envs,), args.motion_index, dtype=torch.long, device=device)
    time_indices = torch.full((args.num_envs,), args.time_index, dtype=torch.long, device=device)
    
    # Manually reset robot state
    motion_pos = env_unwrapped._motion_loader.dof_positions[motion_indices, time_indices]
    motion_vel = torch.zeros_like(motion_pos)
    env_unwrapped.robot.write_joint_state_to_sim(
        motion_pos, motion_vel, joint_ids=env_unwrapped.motion_joint_ids
    )
    env_unwrapped.robot.set_joint_position_target(
        motion_pos, joint_ids=env_unwrapped.motion_joint_ids
    )
    
    # Reset internal buffers of the env (like last_action, etc)
    # env_unwrapped.reset() # We avoid full reset to keep our motion indices control
    env_unwrapped.last_delta_action[:] = 0
    if hasattr(env_unwrapped, "model_history"):
        env_unwrapped.model_history[:] = 0
    
    # Also need to reset wrapped env buffers if any
    obs, _ = env_wrapped.get_observations() # Initial observation
    
    # Hack: Force overwrite motion/time indices again because get_observations might depend on them 
    # (actually get_observations uses current robot state, which we just set)
    # But we need to make sure the env knows we are at start of motion.
    env_unwrapped.motion_indices[:] = motion_indices
    env_unwrapped.time_indices[:] = time_indices
    
    sim_comp_traj = []
    
    # Re-run simulation
    for step in range(motion_len):
        # time_indices is already updated in env from previous loop? 
        # No, we reset it.
        
        # Inference
        with torch.inference_mode():
            actions = policy(obs)
            
        # Step
        obs, _, _, _ = env_wrapped.step(actions)
        
        # Collect data
        sim_pos = env_unwrapped.robot.data.joint_pos[:, env_unwrapped.motion_joint_ids].detach().cpu().numpy()
        sim_comp_traj.append(sim_pos[0])
        
        # Check done (managed by wrapped env usually, but we check unwrapped for motion end)
        # The wrapped env step calls unwrapped.step which updates time_indices
        if env_unwrapped.time_indices[0] >= env_unwrapped._motion_loader.motion_len[args.motion_index] - 1:
            break
            
    sim_comp_arr = np.stack(sim_comp_traj, axis=0)
    print(f"[INFO] Compensated simulation complete. Steps: {len(sim_comp_traj)}")
    
    # =========================================================================
    # Visualization
    # =========================================================================
    # Trim to matching lengths (in case of slight off-by-one due to done logic)
    min_len = min(sim_baseline_arr.shape[0], sim_comp_arr.shape[0], real_arr.shape[0])
    sim_baseline_arr = sim_baseline_arr[:min_len]
    sim_comp_arr = sim_comp_arr[:min_len]
    real_arr = real_arr[:min_len]
    
    steps = np.arange(min_len)
    
    # Convert to degrees
    sim_base_deg = np.degrees(sim_baseline_arr)
    sim_comp_deg = np.degrees(sim_comp_arr)
    real_deg = np.degrees(real_arr)
    
    dof_names = env_unwrapped._motion_loader.dof_names
    
    # Output Directory
    plot_dir = Path(args.plot_dir) / f"episode_{args.motion_index:06d}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Saving plots to: {plot_dir}")
    
    for idx, name in enumerate(dof_names):
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot 1: Trajectories
        ax0.plot(steps, real_deg[:, idx], 'r-', label="Real", linewidth=1.5, alpha=0.8)
        ax0.plot(steps, sim_base_deg[:, idx], 'b--', label="Sim (No Comp)", linewidth=1.5, alpha=0.8)
        ax0.plot(steps, sim_comp_deg[:, idx], 'g-', label="Sim (Compensated)", linewidth=1.5, alpha=0.8)
        
        ax0.set_title(f"Joint: {name}")
        ax0.set_ylabel("Position (deg)")
        ax0.legend()
        ax0.grid(True, alpha=0.3)
        
        # Plot 2: Errors
        err_base = np.abs(sim_base_deg[:, idx] - real_deg[:, idx])
        err_comp = np.abs(sim_comp_deg[:, idx] - real_deg[:, idx])
        
        ax1.plot(steps, err_base, 'b--', label=f"Error (No Comp) | Mean: {np.mean(err_base):.3f}")
        ax1.plot(steps, err_comp, 'g-', label=f"Error (Compensated) | Mean: {np.mean(err_comp):.3f}")
        
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Abs Error (deg)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / f"{name}_traj_comp.png")
        plt.close(fig)
        
    print("[INFO] Done.")
    
    # Cleanup
    env.close()
    simulation_app.close()
    return 0

if __name__ == "__main__":
    main()
