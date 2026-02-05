"""Pretrain action branch with real-sim delta targets."""

from __future__ import annotations

import argparse
import faulthandler
import sys
from pathlib import Path
import importlib.util
import types

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher


def _compute_sim_sensor(env_unwrapped, prev_joint_pos: torch.Tensor, prev_joint_vel: torch.Tensor) -> torch.Tensor:
    joint_pos = env_unwrapped.robot.data.joint_pos[:, env_unwrapped.motion_joint_ids]
    joint_vel = env_unwrapped.robot.data.joint_vel[:, env_unwrapped.motion_joint_ids]
    sensor = torch.cat([joint_pos, joint_vel * env_unwrapped.step_dt], dim=1)
    if env_unwrapped.cfg.delta_sensor_value:
        prev = torch.cat([prev_joint_pos, prev_joint_vel * env_unwrapped.step_dt], dim=1)
        sensor = sensor - prev
    sensor = sensor.view(env_unwrapped.num_envs, env_unwrapped.num_sensor_positions, env_unwrapped.cfg.sensor_dim)
    return sensor


def _set_trainable_params(policy, train_log_std: bool) -> None:
    for param in policy.parameters():
        param.requires_grad = False
    for param in policy.branch_net.parameters():
        param.requires_grad = True
    for param in policy.trunk_net.parameters():
        param.requires_grad = True
    if train_log_std:
        policy.log_std.requires_grad = True


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrain PPO action branch with real-sim delta targets.")

    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Humanoid-AGIBOT-Delta-Action",
        help="Gym task id to run.",
    )
    parser.add_argument("--num-envs", type=int, default=128, help="Number of envs to create.")
    parser.add_argument("--num-steps", type=int, default=32, help="Steps per iteration for data collection.")
    parser.add_argument("--num-iters", type=int, default=200, help="Number of pretrain iterations.")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of epochs per iteration.")
    parser.add_argument("--num-mini-batches", type=int, default=4, help="Number of mini-batches per epoch.")
    parser.add_argument("--lr", type=float, default=1.0e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--train-log-std", action="store_true", help="Also train action log_std.")

    parser.add_argument("--motion-file", type=str, default=None, help="Override motion file path.")
    parser.add_argument("--use-model-sensor", action="store_true", help="Use model-based sensor for branch inputs.")
    parser.add_argument("--disable-model-sensor", action="store_true", help="Force using sim sensor inputs.")
    parser.add_argument("--use-obs-norm", action="store_true", help="Enable empirical obs normalization.")
    parser.add_argument("--no-obs-norm", action="store_true", help="Disable empirical obs normalization.")

    parser.add_argument("--save-dir", type=str, default="logs/pretrain", help="Directory to save checkpoints.")
    parser.add_argument("--save-interval", type=int, default=50, help="Iterations between checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Simulation args
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument(
        "--hang-timeout",
        type=int,
        default=600,
        help="Seconds before dumping traceback for long-running steps; 0 disables.",
    )

    # Add RSL-RL args
    rsl_rl_script_dir = Path(__file__).resolve().parents[2] / "scripts" / "reinforcement_learning" / "rsl_rl"
    sys.path.append(str(rsl_rl_script_dir))
    try:
        import cli_args
    except ImportError:
        print(f"Warning: Could not import cli_args from {rsl_rl_script_dir}. Some arguments might be missing.")
        cli_args = None

    if cli_args:
        cli_args.add_rsl_rl_args(parser)

    # Add AppLauncher args
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()

    if args.hang_timeout <= 0:
        faulthandler.cancel_dump_traceback_later()
    else:
        faulthandler.dump_traceback_later(args.hang_timeout, repeat=True)

    torch.manual_seed(args.seed)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import Isaac Sim dependent modules (MUST be after AppLauncher)
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
    from sim2real.rsl_rl.modules import DeepONetActorCritic
    from rsl_rl.modules import EmpiricalNormalization

    # -------------------------------------------------------------------------
    # Import tasks and env config
    # -------------------------------------------------------------------------
    source_root = Path(__file__).resolve().parents[2] / "source"
    sys.path.append(str(source_root))

    tasks_init = (
        source_root
        / "sim2real"
        / "sim2real"
        / "tasks"
        / "humanoid_agibot"
        / "__init__.py"
    )
    if not tasks_init.is_file():
        raise RuntimeError(f"Task module not found: {tasks_init}")

    if "sim2real" not in sys.modules:
        sim2real_pkg = types.ModuleType("sim2real")
        sim2real_pkg.__path__ = [str(source_root / "sim2real")]
        sys.modules["sim2real"] = sim2real_pkg
    elif not hasattr(sys.modules["sim2real"], "__path__"):
        sys.modules["sim2real"].__path__ = [str(source_root / "sim2real")]
    tasks_pkg_path = source_root / "sim2real" / "sim2real" / "tasks"
    if "sim2real.tasks" not in sys.modules:
        tasks_pkg = types.ModuleType("sim2real.tasks")
        tasks_pkg.__path__ = [str(tasks_pkg_path)]
        sys.modules["sim2real.tasks"] = tasks_pkg
    elif not hasattr(sys.modules["sim2real.tasks"], "__path__"):
        sys.modules["sim2real.tasks"].__path__ = [str(tasks_pkg_path)]

    spec = importlib.util.spec_from_file_location("sim2real.tasks.humanoid_agibot", tasks_init)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load task module: {tasks_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sim2real.tasks.humanoid_agibot"] = module
    spec.loader.exec_module(module)

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
    # Create environment
    # -------------------------------------------------------------------------
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env_unwrapped = env.unwrapped
    device = env_unwrapped.device
    num_actions = env_unwrapped.cfg.action_space

    env_unwrapped.sample_all_environments()

    # -------------------------------------------------------------------------
    # Build policy (DeepONetActorCritic)
    # -------------------------------------------------------------------------
    if cli_args:
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    else:
        agent_cfg = RslRlOnPolicyRunnerCfg(experiment_name="pretrain", run_name="pretrain")

    policy_cfg = agent_cfg.policy.to_dict() if hasattr(agent_cfg.policy, "to_dict") else dict(agent_cfg.policy)
    policy_cfg.pop("class_name", None)
    policy_cfg["model_input_dim"] = env_unwrapped.compute_model_observation(add_noise=False, update_history=False).shape[1]

    policy_class = DeepONetActorCritic
    policy = policy_class(0, 0, num_actions, **policy_cfg).to(device)
    _set_trainable_params(policy, args.train_log_std)

    use_model_sensor = False
    if args.use_model_sensor:
        use_model_sensor = True
    elif not args.disable_model_sensor:
        if bool(getattr(agent_cfg, "model_based_sensor", False)) and policy_cfg.get("model_pretrained_path"):
            use_model_sensor = True
        elif bool(getattr(agent_cfg, "model_based_sensor", False)):
            print("[WARN] model_based_sensor is True but model_pretrained_path is empty. Using sim sensor.")

    obs_normalizer = None
    privileged_obs_normalizer = None
    enable_norm_stats = bool(getattr(agent_cfg, "empirical_normalization", False)) and not args.no_obs_norm
    if enable_norm_stats:
        num_obs = policy.total_branch_dim + policy.total_trunk_dim
        obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(device)
        critic_dim = env_unwrapped.compute_operator_observation()["critic"].shape[1]
        privileged_obs_normalizer = EmpiricalNormalization(shape=[critic_dim], until=1.0e8).to(device)

    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    zero_action = torch.zeros((env_unwrapped.num_envs, num_actions), device=device)
    prev_joint_pos = env_unwrapped.robot.data.joint_pos[:, env_unwrapped.motion_joint_ids].clone()
    prev_joint_vel = env_unwrapped.robot.data.joint_vel[:, env_unwrapped.motion_joint_ids].clone()

    print(f"[INFO] Starting pretrain. device={device}, num_envs={args.num_envs}, use_model_sensor={use_model_sensor}")

    for iteration in range(1, args.num_iters + 1):
        obs_list = []
        target_list = []

        for _ in range(args.num_steps):
            if use_model_sensor:
                with torch.no_grad():
                    model_obs = env_unwrapped.compute_model_observation(add_noise=False, update_history=False).to(device)
                    sensor_data = policy.model_sensor(model_obs).reshape(
                        env_unwrapped.num_envs, env_unwrapped.num_sensor_positions, -1
                    )
                env_unwrapped.set_sensor_data(sensor_data)
            else:
                sensor = _compute_sim_sensor(env_unwrapped, prev_joint_pos, prev_joint_vel)
                env_unwrapped.set_sensor_data(sensor)

            with torch.no_grad():
                obs_dict = env_unwrapped.compute_operator_observation()
                obs = torch.cat([obs_dict["branch"], obs_dict["trunk"]], dim=1)
                if obs_normalizer is not None:
                    _ = obs_normalizer(obs)
                if privileged_obs_normalizer is not None:
                    _ = privileged_obs_normalizer(obs_dict["critic"])
                if obs_normalizer is not None and args.use_obs_norm:
                    obs = obs_normalizer(obs)

                real_pos = env_unwrapped._motion_loader.dof_positions[
                    env_unwrapped.motion_indices, env_unwrapped.time_indices
                ]
                sim_pos = env_unwrapped.robot.data.joint_pos[:, env_unwrapped.motion_joint_ids]
                delta_full = real_pos - sim_pos
                delta_target = delta_full[:, env_unwrapped._motion_loader.joint_sequence_index]

            obs_list.append(obs)
            target_list.append(delta_target)

            time_indices_step = env_unwrapped.time_indices.clone()
            env_unwrapped.step_operator(
                zero_action, motion_coords=(env_unwrapped.motion_indices, time_indices_step)
            )

            if use_model_sensor:
                with torch.no_grad():
                    env_unwrapped.compute_model_observation(add_noise=False, update_history=True)
            else:
                prev_joint_pos = env_unwrapped.robot.data.joint_pos[:, env_unwrapped.motion_joint_ids].clone()
                prev_joint_vel = env_unwrapped.robot.data.joint_vel[:, env_unwrapped.motion_joint_ids].clone()

        obs_batch = torch.cat(obs_list, dim=0)
        target_batch = torch.cat(target_list, dim=0)

        batch_size = obs_batch.shape[0]
        if batch_size % args.num_mini_batches != 0:
            raise RuntimeError(
                f"Batch size {batch_size} is not divisible by num_mini_batches {args.num_mini_batches}."
            )
        mini_batch_size = batch_size // args.num_mini_batches

        policy.train()
        epoch_losses = []
        for _ in range(args.num_epochs):
            perm = torch.randperm(batch_size, device=device)
            for i in range(args.num_mini_batches):
                idx = perm[i * mini_batch_size : (i + 1) * mini_batch_size]
                obs_mb = obs_batch[idx]
                target_mb = target_batch[idx]

                pred = policy.act_inference(obs_mb)
                loss = torch.mean((pred - target_mb) ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

        mean_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        if iteration % 10 == 0 or iteration == 1:
            print(f"[INFO] Iter {iteration}/{args.num_iters} - loss: {mean_loss:.6f}")

        if iteration % args.save_interval == 0 or iteration == args.num_iters:
            ckpt_path = save_dir / f"pretrain_iter_{iteration:06d}.pt"
            torch.save(
                {
                    "model_state_dict": policy.state_dict(),
                    "obs_norm_state_dict": obs_normalizer.state_dict() if obs_normalizer is not None else None,
                    "privileged_obs_norm_state_dict": privileged_obs_normalizer.state_dict()
                    if privileged_obs_normalizer is not None
                    else None,
                    "iteration": iteration,
                    "loss": mean_loss,
                    "config": {
                        "task": args.task,
                        "num_envs": args.num_envs,
                        "num_steps": args.num_steps,
                        "num_epochs": args.num_epochs,
                        "num_mini_batches": args.num_mini_batches,
                        "lr": args.lr,
                        "use_model_sensor": use_model_sensor,
                    },
                },
                ckpt_path,
            )
            print(f"[INFO] Saved checkpoint: {ckpt_path}")

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
