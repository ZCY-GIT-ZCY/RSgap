# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--checkpoint_path", type=str, default=None, help="Absolute or relative path to a checkpoint file."
)
parser.add_argument(
    "--warmup_critic",
    action="store_true",
    default=False,
    help="Warm up value branch only (freeze action branch) before training.",
)
parser.add_argument(
    "--warmup_threshold",
    type=float,
    default=0.01,
    help="Stop warmup when critic relative parameter change is below this threshold.",
)
parser.add_argument(
    "--warmup_lr",
    type=float,
    default=0.0,
    help="Override learning rate during warmup (0 means keep current).",
)
parser.add_argument(
    "--warmup_use_adaptive_lr",
    action="store_true",
    default=False,
    help="Keep adaptive learning rate schedule during warmup.",
)
parser.add_argument(
    "--warmup_min_iters",
    type=int,
    default=5,
    help="Minimum warmup iterations before checking threshold.",
)
parser.add_argument(
    "--warmup_max_iters",
    type=int,
    default=0,
    help="Maximum warmup iterations (0 means no limit).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

import sim2real.tasks.humanoid_operator
import sim2real.tasks.humanoid_agibot
from sim2real.rsl_rl.modules import DeepONetActorCritic
import rsl_rl.runners.on_policy_runner as on_policy_runner

# Make DeepONetActorCritic discoverable by rsl_rl eval()
on_policy_runner.DeepONetActorCritic = DeepONetActorCritic
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _load_checkpoint_compatible(runner: OnPolicyRunner, resume_path: str) -> None:
    checkpoint = None
    try:
        checkpoint = torch.load(resume_path, map_location="cpu")
    except Exception:
        checkpoint = None

    if isinstance(checkpoint, dict) and "iter" in checkpoint:
        runner.load(resume_path)
        return

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        policy = None
        if hasattr(runner.alg, "policy"):
            policy = runner.alg.policy
        elif hasattr(runner.alg, "actor_critic"):
            policy = runner.alg.actor_critic
        if policy is None:
            raise RuntimeError("Unsupported runner: no policy/actor_critic to load.")

        load_result = policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
        missing = getattr(load_result, "missing_keys", None)
        unexpected = getattr(load_result, "unexpected_keys", None)
        if missing is not None or unexpected is not None:
            if missing or unexpected:
                print(f"[WARN] Pretrain checkpoint load mismatch. missing={missing}, unexpected={unexpected}")

        obs_norm_state = checkpoint.get("obs_norm_state_dict")
        if obs_norm_state is not None and getattr(runner, "obs_normalizer", None) is not None:
            runner.obs_normalizer.load_state_dict(obs_norm_state, strict=False)

        runner.current_learning_iteration = int(checkpoint.get("iteration", 0))
        print("[INFO] Loaded pretrain checkpoint (weights only). Optimizer reset.")
        return

    runner.load(resume_path)


def _get_runner_policy(runner: OnPolicyRunner):
    if hasattr(runner.alg, "policy"):
        return runner.alg.policy
    if hasattr(runner.alg, "actor_critic"):
        return runner.alg.actor_critic
    return None


def _set_all_trainable(policy, trainable: bool) -> None:
    for param in policy.parameters():
        param.requires_grad = trainable


def _set_critic_trainable(policy, trainable: bool) -> None:
    if hasattr(policy, "critic"):
        for param in policy.critic.parameters():
            param.requires_grad = trainable


def _clone_params(params):
    return [p.detach().clone() for p in params]


def _param_delta(prev_params, params) -> tuple[float, float]:
    if not prev_params:
        return float("inf"), float("inf")
    total = 0.0
    norm = 0.0
    for prev, curr in zip(prev_params, params):
        diff = curr.detach() - prev
        total += float(torch.sum(diff * diff).item())
        norm += float(torch.sum(curr.detach() * curr.detach()).item())
    return total ** 0.5, norm ** 0.5


def _warmup_critic_only(
    runner: OnPolicyRunner,
    threshold: float,
    min_iters: int,
    max_iters: int,
    warmup_lr: float,
    use_adaptive_lr: bool,
) -> None:
    policy = _get_runner_policy(runner)
    if policy is None:
        print("[WARN] Warmup skipped: no policy/actor_critic on runner.")
        return

    orig_lrs = None
    if hasattr(runner.alg, "optimizer"):
        orig_lrs = [group.get("lr", None) for group in runner.alg.optimizer.param_groups]
        if warmup_lr and warmup_lr > 0.0:
            for group in runner.alg.optimizer.param_groups:
                group["lr"] = warmup_lr

    orig_schedule = None
    if not use_adaptive_lr and hasattr(runner.alg, "schedule"):
        orig_schedule = runner.alg.schedule
        runner.alg.schedule = "fixed"

    original_requires_grad = [(p, p.requires_grad) for p in policy.parameters()]
    _set_all_trainable(policy, False)
    _set_critic_trainable(policy, True)

    critic_params = [p for p in policy.critic.parameters()] if hasattr(policy, "critic") else []
    prev_params = _clone_params(critic_params)

    warmup_iter = 0
    print(
        f"[INFO] Warmup (critic only) start. threshold={threshold}, min_iters={min_iters}, "
        f"max_iters={max_iters}, warmup_lr={warmup_lr if warmup_lr > 0.0 else 'keep'}, "
        f"adaptive_lr={'on' if use_adaptive_lr else 'off'}"
    )
    while True:
        warmup_iter += 1
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
        delta, norm = _param_delta(prev_params, critic_params)
        rel_delta = delta / (norm + 1.0e-12)
        print(
            f"[INFO] Warmup iter {warmup_iter}: critic param delta={delta:.6e}, "
            f"rel_delta={rel_delta:.6e}"
        )

        if warmup_iter >= min_iters and rel_delta < threshold:
            print("[INFO] Warmup converged by threshold.")
            break
        if max_iters and warmup_iter >= max_iters:
            print("[WARN] Warmup reached max_iters; stopping.")
            break

        prev_params = _clone_params(critic_params)

    for param, requires_grad in original_requires_grad:
        param.requires_grad = requires_grad

    if orig_schedule is not None:
        runner.alg.schedule = orig_schedule

    if orig_lrs is not None and hasattr(runner.alg, "optimizer"):
        for group, lr in zip(runner.alg.optimizer.param_groups, orig_lrs):
            if lr is not None:
                group["lr"] = lr


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if args_cli.checkpoint_path:
        agent_cfg.resume = True
        resume_path = args_cli.checkpoint_path
    elif agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        _load_checkpoint_compatible(runner, resume_path)

    if args_cli.warmup_critic:
        _warmup_critic_only(
            runner,
            threshold=float(args_cli.warmup_threshold),
            min_iters=int(args_cli.warmup_min_iters),
            max_iters=int(args_cli.warmup_max_iters),
            warmup_lr=float(args_cli.warmup_lr),
            use_adaptive_lr=bool(args_cli.warmup_use_adaptive_lr),
        )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
