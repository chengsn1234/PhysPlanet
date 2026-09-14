# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train a registered RoverLab task with skrl."""

import argparse
import sys

from isaaclab.app import AppLauncher

# argparse
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Registered Gym task name.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="Legacy on-policy iteration count (rollouts x N).")
parser.add_argument("--max_timesteps", type=int, default=None, help="Environment steps for any supported algorithm.")
parser.add_argument(
    "--algorithm", type=str.upper, default="PPO", choices=["PPO", "TD3", "SAC"],
    help="The RL algorithm.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime

import skrl
from packaging import version
from skrl.utils import set_seed

# check for minimum supported skrl version
# 2.0 起 agent 配置改为 dataclass（ExperimentCfg / PPO_CFG），skrl_utils 依赖该形式
SKRL_VERSION = "2.0.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from isaaclab.envs import (
    DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
import isaaclab_tasks  # noqa: F401
import physplanet_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from physplanet_rl.skrl_utils import ON_POLICY_ALGORITHMS, RoverIsaacLabWrapper, algorithm_name, build_agent

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    if args_cli.max_timesteps is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_timesteps
    elif args_cli.max_iterations:
        if args_cli.algorithm not in ON_POLICY_ALGORITHMS:
            raise ValueError("--max_iterations is on-policy only; use --max_timesteps for TD3 or SAC")
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    agent_cfg["seed"] = set_seed(agent_cfg["seed"])
    env_cfg.seed = agent_cfg["seed"]
    if algorithm_name(agent_cfg) != args_cli.algorithm:
        raise ValueError(f"--algorithm {args_cli.algorithm} does not match agent config {algorithm_name(agent_cfg)}")

    # logging directory
    task_log_name = args_cli.task.removeprefix("Isaac-").removesuffix("-v0").replace("-", "_").lower()
    log_root_path = os.path.join(os.environ.get("ROVER_RL_LOG_ROOT", "logs"), "skrl", task_log_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_torch"
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv) and args_cli.algorithm in ON_POLICY_ALGORITHMS:
        env = multi_agent_to_single_agent(env)

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

    env = RoverIsaacLabWrapper(env)

    from skrl.trainers.torch import SequentialTrainer
    agent = build_agent(env, agent_cfg)

    trainer_cfg = {k: v for k, v in agent_cfg["trainer"].items() if k != "class"}
    trainer = SequentialTrainer(cfg=trainer_cfg, agents=agent, env=env)

    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent.load(resume_path)

    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
