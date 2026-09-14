"""Play a skrl checkpoint for a registered RoverLab task."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a RoverLab skrl checkpoint.")
parser.add_argument("--task", required=True, help="Registered Gym task name.")
parser.add_argument("--checkpoint", required=True, help="Path to a skrl .pt checkpoint.")
parser.add_argument("--algorithm", type=str.upper, default="PPO", choices=["PPO", "TD3", "SAC"])
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--real_time", action="store_true")
parser.add_argument("--follow_camera", action="store_true", help="Lock the viewport to the rover.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after N steps; 0 runs until the window closes.")
parser.add_argument("--scenario", choices=("clear", "wide", "tight"), help="Force one navigation scenario.")
parser.add_argument("--goal_radius", type=float, default=0.8, help="Goal marker radius for tasks with target_pose.")
parser.add_argument("--debug_actions", action="store_true", help="Print policy and controller actions.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time

import gymnasium as gym
import torch
from skrl.utils import set_seed
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.sim as sim_utils
from isaacsim.core.prims import SingleXFormPrim
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import physplanet_tasks  # noqa: F401
from physplanet_rl.skrl_utils import algorithm_name, build_agent, eval_actions


algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["seed"] = set_seed(agent_cfg["seed"])
    env_cfg.seed = agent_cfg["seed"]
    if algorithm_name(agent_cfg) != args_cli.algorithm:
        raise ValueError(f"--algorithm {args_cli.algorithm} does not match agent config {algorithm_name(agent_cfg)}")
    if args_cli.scenario:
        if not hasattr(env_cfg.commands, "target_pose"):
            raise ValueError("--scenario requires an environment with a target_pose command")
        weights = {"clear": (1.0, 0.0, 0.0), "wide": (0.0, 1.0, 0.0), "tight": (0.0, 0.0, 1.0)}
        env_cfg.commands.target_pose.scenario_weights = weights[args_cli.scenario]
        print(f"[PLAY] forcing scenario={args_cli.scenario}")
    if args_cli.follow_camera:
        env_cfg.viewer.origin_type = "asset_root"
    checkpoint = retrieve_file_path(args_cli.checkpoint)

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    task_env = env.unwrapped
    action_term = task_env.action_manager.get_term("actions")
    dt = env.unwrapped.step_dt
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(os.path.dirname(checkpoint), "videos"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    agent = build_agent(env, agent_cfg, disable_logging=True)
    agent.init()
    agent.load(checkpoint)
    agent.enable_models_training_mode(False)
    print(f"[INFO] Loaded checkpoint: {checkpoint}")

    obs, _ = env.reset()
    def target_pose():
        try:
            return task_env.command_manager.get_term("target_pose").pos_command_w.clone()
        except (AttributeError, KeyError):
            return None

    def frame_goal_view(target):
        if target is None:
            return
        robot_xy = task_env.scene["robot"].data.root_pos_w[0, :2]
        distance = torch.linalg.vector_norm(target[:2] - robot_xy).item()
        print(f"[PLAY] target=({target[0]:.1f}, {target[1]:.1f}) distance={distance:.1f}m")

    def termination_reasons() -> list[str]:
        manager = task_env.termination_manager
        return [name for name in manager.active_terms if bool(manager.get_term(name)[0].item())]

    target = target_pose()
    frame_goal_view(target[0] if target is not None else None)
    goal_marker = None
    if target is not None:
        target[:, 2] = 2.0
        goal_cfg = sim_utils.SphereCfg(
            radius=args_cli.goal_radius,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.1)),
        )
        goal_cfg.func("/World/Visuals/navigation_goal", goal_cfg, translation=tuple(target[0].cpu().tolist()))
        goal_marker = SingleXFormPrim("/World/Visuals/navigation_goal")
    timestep = 0
    while simulation_app.is_running():
        start = time.time()
        with torch.inference_mode():
            actions = eval_actions(agent, obs)
            obs, _, terminated, truncated, _ = env.step(actions)
        if args_cli.debug_actions and timestep % 10 == 0:
            raw = actions[0].detach().cpu().tolist()
            processed = action_term.processed_actions[0].detach().cpu().tolist()
            print(f"[PLAY] action={raw} processed={processed}")
        target = target_pose()
        if target is not None and goal_marker is not None:
            target[:, 2] = 2.0
            goal_marker.set_world_pose(position=target[0].cpu().numpy())
        timestep += 1
        if torch.any(terminated | truncated):
            print(f"[PLAY] episode_done reasons={termination_reasons()}")
            frame_goal_view(target[0] if target is not None else None)
        elif target is not None and timestep % 30 == 0:
            robot_xy = task_env.scene["robot"].data.root_pos_w[0, :2]
            print(f"[PLAY] step={timestep} distance={torch.linalg.vector_norm(target[0, :2] - robot_xy):.1f}m")
        if args_cli.video and timestep >= args_cli.video_length:
            break
        if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
            break
        if args_cli.real_time:
            time.sleep(max(0.0, dt - (time.time() - start)))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
