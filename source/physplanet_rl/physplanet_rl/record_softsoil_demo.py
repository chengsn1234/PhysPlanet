"""录制软土导航演示：跟随视角视频 + 逐步遥测。

遥测用于事后画俯视对比图（轨迹叠在土类图上），视频用于直观展示。
同一 seed + 同一地形下，各策略拿到的出生点/目标点序列一致，可直接横向对比；
脚本会把 spawn/goal 一并存进 NPZ，画图时会校验是否真的对齐。
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a soft-soil navigation demo.")
parser.add_argument("--task", required=True, help="Registered Gym task name.")
parser.add_argument("--checkpoint", required=True, help="Path to a skrl .pt checkpoint.")
parser.add_argument("--algorithm", type=str.upper, default="PPO", choices=["PPO", "TD3", "SAC"])
parser.add_argument("--num_envs", type=int, default=16, help="Parallel scenarios logged to telemetry.")
parser.add_argument("--episodes", type=int, default=16, help="Episodes to record before stopping.")
parser.add_argument("--min_steps", type=int, default=0,
                    help="Keep stepping until this many steps, so slow episodes finish inside the window.")
parser.add_argument("--seed", type=int, default=7, help="Shared across policies to align scenarios.")
parser.add_argument("--video", action="store_true", help="Also record a follow-camera video.")
parser.add_argument("--video_length", type=int, default=400, help="Video frames (control steps).")
parser.add_argument("--video_env", type=int, default=0, help="Which env the follow camera tracks.")
parser.add_argument("--camera_eye", type=float, nargs=3, default=(-4.5, -4.5, 2.6),
                    help="Follow-camera offset from the rover, in metres.")
parser.add_argument("--spawn", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="Pin the spawn point; with --goal this replays one exact scenario.")
parser.add_argument("--goal", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="Pin the goal point. Use with --num_envs 1 for a single-rover video.")
parser.add_argument("--out", required=True, help="Output NPZ path; video goes next to it.")
parser.add_argument("--label", default="", help="Free-form tag stored in the NPZ.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from skrl.utils import set_seed

import isaaclab_tasks  # noqa: F401
import physplanet_tasks  # noqa: F401
from physplanet_rl.skrl_utils import algorithm_name, build_agent, eval_actions

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg["seed"] = set_seed(args_cli.seed)
    env_cfg.seed = args_cli.seed
    if algorithm_name(agent_cfg) != args_cli.algorithm:
        raise ValueError(f"--algorithm {args_cli.algorithm} does not match agent config {algorithm_name(agent_cfg)}")
    if (args_cli.spawn is None) != (args_cli.goal is None):
        raise ValueError("--spawn and --goal must be given together")
    if args_cli.spawn is not None:
        # 钉死场景后每个回合都一样，用 --num_envs 1 就能录到画面里只有一台车的视频。
        env_cfg.commands.target_pose.fixed_scenario = (
            tuple(args_cli.spawn), tuple(args_cli.goal))
        print(f"[record] fixed scenario: spawn {tuple(args_cli.spawn)} -> goal {tuple(args_cli.goal)}")
    if args_cli.video:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = args_cli.video_env
        env_cfg.viewer.eye = tuple(args_cli.camera_eye)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    checkpoint = retrieve_file_path(args_cli.checkpoint)
    out_path = os.path.abspath(args_cli.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    task_env = env.unwrapped
    soil_term = task_env.action_manager.get_term("soft_soil")
    command_term = task_env.command_manager.get_term("target_pose")
    if args_cli.video:
        # 相机控制器在构造时对 viewer cfg 做了 deepcopy，只改 env_cfg 不生效，
        # 必须在环境建好之后操作实例。
        controller = task_env.viewport_camera_controller
        controller.set_view_env_index(args_cli.video_env)
        controller.update_view_to_asset_root("robot")
        print(f"[record] follow camera tracks env {controller.cfg.env_index}")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.dirname(out_path),
            name_prefix=os.path.splitext(os.path.basename(out_path))[0],
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    agent = build_agent(env, agent_cfg, disable_logging=True)
    agent.init()
    agent.load(checkpoint)
    agent.enable_models_training_mode(False)

    soil_class = torch.as_tensor(
        np.load(os.path.join(soil_term.cfg.terrain_dir, "terrain_class_map.npy")).astype(np.int64),
        device=task_env.device)
    soil_mpp = soil_term.terrain_size / max(soil_class.shape[0] - 1, 1)

    robot = task_env.scene["robot"]
    manager = task_env.termination_manager
    reason_names = list(manager.active_terms)

    # 逐步缓冲：帧 x 环境
    frames: dict[str, list] = {k: [] for k in
                               ("xy", "goal", "sinkage", "slip", "soil", "speed", "episode")}
    episode_id = torch.zeros(task_env.num_envs, dtype=torch.long, device=task_env.device)
    outcomes: list[dict] = []
    finished = 0

    obs, _ = env.reset()
    step = 0
    while (finished < args_cli.episodes or step < args_cli.min_steps) \
            and (not args_cli.video or step < args_cli.video_length):
        with torch.inference_mode():
            obs, _, terminated, truncated, _ = env.step(eval_actions(agent, obs))
        done = (terminated | truncated).flatten()
        if done.any():
            reasons = {name: manager.get_term(name) for name in reason_names}
            for i in done.nonzero().flatten().tolist():
                outcomes.append({"env": i, "episode": int(episode_id[i]),
                                 **{name: bool(flag[i]) for name, flag in reasons.items()}})
            # 重置在 step() 内部完成，本帧读到的已是新回合的出生状态，
            # 所以要先递增编号再记录，否则轨迹里会混进一段跨图传送。
            episode_id[done] += 1
            finished += int(done.sum())

        xy = robot.data.root_pos_w[:, :2]
        col = (xy[:, 0] / soil_mpp).long().clamp(0, soil_class.shape[1] - 1)
        row = (xy[:, 1] / soil_mpp).long().clamp(0, soil_class.shape[0] - 1)
        frames["xy"].append(xy.cpu().numpy().copy())
        frames["goal"].append(command_term.pos_command_w[:, :2].cpu().numpy().copy())
        frames["sinkage"].append(soil_term.sinkage.cpu().numpy().copy())
        frames["slip"].append(soil_term.slip.cpu().numpy().copy())
        frames["soil"].append(soil_class[row, col].cpu().numpy().copy())
        frames["speed"].append(robot.data.root_lin_vel_b[:, 0].cpu().numpy().copy())
        frames["episode"].append(episode_id.cpu().numpy().copy())
        step += 1

    np.savez_compressed(
        out_path,
        label=args_cli.label,
        task=args_cli.task,
        checkpoint=checkpoint,
        terrain_dir=soil_term.cfg.terrain_dir,
        terrain_size=soil_term.terrain_size,
        step_dt=task_env.step_dt,
        reason_names=np.array(reason_names),
        outcomes=np.array([[o["env"], o["episode"]] + [int(o[n]) for n in reason_names] for o in outcomes]),
        **{k: np.asarray(v) for k, v in frames.items()},
    )
    print(f"[record] {out_path}: {step} steps x {task_env.num_envs} envs, {finished} episodes")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
