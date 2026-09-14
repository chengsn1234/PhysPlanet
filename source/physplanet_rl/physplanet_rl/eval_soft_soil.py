"""批量评测软土导航策略，输出 JSON 供多种子/多任务对比。

统计三类指标：任务完成度（成功率、耗时）、安全性（下陷分布、陷车率）、
行为解释（各土层上的停留占比、离岩石的最近距离）。
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a soft-soil navigation checkpoint.")
parser.add_argument("--task", required=True, help="Registered Gym task name.")
parser.add_argument("--checkpoint", required=True, help="Path to a skrl .pt checkpoint.")
parser.add_argument("--algorithm", type=str.upper, default="PPO", choices=["PPO", "TD3", "SAC"])
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=256, help="Number of episodes to aggregate.")
parser.add_argument("--out", default=None, help="Write the summary as JSON to this path.")
parser.add_argument("--label", default="", help="Free-form tag stored in the JSON.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
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

# terrain_class_map 1-based 语义类（与生成 yaml texture_layers 顺序一致）
SOIL_NAMES = ("P10 firm", "P8 medium", "P4 dry sand", "P5 GRC-3")
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"


class EpisodeAccumulator:
    """按环境累计逐步统计量，回合结束时结算成一条记录。"""

    def __init__(self, num_envs: int, device):
        self.records: list[dict] = []
        self.per_env = [0] * num_envs
        zeros = lambda: torch.zeros(num_envs, device=device)  # noqa: E731
        self.steps = zeros()
        self.sinkage_sum = zeros()
        self.sinkage_max = zeros()
        self.deep_steps = zeros()
        self.slip_sum = zeros()
        self.distance = zeros()
        self.drive_energy = zeros()
        self.drive_energy_cls = torch.zeros(num_envs, len(SOIL_NAMES), device=device)
        self.distance_cls = torch.zeros(num_envs, len(SOIL_NAMES), device=device)
        self.soil_steps = torch.zeros(num_envs, len(SOIL_NAMES), device=device)

    def update(self, sinkage, slip, soil_class, step_distance, safe_sinkage, mask, drive_power, dt):
        # mask 排除本步已重置的环境：它们读到的是新回合的出生状态，
        # 位移更是横跨地图的传送，不能计入上一回合。
        active = mask.float()
        mean_sinkage = sinkage.mean(dim=1)
        self.steps += active
        self.sinkage_sum += mean_sinkage * active
        self.sinkage_max = torch.maximum(self.sinkage_max, sinkage.max(dim=1).values * active)
        self.deep_steps += (mean_sinkage > safe_sinkage).float() * active
        self.slip_sum += slip.abs().mean(dim=1) * active
        self.distance += step_distance * active
        self.drive_energy += drive_power * dt * active
        # 类内归属：当步功率/里程记在轮下当前土类上
        self.drive_energy_cls.scatter_add_(1, soil_class.unsqueeze(1), (drive_power * dt * active).unsqueeze(1))
        self.distance_cls.scatter_add_(1, soil_class.unsqueeze(1), (step_distance * active).unsqueeze(1))
        self.soil_steps.scatter_add_(1, soil_class.unsqueeze(1), active.unsqueeze(1))

    def close(self, env_ids, reasons: dict[str, torch.Tensor], step_dt: float):
        for i in env_ids.tolist():
            steps = max(float(self.steps[i]), 1.0)
            self.per_env[i] += 1
            self.records.append({
                "env": i,
                "steps": int(self.steps[i]),
                "duration_s": float(self.steps[i]) * step_dt,
                "distance_m": float(self.distance[i]),
                "sinkage_mean_cm": float(self.sinkage_sum[i]) / steps * 100.0,
                "sinkage_max_cm": float(self.sinkage_max[i]) * 100.0,
                "deep_fraction": float(self.deep_steps[i]) / steps,
                "slip_mean": float(self.slip_sum[i]) / steps,
                "drive_energy_J": float(self.drive_energy[i]),
                "drive_energy_J_per_m": float(self.drive_energy[i]) / max(float(self.distance[i]), 0.01),
                "drive_energy_J_per_m_by_class": {
                    SOIL_NAMES[c]: float(self.drive_energy_cls[i, c]) / max(float(self.distance_cls[i, c]), 0.01)
                    for c in range(len(SOIL_NAMES)) if float(self.distance_cls[i, c]) > 0.05
                },
                "soil_fraction": (self.soil_steps[i] / steps).tolist(),
                **{f"end_{name}": bool(flag[i]) for name, flag in reasons.items()},
            })
        for buf in (self.steps, self.sinkage_sum, self.sinkage_max,
                    self.deep_steps, self.slip_sum, self.distance, self.drive_energy,
                    self.drive_energy_cls, self.distance_cls):
            buf[env_ids] = 0.0
        self.soil_steps[env_ids] = 0.0


def summarize(records: list[dict], reason_names: list[str]) -> dict:
    def mean(key, subset=None):
        rows = subset if subset is not None else records
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    successes = [r for r in records if r.get("end_is_success")]
    summary = {
        "episodes": len(records),
        "success_rate": len(successes) / max(len(records), 1),
        "drive_energy_J": mean("drive_energy_J"),
        "drive_energy_J_per_m": mean("drive_energy_J_per_m"),
        "sinkage_mean_cm": mean("sinkage_mean_cm"),
        "sinkage_p95_cm": float(np.percentile([r["sinkage_max_cm"] for r in records], 95)),
        "deep_fraction": mean("deep_fraction"),
        "slip_mean": mean("slip_mean"),
        "success_duration_s": mean("duration_s", successes),
        "success_distance_m": mean("distance_m", successes),
        "soil_fraction": {
            name: float(np.mean([r["soil_fraction"][i] for r in records]))
            for i, name in enumerate(SOIL_NAMES)
        },
    }
    for name in reason_names:
        summary[f"rate_{name}"] = float(np.mean([r.get(f"end_{name}", False) for r in records]))
    per_class = {}
    for name in SOIL_NAMES:
        vals = [r["drive_energy_J_per_m_by_class"][name] for r in records
                if name in r.get("drive_energy_J_per_m_by_class", {})]
        per_class[name] = float(np.mean(vals)) if vals else float("nan")
    summary["drive_energy_J_per_m_by_class"] = per_class
    return summary


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg["seed"] = set_seed(agent_cfg["seed"])
    env_cfg.seed = agent_cfg["seed"]
    if algorithm_name(agent_cfg) != args_cli.algorithm:
        raise ValueError(f"--algorithm {args_cli.algorithm} does not match agent config {algorithm_name(agent_cfg)}")

    checkpoint = retrieve_file_path(args_cli.checkpoint)
    env = gym.make(args_cli.task, cfg=env_cfg)
    task_env = env.unwrapped
    soil_term = task_env.action_manager.get_term("soft_soil")
    safe_sinkage = task_env.cfg.rewards.sinkage.params["safe_sinkage"]

    soil_class = torch.as_tensor(
        np.load(os.path.join(soil_term.cfg.terrain_dir, "terrain_class_map.npy")).astype(np.int64),
        device=task_env.device)
    soil_mpp = soil_term.terrain_size / max(soil_class.shape[0] - 1, 1)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    agent = build_agent(env, agent_cfg, disable_logging=True)
    agent.init()
    agent.load(checkpoint)
    agent.enable_models_training_mode(False)
    print(f"[eval] task={args_cli.task} checkpoint={checkpoint}")

    robot = task_env.scene["robot"]
    manager = task_env.termination_manager
    reason_names = list(manager.active_terms)
    accumulator = EpisodeAccumulator(task_env.num_envs, task_env.device)

    # 每个环境固定采满 per_env 个回合再停。若按"总回合数达标就停"，
    # 先结束的短回合会被优先收进来、长回合被截断丢弃，样本系统性偏向失败案例
    # （撞岩和陷车都发生得早，成功要跑更久）。
    per_env = max(1, -(-args_cli.episodes // task_env.num_envs))
    print(f"[eval] collecting {per_env} episodes per env x {task_env.num_envs} envs")

    obs, _ = env.reset()
    previous_xy = robot.data.root_pos_w[:, :2].clone()
    while min(accumulator.per_env) < per_env:
        with torch.inference_mode():
            obs, _, terminated, truncated, _ = env.step(eval_actions(agent, obs))

        xy = robot.data.root_pos_w[:, :2]
        col = (xy[:, 0] / soil_mpp).long().clamp(0, soil_class.shape[1] - 1)
        row = (xy[:, 1] / soil_mpp).long().clamp(0, soil_class.shape[0] - 1)
        done = (terminated | truncated).flatten()
        # 驱动机械功率：土壤阻力矩 × 轮角速度（净力代理在巡航时 ≈ 0，会漏掉全部土壤阻力功）
        drive_power = soil_term.power[0]
        accumulator.update(
            soil_term.sinkage, soil_term.slip, (soil_class[row, col] - 1).clamp(0, len(SOIL_NAMES) - 1),
            (xy - previous_xy).norm(dim=-1), safe_sinkage, ~done,
            drive_power, task_env.step_dt)
        previous_xy = xy.clone()

        if done.any():
            reasons = {name: manager.get_term(name) for name in reason_names}
            accumulator.close(done.nonzero().flatten(), reasons, task_env.step_dt)

    # 每个环境只取前 per_env 个，跑得快的环境不能多占样本
    seen: dict[int, int] = {}
    records = []
    for record in accumulator.records:
        index = seen.get(record["env"], 0)
        if index < per_env:
            records.append(record)
        seen[record["env"]] = index + 1
    summary = {"task": args_cli.task, "checkpoint": checkpoint, "label": args_cli.label,
               **summarize(records, reason_names)}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump({"summary": summary, "episodes": records}, f, indent=2, ensure_ascii=False)
        print(f"[eval] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
