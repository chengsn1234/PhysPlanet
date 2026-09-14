# Portions adapted from RLRoverLab: https://github.com/abmoRobotics/RLRoverLab
# SPDX-License-Identifier: Apache-2.0

"""Navigation reward terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def progress_to_target_reward(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """Reward only movement that reduces distance to the current target."""
    return env.command_manager.get_term(command_name).metrics["progress"]


def reached_target(env: "ManagerBasedRLEnv", command_name: str, threshold: float) -> torch.Tensor:
    """到达位置目标时给剩余时间奖励。"""
    target = env.command_manager.get_command(command_name)
    distance = torch.norm(target[:, :2], p=2, dim=-1)
    time_steps_to_goal = env.max_episode_length - env.episode_length_buf
    reward_scale = time_steps_to_goal / env.max_episode_length
    return torch.where(distance < threshold, 2.0 * reward_scale, 0.0)


def oscillation_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """action 抖动惩罚：连续两步 action 差分的平方。"""
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    angular_diff = action[:, 0] - prev_action[:, 0]
    penalty = torch.where(torch.abs(angular_diff * 3) > 0.05, torch.square(angular_diff * 3), 0.0)
    return penalty / env.max_episode_length


def action_change_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """惩罚相邻控制动作的大幅变化，抑制前进/倒车抖动。"""
    return torch.square(env.action_manager.action - env.action_manager.prev_action).sum(dim=1)


def speed_action_change_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize speed-command changes without discouraging necessary steering."""
    action = env.action_manager.action[:, 0]
    previous = env.action_manager.prev_action[:, 0]
    return torch.square(action - previous)


def direction_change_penalty(env: "ManagerBasedRLEnv", threshold: float) -> torch.Tensor:
    """Charge a one-step cost when the policy explicitly swaps drive direction."""
    action = env.action_manager.action[:, 0]
    previous = env.action_manager.prev_action[:, 0]
    changed = ((action > threshold) & (previous < -threshold)) | (
        (action < -threshold) & (previous > threshold)
    )
    return changed.float()


def committed_direction_change_penalty(env: "ManagerBasedRLEnv", action_name: str) -> torch.Tensor:
    """Charge only a direction change accepted by the committed-speed controller."""
    action_term = env.action_manager.get_term(action_name)
    direction_switch = getattr(action_term, "direction_switch", None)
    if direction_switch is None:
        return torch.zeros(env.num_envs, device=env.device)
    return direction_switch.clone()


def reverse_motion_penalty(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, max_reverse_velocity: float
) -> torch.Tensor:
    """Charge reverse distance so it remains a recovery action rather than cruising."""
    velocity = env.scene[asset_cfg.name].data.root_lin_vel_b[:, 0]
    return (-velocity).clamp_min(0.0) / max_reverse_velocity


def time_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """每个控制步固定代价，鼓励更快到达目标。"""
    return torch.ones(env.num_envs, device=env.device)


def idle_while_far_penalty(
    env: "ManagerBasedRLEnv", command_name: str, speed_threshold: float, distance_threshold: float
) -> torch.Tensor:
    """Penalize choosing to idle while the goal is still far away."""
    speed_action = env.action_manager.action[:, 0]
    target = env.command_manager.get_command(command_name)[:, :2]
    distance = torch.linalg.vector_norm(target, dim=1)
    return ((speed_action.abs() < speed_threshold) & (distance > distance_threshold)).float()


def low_speed_while_far_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
    speed_threshold: float,
    distance_threshold: float,
) -> torch.Tensor:
    """Penalize real stalling instead of trusting a raw policy action threshold."""
    speed = env.scene[asset_cfg.name].data.root_lin_vel_b[:, 0].abs()
    target = env.command_manager.get_command(command_name)[:, :2]
    distance = torch.linalg.vector_norm(target, dim=1)
    return ((speed < speed_threshold) & (distance > distance_threshold)).float()


def angle_to_target_penalty(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """朝向偏离目标>2rad 时的惩罚。"""
    target_b = env.command_manager.get_command(command_name)[:, :2]
    angle = torch.atan2(target_b[:, 1], target_b[:, 0])
    return torch.where(torch.abs(angle) > 2.0, torch.abs(angle) / env.max_episode_length, 0.0)


def collision_penalty(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """碰撞惩罚：contact sensor 检测到力>threshold 时给 1。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    forces_active = torch.norm(forces, dim=-1).amax(dim=(1, 2)) > threshold
    return torch.where(forces_active, 1.0, 0.0)


def out_of_bounds_penalty(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, terrain_size: float, margin: float
) -> torch.Tensor:
    """Penalty when the rover leaves the valid area of the shared terrain."""
    pos = env.scene[asset_cfg.name].data.root_pos_w[:, :2]
    outside = ((pos < margin) | (pos > terrain_size - margin)).any(dim=1)
    return outside.float()


def far_from_target_reward(env: "ManagerBasedRLEnv", command_name: str, margin: float = 3.0) -> torch.Tensor:
    """远离目标惩罚：距离超过初始距离加安全余量时给 1。"""
    target = env.command_manager.get_command(command_name)[:, :2]
    distance = torch.norm(target, p=2, dim=-1)
    command_term = env.command_manager.get_term(command_name)
    threshold = command_term.target_distance + margin
    return torch.where(distance > threshold, 1.0, 0.0)


def angle_to_goal_reward(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """朝向目标的 dense 正向 reward。"""
    target_b = env.command_manager.get_command(command_name)[:, :2]
    distance = torch.norm(target_b, p=2, dim=-1)
    angle_b = env.command_manager.get_command(command_name)[:, 3]
    angle_reward = (1.0 / (1.0 + distance)) * 1.0 / (1.0 + torch.abs(angle_b))
    return angle_reward / env.max_episode_length
