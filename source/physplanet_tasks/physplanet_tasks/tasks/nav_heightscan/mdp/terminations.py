# Portions adapted from RLRoverLab: https://github.com/abmoRobotics/RLRoverLab
# SPDX-License-Identifier: Apache-2.0

"""Navigation termination terms."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensor
from scipy.ndimage import distance_transform_edt

from .observations import _bilinear_sample

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def is_success(env: "ManagerBasedRLEnv", command_name: str, threshold: float) -> torch.Tensor:
    """到达位置目标。"""
    target = env.command_manager.get_command(command_name)
    distance = torch.norm(target[:, :2], p=2, dim=-1)
    return distance < threshold


def far_from_target(env: "ManagerBasedRLEnv", command_name: str, margin: float = 3.0) -> torch.Tensor:
    """远离目标：距离超过初始距离加安全余量。"""
    target = env.command_manager.get_command(command_name)[:, :2]
    distance = torch.norm(target, p=2, dim=-1)
    command_term = env.command_manager.get_term(command_name)
    threshold = command_term.target_distance + margin
    return torch.where(distance > threshold, True, False)


def collision_with_obstacles(
    env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """碰撞岩石：contact sensor 检测到力>threshold。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    return torch.norm(forces, dim=-1).amax(dim=(1, 2)) > threshold


class RockCollisionGT(ManagerTermBase):
    """Terminate when the rover footprint enters the rock occupancy GT."""

    def __init__(self, cfg: TerminationTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        rock_path = os.path.join(cfg.params["terrain_dir"], "rock_map.npy")
        if not os.path.exists(rock_path):
            raise FileNotFoundError(f"Rock map not found: {rock_path}")
        self.terrain_size = cfg.params["terrain_size"]
        rock_map = torch.as_tensor(np.load(rock_path), dtype=torch.float32, device=env.device)
        self.height, self.width = rock_map.shape
        self.mpp = self.terrain_size / max(self.height - 1, 1)
        radius = max(math.ceil(cfg.params.get("clearance", 0.0) / self.mpp - 1.0e-6), 0)
        if radius:
            rock_map = F.max_pool2d(rock_map[None, None], 2 * radius + 1, stride=1, padding=radius)[0, 0]
        self.rock_map = rock_map
        asset = env.scene[cfg.params["asset_cfg"].name]
        self.body_ids, _ = asset.find_bodies(cfg.params["body_names"])

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg,
        body_names: list[str],
        terrain_dir: str,
        terrain_size: float,
        clearance: float,
    ) -> torch.Tensor:
        asset = env.scene[self.cfg.params["asset_cfg"].name]
        xy = asset.data.body_pos_w[:, self.body_ids, :2]
        inside = ((xy >= 0.0) & (xy <= self.terrain_size)).all(dim=-1)
        col = (xy[..., 0] / self.mpp).long().clamp(0, self.width - 1)
        row = (xy[..., 1] / self.mpp).long().clamp(0, self.height - 1)
        occupied = self.rock_map[row, col] > 0.5
        return (occupied & inside).any(dim=1)


class RockCollisionPenalty(RockCollisionGT):
    """Binary rock-contact penalty using the same footprint GT."""

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg,
        body_names: list[str],
        terrain_dir: str,
        terrain_size: float,
        clearance: float,
    ) -> torch.Tensor:
        return super().__call__(env, asset_cfg, body_names, terrain_dir, terrain_size, clearance).float()


class RockProximityPenalty(RockCollisionGT):
    """Continuous cost inside a safety band around the RockGT collision envelope."""

    def __init__(self, cfg: TerminationTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        warning_distance = float(cfg.params["warning_distance"])
        if warning_distance <= 0.0:
            raise ValueError("warning_distance must be positive")
        if torch.any(self.rock_map > 0.5):
            free_space = (self.rock_map < 0.5).cpu().numpy()
            distance_map = distance_transform_edt(free_space).astype(np.float32) * self.mpp
        else:
            distance_map = np.full(self.rock_map.shape, warning_distance, dtype=np.float32)
        self.distance_map = torch.as_tensor(distance_map, device=env.device)

    def _risk(self, env: "ManagerBasedRLEnv", warning_distance: float) -> torch.Tensor:
        asset = env.scene[self.cfg.params["asset_cfg"].name]
        xy = asset.data.body_pos_w[:, self.body_ids, :2]
        flat_xy = xy.reshape(-1, 2)
        distance = _bilinear_sample(self.distance_map, flat_xy, self.terrain_size).reshape(xy.shape[:2])
        inside = ((xy >= 0.0) & (xy <= self.terrain_size)).all(dim=-1)
        distance = torch.where(inside, distance, torch.full_like(distance, warning_distance))
        nearest_distance = distance.amin(dim=1)
        return ((warning_distance - nearest_distance) / warning_distance).clamp(0.0, 1.0)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg,
        body_names: list[str],
        terrain_dir: str,
        terrain_size: float,
        clearance: float,
        warning_distance: float,
    ) -> torch.Tensor:
        return self._risk(env, warning_distance)


class RockClearanceProgressReward(RockProximityPenalty):
    """Reward only a net movement out of the RockGT safety band."""

    def __init__(self, cfg: TerminationTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._previous_risk = torch.zeros(self.num_envs, device=env.device)
        self._risk_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._previous_risk.zero_()
            self._risk_initialized.zero_()
        else:
            self._previous_risk[env_ids] = 0.0
            self._risk_initialized[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg,
        body_names: list[str],
        terrain_dir: str,
        terrain_size: float,
        clearance: float,
        warning_distance: float,
    ) -> torch.Tensor:
        risk = self._risk(env, warning_distance)
        reward = torch.where(self._risk_initialized, self._previous_risk - risk, torch.zeros_like(risk))
        self._previous_risk[:] = risk
        self._risk_initialized[:] = True
        return reward


def terrain_out_of_bounds(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, terrain_size: float, margin: float
) -> torch.Tensor:
    """Terminate when the rover approaches the shared terrain boundary."""
    pos = env.scene[asset_cfg.name].data.root_pos_w[:, :2]
    return ((pos < margin) | (pos > terrain_size - margin)).any(dim=1)
