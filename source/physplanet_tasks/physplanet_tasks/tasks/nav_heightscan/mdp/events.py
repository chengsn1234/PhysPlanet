"""Event terms：在全局地形的各 env 区域中心重置 rover。"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from .observations import _bilinear_sample


class ResetRootStateRover(ManagerTermBase):
    """Reset rover at its assigned point on the shared terrain."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        height_path = os.path.join(cfg.params["terrain_dir"], "height_gt_map.npy")
        if not os.path.exists(height_path):
            raise FileNotFoundError(f"Ground height map not found: {height_path}")
        self.height_map = torch.as_tensor(
            np.load(height_path), dtype=torch.float32, device=env.device
        )
        self.terrain_size = cfg.params["terrain_size"]
        self.spawn_margin = cfg.params["spawn_margin"]
        self.spawn_search_radius = cfg.params["spawn_search_radius"]
        rock_path = os.path.join(cfg.params["terrain_dir"], "rock_map.npy")
        self.rock_map = None
        if os.path.exists(rock_path):
            rock_map = torch.as_tensor(np.load(rock_path), dtype=torch.float32, device=env.device)
            self.rock_mpp = self.terrain_size / max(rock_map.shape[0] - 1, 1)
            radius = max(math.ceil(cfg.params["spawn_clearance"] / self.rock_mpp - 1.0e-6), 0)
            if radius:
                rock_map = F.max_pool2d(rock_map[None, None], 2 * radius + 1, stride=1, padding=radius)[0, 0]
            self.rock_map = rock_map

    def _is_clear(self, xy: torch.Tensor) -> torch.Tensor:
        inside = (
            (xy[:, 0] >= self.spawn_margin)
            & (xy[:, 0] <= self.terrain_size - self.spawn_margin)
            & (xy[:, 1] >= self.spawn_margin)
            & (xy[:, 1] <= self.terrain_size - self.spawn_margin)
        )
        if self.rock_map is None:
            return inside
        col = (xy[:, 0] / self.rock_mpp).long().clamp(0, self.rock_map.shape[1] - 1)
        row = (xy[:, 1] / self.rock_mpp).long().clamp(0, self.rock_map.shape[0] - 1)
        return inside & (self.rock_map[row, col] < 0.5)

    def _fallback_positions(self, centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose the nearest clear point from a deterministic local grid."""
        axis = torch.linspace(-self.spawn_search_radius, self.spawn_search_radius, 11, device=self.device)
        grid_x, grid_y = torch.meshgrid(axis, axis, indexing="xy")
        offsets = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1)
        offsets = offsets[torch.argsort(torch.sum(offsets.square(), dim=1))]
        candidates = centers[:, None, :] + offsets[None, :, :]
        clear = self._is_clear(candidates.flatten(0, 1)).view(len(centers), -1)
        found = clear.any(dim=1)
        index = clear.float().argmax(dim=1)
        return candidates[torch.arange(len(centers), device=self.device), index], found

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        terrain_dir: str,
        terrain_size: float,
        spawn_jitter: float,
        spawn_clearance: float,
        spawn_search_radius: float = 4.0,
        spawn_margin: float = 3.0,
        z_offset: float = 0.5,
    ):
        asset: Articulation = env.scene[asset_cfg.name]
        positions = env.scene.env_origins[env_ids].clone()
        positions[:, :2] += terrain_size * 0.5

        clear = self._is_clear(positions[:, :2])
        for _ in range(50):
            need = torch.nonzero(~clear, as_tuple=False).flatten()
            if len(need) == 0:
                break
            center_xy = env.scene.env_origins[env_ids[need], :2] + terrain_size * 0.5
            candidate = center_xy + torch.empty_like(center_xy).uniform_(-spawn_jitter, spawn_jitter)
            valid = self._is_clear(candidate)
            positions[need, :2] = torch.where(valid.unsqueeze(1), candidate, positions[need, :2])
            clear = self._is_clear(positions[:, :2])
        if not torch.all(clear):
            need = torch.nonzero(~clear, as_tuple=False).flatten()
            centers = env.scene.env_origins[env_ids[need], :2] + terrain_size * 0.5
            fallback, found = self._fallback_positions(centers)
            positions[need[found], :2] = fallback[found]
            clear = self._is_clear(positions[:, :2])
        if not torch.all(clear):
            bad_ids = env_ids[~clear].tolist()
            raise RuntimeError(f"No clear spawn point within {self.spawn_search_radius}m for envs {bad_ids}")

        positions[:, 2] = _bilinear_sample(self.height_map, positions[:, :2], terrain_size) + z_offset
        if not hasattr(env, "_rover_spawn_xy"):
            env._rover_spawn_xy = torch.zeros(env.num_envs, 2, device=env.device)
        env._rover_spawn_xy[env_ids] = positions[:, :2]

        angle = torch.rand(len(env_ids), device=env.device) * 2.0 * torch.pi
        quat = torch.zeros(len(env_ids), 4, device=env.device)
        quat[:, 0] = torch.cos(angle / 2)
        quat[:, 3] = torch.sin(angle / 2)

        root_state = asset.data.default_root_state[env_ids].clone()
        root_state[:, :3] = positions
        root_state[:, 3:7] = quat
        root_state[:, 7:] = 0.0
        if not hasattr(env, "_rover_spawn_pose"):
            env._rover_spawn_pose = torch.zeros(env.num_envs, 7, device=env.device)
        env._rover_spawn_pose[env_ids] = root_state[:, :7]
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)
        asset.write_joint_state_to_sim(
            asset.data.default_joint_pos[env_ids],
            torch.zeros_like(asset.data.default_joint_vel[env_ids]),
            env_ids=env_ids,
        )
