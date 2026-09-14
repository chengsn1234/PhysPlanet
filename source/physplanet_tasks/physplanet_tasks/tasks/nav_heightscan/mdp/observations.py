# Portions adapted from RLRoverLab: https://github.com/abmoRobotics/RLRoverLab
# SPDX-License-Identifier: Apache-2.0

"""Observation terms：height_scan（查表）+ 导航辅助 obs。

height_scan 内联 gt_utils 的 make_world_grid + compute_patches + terramechanics.bilinear_sample，
查 height_merged_map.npy（含岩石高程）。旧 terrain 无此文件则回退 height_gt_map.npy + 警告。
不依赖 physplanet_assets.utils（避免 packaging 问题）。
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── 内联：bilinear_sample（移植自 terramechanics.py）──

def _bilinear_sample(grid: torch.Tensor, xy: torch.Tensor, terrain_size: float) -> torch.Tensor:
    H, W = grid.shape[:2]
    inv_mpp = max(H - 1, 1) / terrain_size
    c = xy[:, 0] * inv_mpp
    r = xy[:, 1] * inv_mpp
    c0 = torch.floor(c).long().clamp(0, W - 2)
    r0 = torch.floor(r).long().clamp(0, H - 2)
    fc = (c - c0.float()).clamp(0.0, 1.0)
    fr = (r - r0.float()).clamp(0.0, 1.0)
    if grid.dim() == 3:
        fc = fc.unsqueeze(-1)
        fr = fr.unsqueeze(-1)
    v00 = grid[r0, c0]
    v01 = grid[r0, c0 + 1]
    v10 = grid[r0 + 1, c0]
    v11 = grid[r0 + 1, c0 + 1]
    v0 = v00 * (1 - fc) + v01 * fc
    v1 = v10 * (1 - fc) + v11 * fc
    return v0 * (1 - fr) + v1 * fr


def _nearest_sample(grid: torch.Tensor, xy: torch.Tensor, terrain_size: float) -> torch.Tensor:
    """Sample a categorical map with the same world-to-grid convention."""
    height, width = grid.shape[:2]
    inv_mpp = max(height - 1, 1) / terrain_size
    col = torch.round(xy[:, 0] * inv_mpp).long().clamp(0, width - 1)
    row = torch.round(xy[:, 1] * inv_mpp).long().clamp(0, height - 1)
    return grid[row, col]


# ── 内联：make_world_grid + compute_patches（移植自 gt_utils.py）──

def _make_local_grid(scan_size, scan_res, device):
    """Create the scan offsets once; the same offsets are reused by every step."""
    sx, sy = float(scan_size[0]), float(scan_size[1])
    res = float(scan_res)
    xs = torch.arange(-sx / 2, sx / 2 + res / 2, res, device=device)
    ys = torch.arange(-sy / 2, sy / 2 + res / 2, res, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

def _make_world_grid(robot_pos_w, heading_w, local_xy):
    """Rotate cached local offsets and return world points as (E, N, 2)."""
    cos_y = torch.cos(heading_w).unsqueeze(1)
    sin_y = torch.sin(heading_w).unsqueeze(1)
    local_x = local_xy[:, 0].unsqueeze(0)
    local_y = local_xy[:, 1].unsqueeze(0)
    world_x = cos_y * local_x - sin_y * local_y
    world_y = sin_y * local_x + cos_y * local_y
    return torch.stack((world_x, world_y), dim=-1) + robot_pos_w[:, None, :2]


def _compute_patches(world_xy, height_t, terrain_size):
    """网格点查高程表。返回 (E, N)。"""
    E, N, _ = world_xy.shape
    flat_xy = world_xy.reshape(E * N, 2)
    result = _bilinear_sample(height_t, flat_xy, terrain_size)
    return result.reshape(E, N)


# ── 纯函数 obs（移植 RLRoverLab，签名一致）──

def angle_to_target_observation(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """车体系下车头到目标的角度 atan2(y_b, x_b)。"""
    target_b = env.command_manager.get_command(command_name)[:, :2]
    angle = torch.atan2(target_b[:, 1], target_b[:, 0])
    return angle.unsqueeze(-1)


def distance_to_target_euclidean(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """车体系下到目标的欧氏距离。"""
    target_b = env.command_manager.get_command(command_name)[:, :2]
    dist = torch.norm(target_b, p=2, dim=-1)
    return dist.unsqueeze(-1)


def angle_diff(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """车体系下朝向误差（command 第 4 维 heading_b）。"""
    return env.command_manager.get_command(command_name)[:, 3:4]


def base_forward_velocity(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """车体系前向线速度。"""
    return env.scene[asset_cfg.name].data.root_lin_vel_b[:, 0:1]


def base_yaw_rate(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """车体系 yaw 角速度。"""
    return env.scene[asset_cfg.name].data.root_ang_vel_b[:, 2:3]


def committed_direction_state(env: "ManagerBasedRLEnv", action_name: str) -> torch.Tensor:
    """Expose the V3 controller state so the policy never acts on hidden hold state."""
    action_term = env.action_manager.get_term(action_name)
    direction = getattr(action_term, "drive_direction", None)
    hold_steps = getattr(action_term, "hold_steps", None)
    if direction is None or hold_steps is None:
        return torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device)
    hold_limit = max(int(getattr(action_term.cfg, "switch_hold_steps", 1)), 1)
    return torch.stack((direction.float(), hold_steps.float() / hold_limit), dim=1)


# ── HeightScanGT：class-based obs term，查表得含岩石高程 ──

class HeightScanGT(ManagerTermBase):
    """init 时加载 height_merged_map（含岩石），call 时 make_world_grid 查表。

    obs = robot_z - sampled_ground_z - height_offset
    返回 (E, grid_h*grid_w)，与 RLRoverLab height_scan_rover 语义一致。
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        terrain_dir = cfg.params["terrain_dir"]
        self.scan_size = cfg.params["scan_size"]
        self.scan_res = cfg.params["scan_res"]
        self.height_offset = cfg.params["height_offset"]
        self.terrain_size = cfg.params["terrain_size"]
        self.enabled = bool(cfg.params.get("enabled", True))
        self.local_xy = _make_local_grid(self.scan_size, self.scan_res, env.device)

        # 优先 height_merged_map（含岩石），回退 height_gt_map（纯地形）
        merged_path = os.path.join(terrain_dir, "height_merged_map.npy")
        if os.path.exists(merged_path):
            arr = np.load(merged_path)
            self.height_t = torch.as_tensor(arr, dtype=torch.float32, device=env.device)
            print(f"[HeightScanGT] Loaded height_merged_map: shape={self.height_t.shape}")
        else:
            gt_path = os.path.join(terrain_dir, "height_gt_map.npy")
            if os.path.exists(gt_path):
                rock_path = os.path.join(terrain_dir, "rock_map.npy")
                if os.path.exists(rock_path) and np.any(np.load(rock_path)):
                    raise FileNotFoundError(
                        f"Terrain has rocks but no height_merged_map.npy: {terrain_dir}. "
                        "Regenerate it with the current terrain pipeline."
                    )
                arr = np.load(gt_path)
                self.height_t = torch.as_tensor(arr, dtype=torch.float32, device=env.device)
                print(f"[HeightScanGT] WARN: height_merged_map not found, "
                      f"fallback height_gt_map (no rocks). shape={self.height_t.shape}")
            else:
                raise FileNotFoundError(
                    f"Neither height_merged_map.npy nor height_gt_map.npy found in {terrain_dir}")

    def __call__(self, env, asset_cfg, scan_size, scan_res, height_offset, terrain_dir, terrain_size,
                 enabled=True):
        robot = env.scene[asset_cfg.name]
        if not enabled or not self.enabled:
            return torch.zeros(
                (robot.data.root_pos_w.shape[0], self.local_xy.shape[0]),
                dtype=torch.float32, device=env.device)
        hs_xy = _make_world_grid(robot.data.root_pos_w, robot.data.heading_w, self.local_xy)
        height_patch = _compute_patches(hs_xy, self.height_t, self.terrain_size)
        return robot.data.root_pos_w[:, 2:3] - height_patch - self.height_offset


class SemanticScanGT(ManagerTermBase):
    """Local one-hot semantic labels from the terrain class map.

    terrain_class_map.npy is 1-based (1 = loose soil ... semantic_count = bedrock)
    and shares the pixel grid with soil_params_map, so the semantic channel is
    always consistent with the physics the rover actually experiences.  The
    observation exposes only the semantic label, never the class index.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg.params
        terrain_dir = params["terrain_dir"]
        class_path = os.path.join(terrain_dir, "terrain_class_map.npy")
        if not os.path.exists(class_path):
            raise FileNotFoundError(
                f"SemanticScanGT requires terrain_class_map.npy; regenerate terrain: {terrain_dir}")

        class_map = np.load(class_path)
        if class_map.ndim != 2:
            raise ValueError(f"terrain_class_map must be 2-D, got {class_map.shape}")
        semantic_count = int(params.get("semantic_count", 0)) or len(np.unique(class_map))
        class_ids = torch.as_tensor(class_map, dtype=torch.long, device=env.device) - 1
        if int(class_ids.min()) < 0 or int(class_ids.max()) >= semantic_count:
            raise ValueError(
                f"terrain_class_map values {int(class_ids.min()) + 1}..{int(class_ids.max()) + 1} "
                f"outside 1..{semantic_count}")

        self.class_map = class_ids
        self.semantic_count = semantic_count
        self.terrain_size = float(params["terrain_size"])
        self.local_xy = _make_local_grid(
            params["scan_size"], params["scan_res"], env.device)
        self.enabled = bool(params.get("enabled", True))
        print(f"[SemanticScanGT] {self.local_xy.shape[0]} points x {semantic_count} channels, "
              f"enabled={self.enabled}, classes={int(class_ids.max()) + 1}")

    def __call__(self, env, asset_cfg, scan_size, scan_res, terrain_dir, terrain_size,
                 semantic_count=None, enabled=True):
        robot = env.scene[asset_cfg.name]
        num_envs = robot.data.root_pos_w.shape[0]
        num_points = self.local_xy.shape[0]
        if not enabled or not self.enabled:
            return torch.zeros(
                (num_envs, num_points * self.semantic_count),
                dtype=torch.float32, device=env.device)

        xy = _make_world_grid(robot.data.root_pos_w, robot.data.heading_w, self.local_xy)
        class_ids = _nearest_sample(
            self.class_map, xy.reshape(-1, 2), self.terrain_size)
        labels = F.one_hot(class_ids, num_classes=self.semantic_count).to(torch.float32)
        return labels.reshape(num_envs, num_points * self.semantic_count)
