"""Navigation command with reset-time RockGT scenario sampling."""

from __future__ import annotations

import math
import os
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.command_manager import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

from ..mdp.observations import _bilinear_sample


class NavigationCommand(CommandTerm):
    """世界坐标导航目标，resample 时用 rock_map 验证避岩石。"""

    cfg: "NavigationCommandCfg"

    def __init__(self, cfg: "NavigationCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # 世界坐标目标缓存
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        # 车体坐标目标（喂给 reward/obs）
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)

        # 初始距离（供 far_from_target reward/termination 判定阈值）
        self.target_distance = torch.zeros(self.num_envs, device=self.device)

        # rock_map 加载用于目标验证
        self.terrain_size = cfg.terrain_size
        rock_path = f"{cfg.terrain_dir}/rock_map.npy"
        self.rock_map = None
        if os.path.exists(rock_path):
            self.rock_map = torch.as_tensor(
                np.load(rock_path), dtype=torch.float32, device=self.device)
            self.rock_h, self.rock_w = self.rock_map.shape
            self.rock_mpp = self.terrain_size / max(self.rock_h - 1, 1)   # 米/像素
            radius = max(math.ceil(cfg.path_clearance / self.rock_mpp - 1.0e-6), 0)
            if radius:
                self.path_rock_map = F.max_pool2d(
                    self.rock_map[None, None], 2 * radius + 1, stride=1, padding=radius
                )[0, 0]
            else:
                self.path_rock_map = self.rock_map
            spawn_radius = max(math.ceil(cfg.spawn_clearance / self.rock_mpp - 1.0e-6), 0)
            if spawn_radius:
                self.spawn_rock_map = F.max_pool2d(
                    self.rock_map[None, None], 2 * spawn_radius + 1, stride=1, padding=spawn_radius
                )[0, 0]
            else:
                self.spawn_rock_map = self.rock_map
            target_radius = max(math.ceil(cfg.rock_safety_margin / self.rock_mpp - 1.0e-6), 0)
            if target_radius:
                self.target_rock_map = F.max_pool2d(
                    self.rock_map[None, None], 2 * target_radius + 1, stride=1, padding=target_radius
                )[0, 0]
            else:
                self.target_rock_map = self.rock_map
        else:
            self.path_rock_map = None
            self.spawn_rock_map = None
            self.target_rock_map = None
            print(f"[NavigationCommand] WARN: rock_map.npy not found at {rock_path}, "
                  f"target validation disabled")

        # 软土排除掩码：最软的几层车根本开不出来，出生点或目标点落在里面的回合
        # 对任何策略都是死局，只会给两组实验加同样的噪声，采样阶段直接避开。
        self.soft_soil_map = None
        if cfg.max_soil_class is not None:
            class_path = os.path.join(cfg.terrain_dir, "terrain_class_map.npy")
            if not os.path.exists(class_path):
                raise FileNotFoundError(
                    f"max_soil_class is set but terrain_class_map.npy is missing: {class_path}")
            soil_class = torch.as_tensor(
                np.load(class_path).astype(np.float32), device=self.device)
            too_soft = (soil_class > cfg.max_soil_class).float()
            self.soil_mpp = self.terrain_size / max(soil_class.shape[0] - 1, 1)
            radius = max(math.ceil(cfg.soil_safety_margin / self.soil_mpp - 1.0e-6), 0)
            if radius:
                too_soft = F.max_pool2d(too_soft[None, None], 2 * radius + 1, stride=1, padding=radius)[0, 0]
            self.soft_soil_map = too_soft
            print(f"[NavigationCommand] soft-soil exclusion: class>{cfg.max_soil_class} "
                  f"covers {float(too_soft.mean()) * 100:.1f}% of the map after {cfg.soil_safety_margin}m inflation")

        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["progress"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["scenario_type"] = torch.zeros(self.num_envs, device=self.device)
        self._previous_error_pos = torch.full((self.num_envs,), torch.nan, device=self.device)
        self._scenario_stats_printed = False
        self.height_map = None
        if cfg.sample_spawn_and_goal:
            height_path = os.path.join(cfg.terrain_dir, "height_gt_map.npy")
            if not os.path.exists(height_path):
                raise FileNotFoundError(f"Ground height map not found: {height_path}")
            self.height_map = torch.as_tensor(np.load(height_path), dtype=torch.float32, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """车体坐标目标 [x_b, y_b, z_b, heading_b]，shape (E, 4)。"""
        # Reset observations are computed before CommandManager.compute().
        # Refresh here so the first step never sees the previous episode's target.
        self._update_command()
        return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)

    def _apply_fixed_scenario(self, env_ids: Sequence[int]):
        """所有环境钉在同一起点/目标，跳过采样，用于单车复现指定场景。"""
        spawn, goal = self.cfg.fixed_scenario
        n = len(env_ids)
        centers = torch.tensor(spawn, dtype=torch.float32, device=self.device).expand(n, 2).clone()
        targets = torch.tensor(goal, dtype=torch.float32, device=self.device).expand(n, 2).clone()
        self.pos_command_w[env_ids, :2] = targets
        self.pos_command_w[env_ids, 2] = self.robot.data.default_root_state[env_ids, 2]
        self.target_distance[env_ids] = torch.norm(targets - centers, dim=1)
        self._previous_error_pos[env_ids] = torch.nan
        heading = torch.atan2(targets[:, 1] - centers[:, 1], targets[:, 0] - centers[:, 0])
        if self.cfg.sample_spawn_and_goal:
            self._write_spawn_state(env_ids, centers, heading)
        elif self.cfg.use_scenario_profile:
            self._write_spawn_heading(env_ids, heading)
        self.heading_command_w[env_ids] = heading
        self.metrics["scenario_type"][env_ids] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if self.cfg.fixed_scenario is not None:
            self._apply_fixed_scenario(env_ids)
            return
        if self.cfg.sample_spawn_and_goal:
            centers = self._sample_spawn_positions(n)
        else:
            spawn_xy = getattr(self._env, "_rover_spawn_xy", None)
            if spawn_xy is None:
                centers = self._env.scene.env_origins[env_ids, :2] + self.terrain_size * 0.5
            else:
                centers = spawn_xy[env_ids]

        targets_xy = torch.zeros(n, 2, device=self.device)
        d_min, d_max = self.cfg.target_distance_range
        requested_scenario_types = self._sample_scenario_types(n)
        # A tight recovery case is geometrically rare on some generated maps.  Keep
        # the requested curriculum distribution when possible, but record the
        # scenario that was actually sampled if a request must be downgraded.
        scenario_types = requested_scenario_types.clone()
        resolved = torch.zeros(n, dtype=torch.bool, device=self.device)

        def sample_remaining() -> None:
            """Sample all unresolved environments for their current scenario type."""
            attempts = (
                self.cfg.max_resample_attempts + self.cfg.candidate_batch_size - 1
            ) // self.cfg.candidate_batch_size
            for _ in range(attempts):
                need = torch.nonzero(~resolved, as_tuple=False).flatten()
                if len(need) == 0:
                    break
                count = len(need)
                batch_size = self.cfg.candidate_batch_size
                if self.cfg.sample_spawn_and_goal:
                    candidate_centers = self._sample_spawn_positions(count * batch_size).view(count, batch_size, 2)
                else:
                    candidate_centers = centers[need, None, :].expand(-1, batch_size, -1)
                dist = torch.empty(count, batch_size, device=self.device).uniform_(d_min, d_max)
                angle = torch.empty(count, batch_size, device=self.device).uniform_(-torch.pi, torch.pi)
                candidates = candidate_centers + torch.stack(
                    (dist * torch.cos(angle), dist * torch.sin(angle)), dim=2
                )
                flat_centers = candidate_centers.reshape(-1, 2)
                flat_candidates = candidates.reshape(-1, 2)
                valid = self._is_clear(flat_candidates)
                if self.cfg.use_scenario_profile and self.path_rock_map is not None:
                    types = scenario_types[need].repeat_interleave(batch_size)
                    valid &= self._classify_paths(flat_centers, flat_candidates) == types
                valid = valid.view(count, batch_size)
                found = valid.any(dim=1)
                choice = valid.float().argmax(dim=1)
                rows = torch.arange(count, device=self.device)
                matched = need[found]
                centers[matched] = candidate_centers[rows[found], choice[found]]
                targets_xy[matched] = candidates[rows[found], choice[found]]
                resolved[matched] = True

        sample_remaining()
        if self.cfg.use_scenario_profile and self.path_rock_map is not None:
            # Do not lose a complete training run because the map cannot provide a
            # small number of rare recovery configurations.  Tight falls back to a
            # normal detour first; any remaining obstacle case falls back to clear.
            for fallback_type in (1, 0):
                downgrade = (~resolved) & (scenario_types > fallback_type)
                if downgrade.any():
                    scenario_types[downgrade] = fallback_type
                    sample_remaining()
        if not torch.all(resolved):
            bad_ids = torch.as_tensor(env_ids, device=self.device)[~resolved].tolist()
            requested = requested_scenario_types[~resolved].tolist()
            resolved_types = scenario_types[~resolved].tolist()
            raise RuntimeError(
                f"Failed to sample navigation scenarios for envs {bad_ids}: "
                f"requested={requested}, after_fallback={resolved_types}. "
                "Relax the scenario weights or path-distance thresholds."
            )

        # 存世界坐标（z 用 default root height）
        self.pos_command_w[env_ids, :2] = targets_xy
        self.pos_command_w[env_ids, 2] = self.robot.data.default_root_state[env_ids, 2]
        self.target_distance[env_ids] = torch.norm(targets_xy - centers, dim=1)
        self._previous_error_pos[env_ids] = torch.nan
        h = torch.atan2(targets_xy[:, 1] - centers[:, 1], targets_xy[:, 0] - centers[:, 0])
        if self.cfg.use_scenario_profile:
            jitter = torch.empty(n, device=self.device).uniform_(
                -self.cfg.spawn_heading_jitter, self.cfg.spawn_heading_jitter
            )
            # Tight episodes are validated at this heading: keep their obstacle directly ahead.
            h += torch.where(scenario_types == 2, torch.zeros_like(jitter), jitter)
        if self.cfg.sample_spawn_and_goal:
            self._write_spawn_state(env_ids, centers, h)
        elif self.cfg.use_scenario_profile:
            self._write_spawn_heading(env_ids, h)
        self.heading_command_w[env_ids] = h
        self.metrics["scenario_type"][env_ids] = scenario_types.float()
        if self.cfg.sample_spawn_and_goal and not self._scenario_stats_printed:
            requested_counts = torch.bincount(requested_scenario_types, minlength=3).tolist()
            resolved_counts = torch.bincount(scenario_types, minlength=3).tolist()
            fallback_count = int((scenario_types != requested_scenario_types).sum().item())
            print(
                "[NavigationCommand] scenarios requested clear/wide/tight="
                f"{requested_counts}, resolved={resolved_counts}, fallbacks={fallback_count}"
            )
            self._scenario_stats_printed = True

    def _sample_spawn_positions(self, count: int) -> torch.Tensor:
        positions = torch.zeros(count, 2, device=self.device)
        resolved = torch.zeros(count, dtype=torch.bool, device=self.device)
        for _ in range(self.cfg.max_resample_attempts):
            need = torch.nonzero(~resolved, as_tuple=False).flatten()
            if len(need) == 0:
                break
            candidates = torch.empty(len(need), 2, device=self.device).uniform_(
                self.cfg.spawn_margin, self.terrain_size - self.cfg.spawn_margin
            )
            valid = self._is_spawn_clear(candidates)
            positions[need[valid]] = candidates[valid]
            resolved[need[valid]] = True
        if not torch.all(resolved):
            raise RuntimeError("Failed to sample clear rover spawn positions")
        return positions

    def _sample_scenario_types(self, count: int) -> torch.Tensor:
        if not self.cfg.use_scenario_profile or self.path_rock_map is None:
            return torch.zeros(count, dtype=torch.long, device=self.device)
        weights = torch.tensor(self.cfg.scenario_weights, dtype=torch.float32, device=self.device)
        expected = weights / weights.sum() * count
        counts = torch.floor(expected).long()
        remainder = count - int(counts.sum())
        if remainder:
            extra = torch.topk(expected - counts, remainder).indices
            counts[extra] += 1
        scenario = torch.repeat_interleave(torch.arange(3, device=self.device), counts)
        return scenario[torch.randperm(count, device=self.device)]

    def _is_spawn_clear(self, xy: torch.Tensor) -> torch.Tensor:
        margin = self.cfg.spawn_margin
        inside = (
            (xy[:, 0] >= margin)
            & (xy[:, 0] <= self.terrain_size - margin)
            & (xy[:, 1] >= margin)
            & (xy[:, 1] <= self.terrain_size - margin)
        )
        if self.spawn_rock_map is not None:
            col = (xy[:, 0] / self.rock_mpp).long().clamp(0, self.rock_w - 1)
            row = (xy[:, 1] / self.rock_mpp).long().clamp(0, self.rock_h - 1)
            inside &= self.spawn_rock_map[row, col] < 0.5
        return inside & self._soil_is_firm(xy)

    def _classify_paths(self, starts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return 0=clear, 1=wide detour, 2=reverse-required detour."""
        direction = targets - starts
        distance = torch.linalg.vector_norm(direction, dim=1).clamp_min(1.0e-6)
        heading = direction / distance.unsqueeze(1)
        sample_distance = torch.arange(
            0.0, self.cfg.target_distance_range[1] + self.cfg.path_sample_resolution,
            self.cfg.path_sample_resolution, device=self.device,
        )
        points = starts[:, None, :] + heading[:, None, :] * sample_distance[None, :, None]
        col = (points[..., 0] / self.rock_mpp).long().clamp(0, self.rock_w - 1)
        row = (points[..., 1] / self.rock_mpp).long().clamp(0, self.rock_h - 1)
        valid = sample_distance[None, :] <= distance[:, None]
        occupied = (self.path_rock_map[row, col] > 0.5) & valid
        has_hit = occupied.any(dim=1)
        first_index = occupied.float().argmax(dim=1)
        first_distance = sample_distance[first_index]
        scenario = torch.zeros(len(starts), dtype=torch.long, device=self.device)
        wide = has_hit & (first_distance >= self.cfg.wide_block_distance)
        tight = has_hit & (first_distance >= self.cfg.tight_block_min_distance) & (
            first_distance < self.cfg.tight_block_max_distance
        )
        heading_angle = torch.atan2(direction[:, 1], direction[:, 0])
        curvatures = torch.as_tensor(self.cfg.tight_test_curvatures, device=self.device)
        forward_heading = heading_angle[:, None].expand(-1, len(curvatures))
        forward_clear = self._paths_are_clear(
            starts, forward_heading, curvatures, self.cfg.tight_forward_distance
        )
        reverse_heading = heading_angle[:, None] + torch.pi
        reverse_clear = self._paths_are_clear(
            starts, reverse_heading, -curvatures, self.cfg.tight_reverse_distance
        )
        # Only validate the expensive recovery route for candidates that already need reverse.
        reverse_required = tight & ~forward_clear.any(dim=1) & reverse_clear.any(dim=1)
        recovery_feasible = torch.zeros_like(reverse_required)
        if reverse_required.any():
            ids = torch.nonzero(reverse_required, as_tuple=False).flatten()
            recovery_feasible[ids] = self._tight_recovery_is_feasible(
                starts[ids], targets[ids], heading_angle[ids], curvatures, reverse_clear[ids]
            )
        tight &= recovery_feasible
        scenario[wide] = 1
        scenario[tight] = 2
        scenario[has_hit & ~wide & ~tight] = -1
        return scenario

    def _tight_recovery_is_feasible(
        self,
        starts: torch.Tensor,
        targets: torch.Tensor,
        heading: torch.Tensor,
        curvatures: torch.Tensor,
        reverse_clear: torch.Tensor,
    ) -> torch.Tensor:
        """Check a reverse arc, a forward arc, then a clear line to the goal."""
        feasible = torch.zeros(len(starts), dtype=torch.bool, device=self.device)
        initial_distance = torch.linalg.vector_norm(targets - starts, dim=1)
        reverse_travel_heading = heading + torch.pi

        for reverse_index, reverse_curvature in enumerate(curvatures):
            reverse_curve = torch.full_like(heading, -reverse_curvature)
            reverse_end, reverse_travel_end = self._arc_endpoint(
                starts, reverse_travel_heading, reverse_curve, self.cfg.tight_recovery_reverse_distance
            )
            reverse_body_heading = wrap_to_pi(reverse_travel_end - torch.pi)

            for forward_curvature in curvatures:
                forward_curve = torch.full_like(heading, forward_curvature)
                forward_clear = self._paths_are_clear(
                    reverse_end,
                    reverse_body_heading[:, None],
                    forward_curvature.reshape(1),
                    self.cfg.tight_recovery_forward_distance,
                )[:, 0]
                forward_end, _ = self._arc_endpoint(
                    reverse_end, reverse_body_heading, forward_curve, self.cfg.tight_recovery_forward_distance
                )
                closer_to_goal = torch.linalg.vector_norm(targets - forward_end, dim=1) < initial_distance
                feasible |= (
                    reverse_clear[:, reverse_index]
                    & forward_clear
                    & closer_to_goal
                    & self._line_is_clear(forward_end, targets)
                )
        return feasible

    def _arc_endpoint(
        self, starts: torch.Tensor, headings: torch.Tensor, curvatures: torch.Tensor, length: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the endpoint and tangent heading of one constant-curvature arc."""
        arc_angle = curvatures * length
        straight = curvatures.abs() < 1.0e-6
        safe_curvature = torch.where(straight, torch.ones_like(curvatures), curvatures)
        local_x = torch.where(straight, torch.full_like(curvatures, length), torch.sin(arc_angle) / safe_curvature)
        local_y = torch.where(straight, torch.zeros_like(curvatures), (1.0 - torch.cos(arc_angle)) / safe_curvature)
        cos_heading = torch.cos(headings)
        sin_heading = torch.sin(headings)
        endpoints = starts + torch.stack(
            (cos_heading * local_x - sin_heading * local_y, sin_heading * local_x + cos_heading * local_y),
            dim=1,
        )
        return endpoints, wrap_to_pi(headings + arc_angle)

    def _line_is_clear(self, starts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Check a straight, inflated-RockGT path with the configured sampling resolution."""
        direction = targets - starts
        distance = torch.linalg.vector_norm(direction, dim=1).clamp_min(1.0e-6)
        sample_distance = torch.arange(
            0.0,
            self.terrain_size * math.sqrt(2.0) + self.cfg.path_sample_resolution,
            self.cfg.path_sample_resolution,
            device=self.device,
        )
        points = starts[:, None, :] + (
            direction / distance[:, None]
        )[:, None, :] * sample_distance[None, :, None]
        inside = (
            (points[..., 0] >= 0.0)
            & (points[..., 0] <= self.terrain_size)
            & (points[..., 1] >= 0.0)
            & (points[..., 1] <= self.terrain_size)
        )
        col = (points[..., 0] / self.rock_mpp).long().clamp(0, self.rock_w - 1)
        row = (points[..., 1] / self.rock_mpp).long().clamp(0, self.rock_h - 1)
        valid = sample_distance[None, :] <= distance[:, None]
        occupied = (self.path_rock_map[row, col] > 0.5) | ~inside
        return ~(occupied & valid).any(dim=1)

    def _paths_are_clear(
        self, starts: torch.Tensor, headings: torch.Tensor, curvatures: torch.Tensor, length: float
    ) -> torch.Tensor:
        """Check constant-curvature paths against the inflated RockGT map."""
        sample_distance = torch.arange(
            0.0, length + self.cfg.path_sample_resolution,
            self.cfg.path_sample_resolution, device=self.device,
        )
        curvature = curvatures[None, :, None]
        arc_angle = curvature * sample_distance[None, None]
        straight = curvature.abs() < 1.0e-6
        local_x = torch.where(
            straight, sample_distance[None, None].expand_as(arc_angle), torch.sin(arc_angle) / curvature
        )
        local_y = torch.where(straight, torch.zeros_like(arc_angle), (1.0 - torch.cos(arc_angle)) / curvature)
        cos_heading = torch.cos(headings)[..., None]
        sin_heading = torch.sin(headings)[..., None]
        x = starts[:, None, None, 0] + cos_heading * local_x - sin_heading * local_y
        y = starts[:, None, None, 1] + sin_heading * local_x + cos_heading * local_y
        inside = (x >= 0.0) & (x <= self.terrain_size) & (y >= 0.0) & (y <= self.terrain_size)
        col = (x / self.rock_mpp).long().clamp(0, self.rock_w - 1)
        row = (y / self.rock_mpp).long().clamp(0, self.rock_h - 1)
        occupied = (self.path_rock_map[row, col] > 0.5) | ~inside
        return ~occupied.any(dim=2)

    def _write_spawn_heading(self, env_ids: Sequence[int], heading: torch.Tensor) -> None:
        spawn_pose = getattr(self._env, "_rover_spawn_pose", None)
        if spawn_pose is None:
            return
        pose = spawn_pose[env_ids].clone()
        pose[:, 3] = torch.cos(heading * 0.5)
        pose[:, 4:6] = 0.0
        pose[:, 6] = torch.sin(heading * 0.5)
        self.robot.write_root_pose_to_sim(pose, env_ids=env_ids)

    def _write_spawn_state(self, env_ids: Sequence[int], positions: torch.Tensor, heading: torch.Tensor) -> None:
        # Positions are map/world coordinates, intentionally not scene.env_origins.
        # Physics replication and collision filtering keep these overlapping rollouts independent.
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :2] = positions
        root_state[:, 2] = _bilinear_sample(self.height_map, positions, self.terrain_size) + self.cfg.spawn_z_offset
        root_state[:, 3] = torch.cos(heading * 0.5)
        root_state[:, 4:6] = 0.0
        root_state[:, 6] = torch.sin(heading * 0.5)
        root_state[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids],
            torch.zeros_like(self.robot.data.default_joint_vel[env_ids]),
            env_ids=env_ids,
        )
        if not hasattr(self._env, "_rover_spawn_xy"):
            self._env._rover_spawn_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._env._rover_spawn_xy[env_ids] = positions

    def _is_clear(self, xy: torch.Tensor) -> torch.Tensor:
        """检查 xy 点 ±safety_margin 范围内 rock_map 全为 0。返回 (n,) bool。"""
        margin = self.cfg.map_boundary_margin
        inside = (
            (xy[:, 0] >= margin)
            & (xy[:, 0] <= self.terrain_size - margin)
            & (xy[:, 1] >= margin)
            & (xy[:, 1] <= self.terrain_size - margin)
        )
        if self.target_rock_map is not None:
            col = (xy[:, 0] / self.rock_mpp).long().clamp(0, self.rock_w - 1)
            row = (xy[:, 1] / self.rock_mpp).long().clamp(0, self.rock_h - 1)
            inside &= self.target_rock_map[row, col] < 0.5
        return inside & self._soil_is_firm(xy)

    def _soil_is_firm(self, xy: torch.Tensor) -> torch.Tensor:
        """点是否落在可通行的土层上（未配置 max_soil_class 时恒为 True）。"""
        if self.soft_soil_map is None:
            return torch.ones(len(xy), dtype=torch.bool, device=self.device)
        col = (xy[:, 0] / self.soil_mpp).long().clamp(0, self.soft_soil_map.shape[1] - 1)
        row = (xy[:, 1] / self.soil_mpp).long().clamp(0, self.soft_soil_map.shape[0] - 1)
        return self.soft_soil_map[row, col] < 0.5

    def _update_command(self):
        """每步把世界目标转车体坐标。"""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.pos_command_b[:] = quat_apply_inverse(
            yaw_quat(self.robot.data.root_quat_w), target_vec)
        self.heading_command_b[:] = wrap_to_pi(
            self.heading_command_w - self.robot.data.heading_w)

    def _update_metrics(self):
        pos_err = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        self.metrics["progress"] = torch.where(
            torch.isfinite(self._previous_error_pos), self._previous_error_pos - pos_err, torch.zeros_like(pos_err)
        )
        self._previous_error_pos[:] = pos_err
        self.metrics["error_pos"] = pos_err
        self.metrics["error_heading"] = torch.abs(
            wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))


@configclass
class NavigationCommandCfg(CommandTermCfg):
    class_type: type[CommandTerm] = NavigationCommand
    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (150.0, 150.0)   # 全 episode 不重采

    terrain_dir: str = ""
    terrain_size: float = 100.0
    target_distance_range: tuple[float, float] = (3.0, 15.0)      # 目标采样距离范围（米）
    rock_safety_margin: float = 1.0                                # 岩石安全距离（米）
    map_boundary_margin: float = 3.0                               # 地图边缘安全距离（米）
    max_resample_attempts: int = 400
    candidate_batch_size: int = 16
    use_scenario_profile: bool = False
    scenario_weights: tuple[float, float, float] = (1.0, 0.0, 0.0)
    path_clearance: float = 0.85
    path_sample_resolution: float = 0.2
    wide_block_distance: float = 2.2
    tight_block_min_distance: float = 0.5
    tight_block_max_distance: float = 2.2
    tight_test_curvatures: tuple[float, ...] = (-0.55, -0.275, 0.0, 0.275, 0.55)
    tight_forward_distance: float = 2.0
    tight_reverse_distance: float = 2.0
    tight_recovery_reverse_distance: float = 1.0
    tight_recovery_forward_distance: float = 3.0
    spawn_heading_jitter: float = 0.17
    sample_spawn_and_goal: bool = False
    spawn_clearance: float = 1.2
    spawn_margin: float = 3.0
    spawn_z_offset: float = 0.5
    # terrain_class_map 里大于该类别的土视为不可通行，出生点和目标点都不落在上面。
    # None = 不做土壤筛选（硬地任务）。
    max_soil_class: int | None = None
    soil_safety_margin: float = 1.5
    # ((spawn_x, spawn_y), (goal_x, goal_y))：钉死场景、跳过全部采样。
    # 用于把某个已选定的场景用 num_envs=1 单车复现录像（多车共用同一地形，
    # 跟随相机会拍到旁车）。仅用于可视化，训练时必须为 None。
    fixed_scenario: tuple[tuple[float, float], tuple[float, float]] | None = None

    @configclass
    class Ranges:
        heading: tuple[float, float] = (-torch.pi, torch.pi)

    ranges: Ranges = Ranges()
