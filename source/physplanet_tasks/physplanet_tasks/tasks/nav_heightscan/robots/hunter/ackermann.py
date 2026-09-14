"""Hunter fixed-speed 4WD/2WS action."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class HunterAckermannAction(ActionTerm):
    cfg: "HunterAckermannActionCfg"

    def __init__(self, cfg: "HunterAckermannActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        names = {name: index for index, name in enumerate(self._asset.joint_names)}
        self._left = [names[name] for name in ("re_left_joint", "fr_left_joint")]
        self._right = [names[name] for name in ("re_right_joint", "fr_right_joint")]
        self._steer = [names[name] for name in ("fr_steer_left_joint", "fr_steer_right_joint")]
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._steer_targets = torch.zeros(self.num_envs, 2, device=self.device)

    @property
    def action_dim(self): return 1
    @property
    def raw_actions(self): return self._raw_actions
    @property
    def processed_actions(self): return self._processed_actions

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions[:, 0] = self.cfg.linear_velocity
        self._processed_actions[:, 1] = actions[:, 0] * self.cfg.angular_velocity_scale

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = 0.0
        self._steer_targets[ids] = 0.0

    def apply_actions(self):
        v, w = self._processed_actions[:, :1], self._processed_actions[:, 1:]
        wheel_left = (v - w * self.cfg.track_width / 2) / self.cfg.wheel_radius
        wheel_right = (v + w * self.cfg.track_width / 2) / self.cfg.wheel_radius
        steer = torch.atan(w * self.cfg.wheel_base / v).clamp(-self.cfg.max_steer_angle, self.cfg.max_steer_angle)
        tangent = torch.tan(steer)
        left = torch.where(steer.abs() > 1e-3, torch.atan(self.cfg.wheel_base / (self.cfg.wheel_base / tangent - self.cfg.track_width / 2)), steer)
        right = torch.where(steer.abs() > 1e-3, torch.atan(self.cfg.wheel_base / (self.cfg.wheel_base / tangent + self.cfg.track_width / 2)), steer)
        desired = torch.cat((left, right), dim=1)
        delta = self.cfg.max_steer_rate * self._env.physics_dt
        self._steer_targets += (desired - self._steer_targets).clamp(-delta, delta)
        pos, vel = torch.zeros_like(self._asset.data.joint_pos), torch.zeros_like(self._asset.data.joint_vel)
        pos[:, self._steer] = self._steer_targets
        vel[:, self._left] = wheel_left
        vel[:, self._right] = wheel_right
        self._asset.set_joint_position_target(pos, joint_ids=slice(None))
        self._asset.set_joint_velocity_target(vel, joint_ids=slice(None))


@configclass
class HunterAckermannActionCfg(ActionTermCfg):
    class_type = HunterAckermannAction
    asset_name: str = "robot"
    linear_velocity: float = 0.35
    angular_velocity_scale: float = 0.25
    max_steer_rate: float = 1.0
    wheel_base: float = 0.5484
    track_width: float = 0.4924
    wheel_radius: float = 0.135
    max_steer_angle: float = 0.4363
