"""祝融号 Ackermann action terms（纯 torch，不依赖 rclpy/ZhurongController）。

提供固定速度、速度-曲率和带方向承诺的速度-曲率控制。运动学从
physplanet_assets.utils.zhurong.zhurong_control 移植，去掉 rover_controller 依赖。
"""

from __future__ import annotations

import torch
from typing import Sequence

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

# ── 祝融号几何常量（同 zhurong_control.py）──
WHEEL_RADIUS = 0.15
WHEEL_BASE_H = 0.652      # 半轮距（横向）
WHEEL_BASE_L = 0.775      # 半轴距（纵向）
MAX_STEER_ANGLE = 0.6     # rad

WHEEL_ORDER = ("FL", "FR", "ML", "MR", "BL", "BR")


# ── Joint 解析（内联，同 resolve_zhurong_joint_groups）──

def _resolve_joint_groups(joint_names: list[str]) -> dict[str, dict[str, int]]:
    idx = {name: i for i, name in enumerate(joint_names)}
    return {
        "wheel": {
            "FL": idx["front_wheel_L_joint"], "FR": idx["front_wheel_R_joint"],
            "ML": idx["middle_wheel_L_joint"], "MR": idx["middle_wheel_R_joint"],
            "BL": idx["back_wheel_L_joint"], "BR": idx["back_wheel_R_joint"],
        },
        "steer": {
            "FL": idx["suspension_steer_F_L_joint"], "FR": idx["suspension_steer_F_R_joint"],
            "ML": idx["suspension_steer_M_L_joint"], "MR": idx["suspension_steer_M_R_joint"],
            "BL": idx["suspension_steer_B_L_joint"], "BR": idx["suspension_steer_B_R_joint"],
        },
        "arm": {
            "FL": idx["suspension_arm_F_L_joint"], "FR": idx["suspension_arm_F_R_joint"],
            "BL": idx["suspension_arm_B_L_joint"], "BR": idx["suspension_arm_B_R_joint"],
        },
        "ptz": {
            "yaw": idx["PTZYaw_joint"],
            **({"pitch": idx["PTZPitch_joint"]} if "PTZPitch_joint" in idx else {}),
        } if "PTZYaw_joint" in idx else {},
    }


# ── 6WD/6WS 逆运动学（内联，同 compute_zhurong_joint_targets）──

def _compute_joint_targets(
    v: torch.Tensor,          # (E, 1) 线速度 m/s
    w: torch.Tensor,          # (E, 1) 角速度 rad/s
    num_joints: int,
    wheel_idx: dict[str, int],
    steer_idx: dict[str, int],
    arm_idx: dict[str, int],
    ptz_idx: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (joint_pos_target, joint_vel_target)，shape (E, num_joints)。"""
    joint_pos = torch.zeros((v.shape[0], num_joints), device=device)
    joint_vel = torch.zeros((v.shape[0], num_joints), device=device)

    straight = w.abs() < 1.0e-3
    spin = v.abs() < 1.0e-3
    turning_radius = v / (w + 1.0e-10)

    wheel_vel_straight = v / WHEEL_RADIUS
    r_lf = ((turning_radius - WHEEL_BASE_H).abs() ** 2 + WHEEL_BASE_L ** 2).sqrt()
    r_rf = ((turning_radius + WHEEL_BASE_H).abs() ** 2 + WHEEL_BASE_L ** 2).sqrt()
    r_lm = (turning_radius - WHEEL_BASE_H).abs()
    r_rm = (turning_radius + WHEEL_BASE_H).abs()

    vel_lf = w.abs() * r_lf / WHEEL_RADIUS
    vel_rf = w.abs() * r_rf / WHEEL_RADIUS
    vel_lm = w.abs() * r_lm / WHEEL_RADIUS
    vel_rm = w.abs() * r_rm / WHEEL_RADIUS
    vel_arr = torch.cat([vel_lf, vel_rf, vel_lm, vel_rm, vel_lf, vel_rf], dim=1)

    vel_arr = torch.where(v > 0, vel_arr, -vel_arr)
    vel_arr = torch.where(spin, -w.sign() * vel_arr, vel_arr)

    theta_lf = torch.atan(WHEEL_BASE_L / (turning_radius - WHEEL_BASE_H + 1.0e-10))
    theta_rf = torch.atan(WHEEL_BASE_L / (turning_radius + WHEEL_BASE_H + 1.0e-10))
    zero = torch.zeros_like(theta_lf)
    theta_arr = torch.cat([
        torch.clamp(theta_lf, -MAX_STEER_ANGLE, MAX_STEER_ANGLE),
        torch.clamp(theta_rf, -MAX_STEER_ANGLE, MAX_STEER_ANGLE),
        zero, zero,
        torch.clamp(-theta_lf, -MAX_STEER_ANGLE, MAX_STEER_ANGLE),
        torch.clamp(-theta_rf, -MAX_STEER_ANGLE, MAX_STEER_ANGLE),
    ], dim=1)

    # 小半径（轮距内）原地转向：左右轮反向
    small_left = (turning_radius >= 0) & (turning_radius < WHEEL_BASE_H) & ~straight
    flip_l = torch.tensor([[-1, 1, -1, 1, -1, 1]], device=device, dtype=vel_arr.dtype)
    vel_arr = torch.where(small_left, vel_arr * flip_l, vel_arr)
    small_right = (turning_radius < 0) & (turning_radius > -WHEEL_BASE_H) & ~straight
    flip_r = torch.tensor([[1, -1, 1, -1, 1, -1]], device=device, dtype=vel_arr.dtype)
    vel_arr = torch.where(small_right, vel_arr * flip_r, vel_arr)

    vel_arr = torch.where(straight, wheel_vel_straight.expand(-1, 6), vel_arr)
    theta_arr = torch.where(straight, torch.zeros_like(theta_arr), theta_arr)

    for i, name in enumerate(WHEEL_ORDER):
        joint_vel[:, wheel_idx[name]] = vel_arr[:, i]
        joint_pos[:, steer_idx[name]] = theta_arr[:, i]

    for aid in arm_idx.values():
        joint_pos[:, aid] = 0.0
    if "yaw" in ptz_idx:
        joint_pos[:, ptz_idx["yaw"]] = 0.0
    if "pitch" in ptz_idx:
        joint_pos[:, ptz_idx["pitch"]] = 0.0

    return joint_pos, joint_vel


# ── ActionTerm 实现 ──

class ZhurongAckermannAction(ActionTerm):
    """1D action [omega] → 固定速度的祝融号 6WD/6WS joint targets。"""

    cfg: "ZhurongAckermannActionCfg"

    def __init__(self, cfg: "ZhurongAckermannActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        groups = _resolve_joint_groups(self._asset.joint_names)
        self._wheel_idx = groups["wheel"]
        self._steer_idx = groups["steer"]
        self._arm_idx = groups["arm"]
        self._ptz_idx = groups["ptz"]
        self._num_joints = self._asset.num_joints

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._steer_ids = torch.tensor(
            [self._steer_idx[name] for name in WHEEL_ORDER], device=self.device
        )
        self._steer_targets = torch.zeros(self.num_envs, len(WHEEL_ORDER), device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:, 0] = self.cfg.linear_velocity
        self._processed_actions[:, 1] = self._raw_actions[:, 0] * self.cfg.angular_velocity_scale

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._steer_targets.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._steer_targets[env_ids] = 0.0

    def apply_actions(self) -> None:
        v = self._processed_actions[:, 0:1]   # (E, 1)
        w = self._processed_actions[:, 1:2]   # (E, 1)
        pos_target, vel_target = _compute_joint_targets(
            v, w, self._num_joints,
            self._wheel_idx, self._steer_idx, self._arm_idx, self._ptz_idx, self.device)
        max_delta = self.cfg.max_steer_rate * self._env.physics_dt
        desired_steer = pos_target[:, self._steer_ids]
        self._steer_targets += (desired_steer - self._steer_targets).clamp(-max_delta, max_delta)
        pos_target[:, self._steer_ids] = self._steer_targets
        self._asset.set_joint_position_target(pos_target, joint_ids=slice(None))
        self._asset.set_joint_velocity_target(vel_target, joint_ids=slice(None))


@configclass
class ZhurongAckermannActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = ZhurongAckermannAction
    asset_name: str = "robot"
    linear_velocity: float = 0.35
    # Matches the USD's +/-0.6 rad steering limit at 0.35 m/s.
    angular_velocity_scale: float = 0.19
    max_steer_rate: float = 1.0


# ── V2：速度 + 曲率控制 ──

class ZhurongSpeedCurvatureAction(ZhurongAckermannAction):
    """2D action [speed, curvature] → 平滑的祝融号 6WD/6WS joint targets。

    曲率定义为 ``omega / v``，所以静止时不会产生原地自转。速度动作的正半轴
    对应前进、负半轴对应倒车；限速在控制周期内执行，避免策略切换时突变。
    """

    cfg: "ZhurongSpeedCurvatureActionCfg"

    def __init__(self, cfg: "ZhurongSpeedCurvatureActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._speed_command = torch.zeros(self.num_envs, device=self.device)

    @property
    def action_dim(self) -> int:
        return 2

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        speed_action = actions[:, 0].clamp(-1.0, 1.0)
        curvature_action = actions[:, 1].clamp(-1.0, 1.0)

        desired_speed = torch.where(
            speed_action >= 0.0,
            speed_action * self.cfg.max_forward_velocity,
            speed_action * self.cfg.max_reverse_velocity,
        )
        desired_speed = torch.where(
            speed_action.abs() < self.cfg.stop_deadband,
            torch.zeros_like(desired_speed),
            desired_speed,
        )
        accelerating = (
            (desired_speed * self._speed_command >= 0.0)
            & (desired_speed.abs() > self._speed_command.abs())
        )
        rate_limit = torch.where(
            accelerating,
            torch.full_like(desired_speed, self.cfg.acceleration_limit),
            torch.full_like(desired_speed, self.cfg.brake_acceleration_limit),
        )
        max_delta = rate_limit * self._env.step_dt
        self._speed_command += (desired_speed - self._speed_command).clamp(-max_delta, max_delta)

        curvature = curvature_action * self.cfg.curvature_scale
        self._processed_actions[:, 0] = self._speed_command
        self._processed_actions[:, 1] = self._speed_command * curvature

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._speed_command.zero_()
        else:
            self._speed_command[env_ids] = 0.0


@configclass
class ZhurongSpeedCurvatureActionCfg(ActionTermCfg):
    """Configuration for the V2 speed-curvature controller."""

    class_type: type[ActionTerm] = ZhurongSpeedCurvatureAction
    asset_name: str = "robot"
    max_forward_velocity: float = 0.4
    max_reverse_velocity: float = 0.2
    curvature_scale: float = 0.55
    stop_deadband: float = 0.05
    acceleration_limit: float = 0.5
    brake_acceleration_limit: float = 0.8
    max_steer_rate: float = 1.0


# ── V3：带方向承诺的速度 + 曲率控制 ──

class ZhurongCommittedSpeedCurvatureAction(ZhurongSpeedCurvatureAction):
    """速度换向必须先刹停，随后短暂保持新方向，避免逐控制步前后抖动。"""

    cfg: "ZhurongCommittedSpeedCurvatureActionCfg"

    def __init__(self, cfg: "ZhurongCommittedSpeedCurvatureActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        if cfg.switch_hold_steps < 0:
            raise ValueError("switch_hold_steps must be non-negative")
        self._drive_direction = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)
        self._hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._held_speed = torch.zeros(self.num_envs, device=self.device)
        self._held_curvature = torch.zeros(self.num_envs, device=self.device)
        self._direction_switch = torch.zeros(self.num_envs, device=self.device)

    @property
    def drive_direction(self) -> torch.Tensor:
        """Current committed direction: -1 reverse, 0 idle, +1 forward."""
        return self._drive_direction

    @property
    def hold_steps(self) -> torch.Tensor:
        """Remaining control steps for the committed command."""
        return self._hold_steps

    @property
    def direction_switch(self) -> torch.Tensor:
        """One for the control step that commits an actual forward/reverse switch."""
        return self._direction_switch

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._direction_switch.zero_()
        speed_action = actions[:, 0].clamp(-1.0, 1.0)
        curvature_action = actions[:, 1].clamp(-1.0, 1.0)
        desired_speed = torch.where(
            speed_action >= 0.0,
            speed_action * self.cfg.max_forward_velocity,
            speed_action * self.cfg.max_reverse_velocity,
        )
        desired_speed = torch.where(
            speed_action.abs() < self.cfg.stop_deadband,
            torch.zeros_like(desired_speed),
            desired_speed,
        )
        desired_curvature = curvature_action * self.cfg.curvature_scale
        requested_direction = desired_speed.sign().to(torch.int8)

        command_speed = desired_speed.clone()
        command_curvature = desired_curvature.clone()
        holding = self._hold_steps > 0
        command_speed[holding] = self._held_speed[holding]
        command_curvature[holding] = self._held_curvature[holding]
        self._hold_steps[holding] -= 1

        available = ~holding
        starting = available & (self._drive_direction == 0) & (requested_direction != 0)
        self._drive_direction[starting] = requested_direction[starting]

        reversing = (
            available
            & (requested_direction != 0)
            & (self._drive_direction != 0)
            & (requested_direction != self._drive_direction)
        )
        braking = reversing & (self._speed_command.abs() > self.cfg.stop_speed_threshold)
        command_speed[braking] = 0.0
        command_curvature[braking] = 0.0

        commit = reversing & ~braking
        self._drive_direction[commit] = requested_direction[commit]
        self._direction_switch[commit] = 1.0
        self._held_speed[commit] = desired_speed[commit]
        self._held_curvature[commit] = desired_curvature[commit]
        command_speed[commit] = self._held_speed[commit]
        command_curvature[commit] = self._held_curvature[commit]
        self._hold_steps[commit] = max(self.cfg.switch_hold_steps - 1, 0)

        accelerating = (
            (command_speed * self._speed_command >= 0.0)
            & (command_speed.abs() > self._speed_command.abs())
        )
        rate_limit = torch.where(
            accelerating,
            torch.full_like(command_speed, self.cfg.acceleration_limit),
            torch.full_like(command_speed, self.cfg.brake_acceleration_limit),
        )
        max_delta = rate_limit * self._env.step_dt
        self._speed_command += (command_speed - self._speed_command).clamp(-max_delta, max_delta)
        self._processed_actions[:, 0] = self._speed_command
        self._processed_actions[:, 1] = self._speed_command * command_curvature

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._drive_direction.zero_()
            self._hold_steps.zero_()
            self._held_speed.zero_()
            self._held_curvature.zero_()
            self._direction_switch.zero_()
        else:
            self._drive_direction[env_ids] = 0
            self._hold_steps[env_ids] = 0
            self._held_speed[env_ids] = 0.0
            self._held_curvature[env_ids] = 0.0
            self._direction_switch[env_ids] = 0.0


@configclass
class ZhurongCommittedSpeedCurvatureActionCfg(ZhurongSpeedCurvatureActionCfg):
    """Configuration for the V3 committed speed-curvature controller."""

    class_type: type[ActionTerm] = ZhurongCommittedSpeedCurvatureAction
    stop_speed_threshold: float = 0.03
    switch_hold_steps: int = 4
