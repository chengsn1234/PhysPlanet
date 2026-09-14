# Hunter 控制
####################
## 功能：cmd_vel / wheel_omega -> Hunter joint target
## 包含：车型几何、joint 解析、Ackermann 计算、ROS Controller
####################

from __future__ import annotations

import torch

from rover_controller import RoverController


WHEEL_BASE = 0.5484
TRACK_WIDTH = 0.4924
MAX_STEER_ANGLE = 0.4363
WHEEL_RADIUS = 0.135
WHEEL_WIDTH = 0.08
WHEEL_LUG_H = 0.0

WHEEL_ORDER = ("re_left", "fr_left", "re_right", "fr_right")
WHEEL_BODY_NAMES = ("re_left_link", "fr_left_link", "re_right_link", "fr_right_link")
LEFT_DRIVE_JOINT_NAMES = ("re_left_joint", "fr_left_joint")
RIGHT_DRIVE_JOINT_NAMES = ("re_right_joint", "fr_right_joint")
STEER_LEFT_JOINT_NAME = "fr_steer_left_joint"
STEER_RIGHT_JOINT_NAME = "fr_steer_right_joint"


####################
## Joint 解析
####################

def resolve_hunter_joint_map(joint_names: list[str]) -> dict[str, list[int] | int]:
    joint_name_to_index = {name: i for i, name in enumerate(joint_names)}
    return {
        "left_drive_indices": [joint_name_to_index[name] for name in LEFT_DRIVE_JOINT_NAMES],
        "right_drive_indices": [joint_name_to_index[name] for name in RIGHT_DRIVE_JOINT_NAMES],
        "steer_left_idx": joint_name_to_index[STEER_LEFT_JOINT_NAME],
        "steer_right_idx": joint_name_to_index[STEER_RIGHT_JOINT_NAME],
    }


def _as_column_tensor(values, device: str | torch.device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.float32).reshape(-1, 1)
    return torch.tensor(values, device=device, dtype=torch.float32).reshape(-1, 1)


####################
## Ackermann 运动学
####################

def compute_ackermann_targets(linear_x, angular_z, device: str | torch.device) -> dict[str, torch.Tensor]:
    forward_speed = _as_column_tensor(linear_x, device)
    angular_speed = _as_column_tensor(angular_z, device)

    wheel_vel_left = (forward_speed - angular_speed * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
    wheel_vel_right = (forward_speed + angular_speed * TRACK_WIDTH / 2.0) / WHEEL_RADIUS

    steer_angle = torch.where(
        forward_speed.abs() > 0.1,
        torch.atan(angular_speed * WHEEL_BASE / forward_speed),
        angular_speed * 0.5,
    )
    steer_angle = torch.clamp(steer_angle, -MAX_STEER_ANGLE, MAX_STEER_ANGLE)

    tan_steer = torch.tan(steer_angle)
    steer_left = torch.where(
        steer_angle.abs() > 0.01,
        torch.atan(WHEEL_BASE / (WHEEL_BASE / tan_steer - TRACK_WIDTH / 2.0)),
        steer_angle,
    )
    steer_right = torch.where(
        steer_angle.abs() > 0.01,
        torch.atan(WHEEL_BASE / (WHEEL_BASE / tan_steer + TRACK_WIDTH / 2.0)),
        steer_angle,
    )

    return {
        "forward_speed": forward_speed,
        "angular_speed": angular_speed,
        "wheel_vel_left": wheel_vel_left,
        "wheel_vel_right": wheel_vel_right,
        "steer_left": steer_left,
        "steer_right": steer_right,
    }


def build_hunter_joint_targets(
    joint_pos_template: torch.Tensor,
    joint_vel_template: torch.Tensor,
    joint_map: dict[str, list[int] | int],
    linear_x,
    angular_z,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    targets = compute_ackermann_targets(linear_x, angular_z, device=device)
    joint_vel_target = torch.zeros_like(joint_vel_template)
    joint_pos_target = torch.zeros_like(joint_pos_template)

    for idx in joint_map["left_drive_indices"]:
        joint_vel_target[:, idx] = targets["wheel_vel_left"].squeeze(1)
    for idx in joint_map["right_drive_indices"]:
        joint_vel_target[:, idx] = targets["wheel_vel_right"].squeeze(1)

    joint_pos_target[:, joint_map["steer_left_idx"]] = targets["steer_left"].squeeze(1)
    joint_pos_target[:, joint_map["steer_right_idx"]] = targets["steer_right"].squeeze(1)
    return joint_pos_target, joint_vel_target, targets


####################
## ROS Controller
####################

class HunterController(RoverController):
    def __init__(self, num_envs=1, robot=None, device="cuda:0"):
        super().__init__("hunter_controller", robot, num_envs, device, wheel_count=4)
        if robot is not None:
            self._init_joint_map(robot)

    def _init_joint_map(self, robot):
        self.joint_map = resolve_hunter_joint_map(robot.joint_names)

    def compute_control_batch(self):
        if self.robot is None:
            raise RuntimeError("Robot not initialized.")

        joint_pos_target, joint_vel_target, _ = build_hunter_joint_targets(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.joint_map,
            self.linear_x,
            self.angular_z,
            device=self.device,
        )

        jm = self.joint_map
        for env_id in range(self.num_envs):
            override = self._wheel_omega_override[env_id]
            if override is None:
                continue
            # wheel_omega_cmd 覆写：按 [re_left, fr_left, re_right, fr_right] 直填轮速。
            left_indices = jm["left_drive_indices"]
            right_indices = jm["right_drive_indices"]
            if len(override) >= 4:
                joint_vel_target[env_id, left_indices[0]] = override[0]
                joint_vel_target[env_id, left_indices[1]] = override[1]
                joint_vel_target[env_id, right_indices[0]] = override[2]
                joint_vel_target[env_id, right_indices[1]] = override[3]
            joint_pos_target[env_id, jm["steer_left_idx"]] = 0.0
            joint_pos_target[env_id, jm["steer_right_idx"]] = 0.0

        return joint_pos_target, joint_vel_target
