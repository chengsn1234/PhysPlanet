# 祝融号控制
####################
## 功能：cmd_vel / wheel_omega -> 祝融号 joint target
## 包含：车型几何、joint 解析、6WD/6WS 计算、ROS Controller
####################

from __future__ import annotations

import numpy as np
import torch

from rover_controller import RoverController


WHEEL_RADIUS = 0.15
WHEEL_WIDTH = 0.2
WHEEL_LUG_H = 0.0
WHEEL_BASE_H = 0.652
WHEEL_BASE_L = 0.775
MAX_STEER_ANGLE = 0.6

WHEEL_ORDER = ("FL", "FR", "ML", "MR", "BL", "BR")
WHEEL_BODY_NAMES = (
    "front_wheel_L", "front_wheel_R",
    "middle_wheel_L", "middle_wheel_R",
    "back_wheel_L", "back_wheel_R",
)


####################
## Joint 解析
####################

def resolve_zhurong_joint_groups(joint_names: list[str]) -> dict[str, dict[str, int]]:
    joint_name_to_idx = {name: i for i, name in enumerate(joint_names)}
    joint_groups = {
        "wheel_indices": {
            "FL": joint_name_to_idx["front_wheel_L_joint"],
            "FR": joint_name_to_idx["front_wheel_R_joint"],
            "ML": joint_name_to_idx["middle_wheel_L_joint"],
            "MR": joint_name_to_idx["middle_wheel_R_joint"],
            "BL": joint_name_to_idx["back_wheel_L_joint"],
            "BR": joint_name_to_idx["back_wheel_R_joint"],
        },
        "steer_indices": {
            "FL": joint_name_to_idx["suspension_steer_F_L_joint"],
            "FR": joint_name_to_idx["suspension_steer_F_R_joint"],
            "ML": joint_name_to_idx["suspension_steer_M_L_joint"],
            "MR": joint_name_to_idx["suspension_steer_M_R_joint"],
            "BL": joint_name_to_idx["suspension_steer_B_L_joint"],
            "BR": joint_name_to_idx["suspension_steer_B_R_joint"],
        },
        "arm_indices": {
            "FL": joint_name_to_idx["suspension_arm_F_L_joint"],
            "FR": joint_name_to_idx["suspension_arm_F_R_joint"],
            "BL": joint_name_to_idx["suspension_arm_B_L_joint"],
            "BR": joint_name_to_idx["suspension_arm_B_R_joint"],
        },
        "ptz_indices": {},
    }

    if "PTZYaw_joint" in joint_name_to_idx:
        joint_groups["ptz_indices"]["yaw"] = joint_name_to_idx["PTZYaw_joint"]
    if "PTZPitch_joint" in joint_name_to_idx:
        joint_groups["ptz_indices"]["pitch"] = joint_name_to_idx["PTZPitch_joint"]
    return joint_groups


def _as_column_tensor(values, device: str | torch.device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.float32).reshape(-1, 1)
    return torch.tensor(values, device=device, dtype=torch.float32).reshape(-1, 1)


####################
## 6WD / 6WS 运动学
####################

def compute_zhurong_joint_targets(
    linear_x,
    angular_z,
    num_joints: int,
    wheel_indices: dict[str, int],
    steer_indices: dict[str, int],
    arm_indices: dict[str, int],
    ptz_indices: dict[str, int],
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    v = _as_column_tensor(linear_x, device)
    w = _as_column_tensor(angular_z, device)

    joint_pos_target = torch.zeros((v.shape[0], num_joints), device=device)
    joint_vel_target = torch.zeros((v.shape[0], num_joints), device=device)

    straight_mask = w.abs() < 1.0e-3
    spin_mask = v.abs() < 1.0e-3
    turning_radius = v / (w + 1.0e-10)

    wheel_vel_straight = v / WHEEL_RADIUS
    r_lf = ((turning_radius - WHEEL_BASE_H).abs() ** 2 + WHEEL_BASE_L**2).sqrt()
    r_rf = ((turning_radius + WHEEL_BASE_H).abs() ** 2 + WHEEL_BASE_L**2).sqrt()
    r_lm = (turning_radius - WHEEL_BASE_H).abs()
    r_rm = (turning_radius + WHEEL_BASE_H).abs()

    vel_lf = w.abs() * r_lf / WHEEL_RADIUS
    vel_rf = w.abs() * r_rf / WHEEL_RADIUS
    vel_lm = w.abs() * r_lm / WHEEL_RADIUS
    vel_rm = w.abs() * r_rm / WHEEL_RADIUS
    vel_arr = torch.cat([vel_lf, vel_rf, vel_lm, vel_rm, vel_lf, vel_rf], dim=1)

    vel_arr = torch.where(v > 0, vel_arr, -vel_arr)
    vel_arr = torch.where(spin_mask, -w.sign() * vel_arr, vel_arr)

    theta_lf = torch.atan(WHEEL_BASE_L / (turning_radius - WHEEL_BASE_H + 1.0e-10))
    theta_rf = torch.atan(WHEEL_BASE_L / (turning_radius + WHEEL_BASE_H + 1.0e-10))
    zero = torch.zeros_like(theta_lf)
    theta_arr = torch.cat(
        [
            torch.clamp(theta_lf, -MAX_STEER_ANGLE, MAX_STEER_ANGLE),
            torch.clamp(theta_rf, -MAX_STEER_ANGLE, MAX_STEER_ANGLE),
            zero,
            zero,
            torch.clamp(-theta_lf, -MAX_STEER_ANGLE, MAX_STEER_ANGLE),
            torch.clamp(-theta_rf, -MAX_STEER_ANGLE, MAX_STEER_ANGLE),
        ],
        dim=1,
    )

    small_radius_left = (turning_radius >= 0) & (turning_radius < WHEEL_BASE_H)
    vel_arr = torch.where(
        small_radius_left & ~straight_mask,
        vel_arr * torch.tensor([[-1, 1, -1, 1, -1, 1]], device=device, dtype=vel_arr.dtype),
        vel_arr,
    )

    small_radius_right = (turning_radius < 0) & (turning_radius > -WHEEL_BASE_H)
    vel_arr = torch.where(
        small_radius_right & ~straight_mask,
        vel_arr * torch.tensor([[1, -1, 1, -1, 1, -1]], device=device, dtype=vel_arr.dtype),
        vel_arr,
    )

    vel_arr = torch.where(straight_mask, wheel_vel_straight.expand(-1, 6), vel_arr)
    theta_arr = torch.where(straight_mask, torch.zeros_like(theta_arr), theta_arr)

    for i, name in enumerate(WHEEL_ORDER):
        joint_vel_target[:, wheel_indices[name]] = vel_arr[:, i]
        joint_pos_target[:, steer_indices[name]] = theta_arr[:, i]

    for idx in arm_indices.values():
        joint_pos_target[:, idx] = 0.0
    if "yaw" in ptz_indices:
        joint_pos_target[:, ptz_indices["yaw"]] = 0.0
    if "pitch" in ptz_indices:
        joint_pos_target[:, ptz_indices["pitch"]] = 0.0
    return joint_pos_target, joint_vel_target


####################
## 旧接口：单环境计算
####################

def compute_wheel_velocities(linear_x: float, angular_z: float) -> dict[str, float]:
    if angular_z == 0:
        wheel_vel = linear_x / WHEEL_RADIUS
        return {name: wheel_vel for name in ("front_L", "front_R", "middle_L", "middle_R", "back_L", "back_R")}

    turning_radius = linear_x / angular_z
    r_lf = np.sqrt((turning_radius - WHEEL_BASE_H) ** 2 + WHEEL_BASE_L**2)
    r_rf = np.sqrt((turning_radius + WHEEL_BASE_H) ** 2 + WHEEL_BASE_L**2)
    r_lm = abs(turning_radius - WHEEL_BASE_H)
    r_rm = abs(turning_radius + WHEEL_BASE_H)

    if linear_x > 0:
        vel_arr = abs(angular_z) * np.array([r_lf, r_rf, r_lm, r_rm, r_lf, r_rf]) / WHEEL_RADIUS
    elif linear_x < 0:
        vel_arr = -abs(angular_z) * np.array([r_lf, r_rf, r_lm, r_rm, r_lf, r_rf]) / WHEEL_RADIUS
    else:
        vel_arr = np.sign(angular_z) * abs(angular_z) * np.array([r_lf, r_rf, r_lm, r_rm, r_lf, r_rf]) / WHEEL_RADIUS

    if 0 <= turning_radius < WHEEL_BASE_H:
        vel_arr[[0, 2, 4]] *= -1
    elif -WHEEL_BASE_H < turning_radius < 0:
        vel_arr[[1, 3, 5]] *= -1

    return {
        "front_L": float(vel_arr[0]),
        "front_R": float(vel_arr[1]),
        "middle_L": float(vel_arr[2]),
        "middle_R": float(vel_arr[3]),
        "back_L": float(vel_arr[4]),
        "back_R": float(vel_arr[5]),
    }


def compute_steer_angles(linear_x: float, angular_z: float) -> dict[str, float]:
    if angular_z == 0:
        return {name: 0.0 for name in ("front_L", "front_R", "middle_L", "middle_R", "back_L", "back_R")}

    turning_radius = linear_x / angular_z
    theta_fl = np.arctan(WHEEL_BASE_L / (turning_radius - WHEEL_BASE_H)) if (turning_radius - WHEEL_BASE_H) != 0 else 0.0
    theta_fr = np.arctan(WHEEL_BASE_L / (turning_radius + WHEEL_BASE_H)) if (turning_radius + WHEEL_BASE_H) != 0 else 0.0

    def clamp(value: float) -> float:
        return float(np.clip(value, -MAX_STEER_ANGLE, MAX_STEER_ANGLE))

    return {
        "front_L": clamp(theta_fl),
        "front_R": clamp(theta_fr),
        "middle_L": 0.0,
        "middle_R": 0.0,
        "back_L": clamp(-theta_fl),
        "back_R": clamp(-theta_fr),
    }


####################
## ROS Controller
####################

class ZhurongController(RoverController):
    WHEEL_RADIUS = WHEEL_RADIUS
    WHEEL_WIDTH = WHEEL_WIDTH
    WHEEL_BASE_H = WHEEL_BASE_H
    WHEEL_BASE_L = WHEEL_BASE_L
    MAX_STEER_ANGLE = MAX_STEER_ANGLE

    def __init__(self, robot=None, num_envs=1, device="cuda:0"):
        super().__init__("zhurong_controller", robot, num_envs, device, wheel_count=6)
        if robot is not None:
            self._init_joint_indices(robot)

    def _init_joint_indices(self, robot):
        self.joint_names = robot.joint_names
        joint_groups = resolve_zhurong_joint_groups(self.joint_names)
        self.wheel_indices = joint_groups["wheel_indices"]
        self.steer_indices = joint_groups["steer_indices"]
        self.arm_indices = joint_groups["arm_indices"]
        self.ptz_indices = joint_groups["ptz_indices"]

        print(f"[ZhurongController] Initialized with {len(self.joint_names)} joints")
        print(f"  Wheels: {list(self.wheel_indices.keys())}")
        print(f"  Steers: {list(self.steer_indices.keys())}")
        print(f"  Arms: {list(self.arm_indices.keys())}")
        if self.ptz_indices:
            print(f"  PTZ: {list(self.ptz_indices.keys())}")

    def compute_control_batch(self):
        if self.robot is None:
            raise RuntimeError("Robot not initialized. Call _init_joint_indices first.")

        joint_pos_target, joint_vel_target = compute_zhurong_joint_targets(
            self.linear_x,
            self.angular_z,
            num_joints=self.robot.num_joints,
            wheel_indices=self.wheel_indices,
            steer_indices=self.steer_indices,
            arm_indices=self.arm_indices,
            ptz_indices=self.ptz_indices,
            device=self.device,
        )

        for env_id in range(self.num_envs):
            override = self._wheel_omega_override[env_id]
            if override is None:
                continue
            # wheel_omega_cmd 覆写：按 WHEEL_ORDER 直填 6 个轮速。
            for idx, key in enumerate(WHEEL_ORDER):
                if key in self.wheel_indices:
                    joint_vel_target[env_id, self.wheel_indices[key]] = override[idx]
            for key in self.steer_indices:
                joint_pos_target[env_id, self.steer_indices[key]] = 0.0

        return joint_pos_target, joint_vel_target
