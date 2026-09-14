"""软土轮地力学 action term：逐物理步给六个轮子施加 Bekker-Wong 力。

放在 action term 而不是 event / 自定义 env，是因为 `ActionManager.apply_action()` 是
manager-based env 里唯一每个物理步都会被调用、且无需复写 `step()` 的钩子。本 term 的
``action_dim`` 为 0，不占策略输出维度。

reset 后的预压（把轮子放到静态下陷位置）不能在 `ActionTerm.reset()` 里做：V2+ 的出生
位姿是 command term 写的，而 command manager 的 reset 排在 action manager 之后。所以由
`SoftSoilNavEnv._reset_idx` 在所有 manager reset 完成后回调 `settle()`。
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..terramechanics import (
    PARAM,
    SOIL_KEYS,
    clamp_symmetric,
    compute_terramechanics,
    solve_static_sinkage,
    steered_wheel_axes_w,
)
from .observations import _bilinear_sample, _make_local_grid, _make_world_grid


def derive_soil_property_maps(
    soil_params: torch.Tensor,
    wheel_r: float,
    wheel_b: float,
    wheel_h: float,
    fn_ave: float,
    slip_ref: float,
) -> torch.Tensor:
    """Extract the two explicit physical parameters used by the policy.

    The terrain stores the complete seven-channel Bekker profile for dynamics,
    while the learning observation exposes only the dominant parameters n0 and
    phi.  Profiles are deduplicated before scattering back to the map.
    """
    flat = soil_params.reshape(-1, len(SOIL_KEYS))
    uniq, inverse = torch.unique(flat, dim=0, return_inverse=True)
    n0 = uniq[:, SOIL_KEYS.index("n0")]
    phi = uniq[:, SOIL_KEYS.index("phi")]
    props = torch.stack([n0, phi], dim=-1)   # (n, 2)
    return props[inverse].reshape(*soil_params.shape[:2], 2)


class SoftSoilAction(ActionTerm):
    """Bekker-Wong 软土支撑力，替代刚性地面碰撞。

    要求场景地形用 ``terrain_only_nocollide.usd``：法向支撑完全由本 term 提供，
    若地形同时有碰撞体会双重支撑，车会浮起来。
    """

    cfg: "SoftSoilActionCfg"

    def __init__(self, cfg: "SoftSoilActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        robot = self._asset
        self.terrain_size = cfg.terrain_size
        self.wheel_r = cfg.wheel_radius
        self.wheel_b = cfg.wheel_width
        self.wheel_h = cfg.wheel_lug_height

        body_index = {name: i for i, name in enumerate(robot.body_names)}
        joint_index = {name: i for i, name in enumerate(robot.joint_names)}
        self.wheel_body_ids = [body_index[n] for n in cfg.wheel_body_names]
        self.steer_joint_ids = torch.tensor(
            [joint_index[n] for n in cfg.steer_joint_names], device=self.device)
        self.wheel_joint_ids = torch.tensor(
            [joint_index[n] for n in cfg.wheel_joint_names], device=self.device)
        self.num_wheels = len(self.wheel_body_ids)

        # 地面高程用不含岩石的 height_gt_map：岩石仍是刚体碰撞体，不该被软土再托一次。
        height_path = os.path.join(cfg.terrain_dir, "height_gt_map.npy")
        if not os.path.exists(height_path):
            raise FileNotFoundError(f"Ground height map not found: {height_path}")
        self.height_t = torch.as_tensor(
            np.load(height_path), dtype=torch.float32, device=self.device)

        soil_path = os.path.join(cfg.terrain_dir, "soil_params_map.npy")
        if not os.path.exists(soil_path):
            raise FileNotFoundError(
                f"Soft-soil task needs soil_params_map.npy but it is missing: {soil_path}")
        soil_np = np.load(soil_path)
        if soil_np.ndim != 3 or soil_np.shape[-1] != len(SOIL_KEYS):
            raise ValueError(
                f"soil_params_map must be (H,W,{len(SOIL_KEYS)}), got {soil_np.shape}")
        self.soil_params_t = torch.as_tensor(soil_np, dtype=torch.float32, device=self.device)

        gravity = abs(float(env.cfg.sim.gravity[2]))
        self.total_mass = float(robot.data.default_mass[0].sum())
        self.fn_ave = self.total_mass * gravity / self.num_wheels

        self.soil_property_map = derive_soil_property_maps(
            self.soil_params_t, self.wheel_r, self.wheel_b, self.wheel_h,
            self.fn_ave, cfg.slip_reference)

        shape = (self.num_envs, self.num_wheels)
        self.sinkage = torch.zeros(shape, device=self.device)
        self.slip = torch.zeros(shape, device=self.device)
        self.sigma = torch.zeros(shape, device=self.device)
        self.tau = torch.zeros(shape, device=self.device)
        self.contact = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.force_z = torch.zeros(shape, device=self.device)
        self.force_forward = torch.zeros(shape, device=self.device)
        self._tm_state = {"sinkage_prev": torch.full(shape, 2.0e-4, device=self.device)}
        self._forces = torch.zeros(*shape, 3, device=self.device)
        self._torques = torch.zeros(*shape, 3, device=self.device)
        self._null_actions = torch.zeros(self.num_envs, 0, device=self.device)

        n0_map = self.soil_property_map[..., 0]
        print(f"[SoftSoil] mass={self.total_mass:.1f}kg g={gravity:.2f} "
              f"Fn/wheel={self.fn_ave:.1f}N | n0 over map: "
              f"min={float(n0_map.min()):.3f} p50={float(n0_map.median()):.3f} "
              f"max={float(n0_map.max()):.3f}")

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._null_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._null_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        self.sinkage[idx] = 0.0
        self.slip[idx] = 0.0
        self.sigma[idx] = 0.0
        self.tau[idx] = 0.0
        self.contact[idx] = False
        self.force_z[idx] = 0.0
        self.force_forward[idx] = 0.0
        self._tm_state["sinkage_prev"][idx] = 2.0e-4
        self._forces[idx] = 0.0

    def _wheel_soil(self, wheel_xy_flat: torch.Tensor, count: int) -> dict:
        sampled = _bilinear_sample(self.soil_params_t, wheel_xy_flat, self.terrain_size)
        sampled = sampled.reshape(count, self.num_wheels, len(SOIL_KEYS))
        return dict(zip(SOIL_KEYS, sampled.unbind(-1)))

    def apply_actions(self) -> None:
        robot = self._asset
        wheel_pos_w = robot.data.body_pos_w[:, self.wheel_body_ids, :]
        wheel_xy_flat = wheel_pos_w[..., :2].reshape(-1, 2)

        ground_z = _bilinear_sample(self.height_t, wheel_xy_flat, self.terrain_size)
        ground_z = ground_z.reshape(self.num_envs, self.num_wheels)
        sinkage = torch.clamp(self.wheel_r - (wheel_pos_w[..., 2] - ground_z), min=0.0)

        steer = robot.data.joint_pos[:, self.steer_joint_ids]
        forward_w, right_w = steered_wheel_axes_w(robot.data.root_quat_w, steer, self.device)
        wheel_vel_w = robot.data.body_lin_vel_w[:, self.wheel_body_ids, :]
        v_forward = (wheel_vel_w * forward_w).sum(dim=-1)

        out = compute_terramechanics(
            sinkage,
            robot.data.joint_vel[:, self.wheel_joint_ids],
            v_forward,
            wheel_vel_w[..., 2],
            wheel={"r": self.wheel_r, "b": self.wheel_b, "h": self.wheel_h},
            fn_ave=self.fn_ave,
            state=self._tm_state,
            soil=self._wheel_soil(wheel_xy_flat, self.num_envs),
        )
        self._tm_state = out["state"]

        # 每环境驱动机械功率（W）：土壤阻力矩 × 轮角速度。净推进力在巡航时 ≈ 0，
        # 会漏掉全部土壤阻力功；阻力矩才是驱动侧真实付出的功，类间区分度在这里。
        omg = robot.data.joint_vel[:, self.wheel_joint_ids]
        self.drive_torque = out["Mr"]
        self.power = (out["Mr"].abs() * omg.abs()).sum(dim=1)

        v_side = (wheel_vel_w * right_w).sum(dim=-1)
        side_drag = clamp_symmetric(
            -PARAM["LateralDampCoef"] * v_side, out["Fn"] * out["tan_phi"])
        forces = forward_w * out["force_forward"][..., None] + right_w * side_drag[..., None]
        forces[:, :, 2] += out["force_z"]
        self._forces = forces

        robot.set_external_force_and_torque(
            forces, self._torques, body_ids=self.wheel_body_ids, is_global=True)

        self.sinkage = out["sinkage"]
        self.slip = out["slip"]
        self.sigma = out["sigma_m"]
        self.tau = out["tao_m"]
        self.contact = out["contact"]
        self.force_z = out["force_z"]
        self.force_forward = out["force_forward"]

    def settle(self, env_ids: Sequence[int] | None = None) -> None:
        """把刚重置的车压到静态平衡位姿：轮子沉到 z_eq，车体贴合六轮拟合平面。

        不做这一步的话开局 sinkage=0 → Fn=0 → 自由落体，落地瞬间 Fn 尖峰会把车弹飞。
        """
        robot = self._asset
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device)
        if env_ids.numel() == 0:
            return
        count = env_ids.numel()

        wheel_pos_w = robot.data.body_pos_w[env_ids][:, self.wheel_body_ids, :]
        wheel_xy = wheel_pos_w[..., :2]
        wheel_xy_flat = wheel_xy.reshape(-1, 2)
        ground_z = _bilinear_sample(self.height_t, wheel_xy_flat, self.terrain_size)
        ground_z = ground_z.reshape(count, self.num_wheels)

        fn = torch.full((count, self.num_wheels), self.fn_ave, device=self.device)
        z_eq = solve_static_sinkage(
            fn, self.wheel_r, self.wheel_b, self.wheel_h,
            soil=self._wheel_soil(wheel_xy_flat, count))

        pose = robot.data.root_state_w[env_ids, :7].clone()
        pose[:, 2] += ((ground_z + self.wheel_r - z_eq) - wheel_pos_w[..., 2]).mean(dim=1)

        # 六个轮下地面点最小二乘拟平面，base +Z 对齐法线，再叠回原有 yaw。
        design = torch.cat([wheel_xy, torch.ones(count, self.num_wheels, 1, device=self.device)], dim=-1)
        abc = torch.linalg.lstsq(design, ground_z.unsqueeze(-1)).solution[:, :, 0]
        normal = torch.stack([-abc[:, 0], -abc[:, 1], torch.ones(count, device=self.device)], dim=-1)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        qw = torch.sqrt(torch.clamp((1.0 + normal[:, 2]) / 2.0, min=1.0e-8))
        q_tilt = torch.stack([qw, -normal[:, 1] / (2.0 * qw), normal[:, 0] / (2.0 * qw),
                              torch.zeros(count, device=self.device)], dim=-1)

        q = pose[:, 3:7]
        yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        q_yaw = torch.zeros_like(q)
        q_yaw[:, 0] = torch.cos(yaw / 2.0)
        q_yaw[:, 3] = torch.sin(yaw / 2.0)
        pose[:, 3:7] = _quat_mul(q_tilt, q_yaw)

        robot.write_root_pose_to_sim(pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim(
            torch.zeros(count, 6, device=self.device), env_ids=env_ids)
        self._tm_state["sinkage_prev"][env_ids] = z_eq
        self.sinkage[env_ids] = z_eq


####################
## 观测 / 奖励 / 终止
####################

def _soft_soil_term(env: ManagerBasedEnv, action_name: str) -> SoftSoilAction:
    term = env.action_manager.get_term(action_name)
    if not isinstance(term, SoftSoilAction):
        raise TypeError(f"Action term '{action_name}' is not a SoftSoilAction.")
    return term


def wheel_sinkage(env: ManagerBasedEnv, action_name: str) -> torch.Tensor:
    """逐轮当前下陷深度，归一化到轮半径。本体感受，三个对照组都有。"""
    term = _soft_soil_term(env, action_name)
    return term.sinkage / term.wheel_r


class SoilScanGT(ManagerTermBase):
    """Vehicle-frame lookup of the two explicit soil parameters n0 and phi.

    两个通道都是查表得到的地面真值，不随车的动态状态变化，所以是"前视"信息——
    策略能看到还没压上去的那片土有多软。``decorrelate_shift`` 把整张图平移一个固定
    偏移，用来做同维度、同分布但空间失配的对照组。
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.local_xy = _make_local_grid(
            cfg.params["scan_size"], cfg.params["scan_res"], env.device)
        source = _soft_soil_term(env, cfg.params["action_name"])
        self.terrain_size = source.terrain_size
        prop_map = source.soil_property_map
        shift = cfg.params.get("decorrelate_shift")
        if shift is not None:
            prop_map = torch.roll(prop_map, shifts=(int(shift[0]), int(shift[1])), dims=(0, 1))
        self.prop_map = prop_map
        self.scale = torch.tensor(cfg.params["channel_scale"], device=env.device)
        self.enabled = bool(cfg.params.get("enabled", True))
        self.num_channels = self.prop_map.shape[-1]
        flat = self.prop_map.reshape(-1, self.num_channels) * self.scale
        print(f"[SoilScanGT] {self.local_xy.shape[0]} points x {self.num_channels} channels, "
              f"normalized range per channel: "
              + " ".join(f"[{float(flat[:, i].min()):.2f},{float(flat[:, i].max()):.2f}]"
                         for i in range(self.num_channels))
              + (f" (decorrelated by {tuple(shift)})" if shift is not None else ""))

    def __call__(self, env, asset_cfg, action_name, scan_size, scan_res,
                 channel_scale, decorrelate_shift=None, enabled=True):
        robot = env.scene[asset_cfg.name]
        if not enabled or not self.enabled:
            return torch.zeros(
                (env.num_envs, self.local_xy.shape[0] * self.num_channels),
                dtype=torch.float32, device=env.device)
        xy = _make_world_grid(robot.data.root_pos_w, robot.data.heading_w, self.local_xy)
        num_envs, num_points, _ = xy.shape
        sampled = _bilinear_sample(self.prop_map, xy.reshape(-1, 2), self.terrain_size)
        return (sampled * self.scale).reshape(num_envs, num_points * self.num_channels)


def sinkage_penalty(
    env: ManagerBasedEnv, action_name: str, safe_sinkage: float
) -> torch.Tensor:
    """超过安全下陷阈值的部分按平方计罚，硬地上恒为 0。

    取平方而非线性，使轻微越过安全线和深度下陷在奖励中保持可区分的代价。
    """
    term = _soft_soil_term(env, action_name)
    excess = (term.sinkage - safe_sinkage).clamp(min=0.0) / term.wheel_r
    return excess.square().mean(dim=1)


def wheel_slip_penalty(
    env: ManagerBasedEnv, action_name: str, slip_threshold: float
) -> torch.Tensor:
    """只惩罚超阈值的打滑。正常行驶滑移率本来就有 0.15~0.3，全罚等于罚"开车"。"""
    slip = _soft_soil_term(env, action_name).slip.abs()
    return (slip - slip_threshold).clamp(min=0.0).mean(dim=1)


class BoggedDown(ManagerTermBase):
    """连续多个控制步"陷得深 + 打滑重 + 几乎不动"就判定陷车，提前结束回合。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._counter.zero_()
        else:
            self._counter[env_ids] = 0

    def __call__(self, env, action_name, asset_cfg, sinkage_ratio,
                 slip_threshold, speed_threshold, patience_steps):
        term = _soft_soil_term(env, action_name)
        speed = env.scene[asset_cfg.name].data.root_lin_vel_w[:, :2].norm(dim=-1)
        stuck = (
            (term.sinkage.mean(dim=1) > sinkage_ratio * term.wheel_r)
            & (term.slip.abs().mean(dim=1) > slip_threshold)
            & (speed < speed_threshold)
        )
        self._counter = torch.where(stuck, self._counter + 1, torch.zeros_like(self._counter))
        return self._counter >= patience_steps


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


@configclass
class SoftSoilActionCfg(ActionTermCfg):
    """Bekker-Wong 软土支撑。terrain_dir / terrain_size 由 env cfg 的 __post_init__ 注入。"""

    class_type: type[ActionTerm] = SoftSoilAction
    asset_name: str = "robot"
    terrain_dir: str = ""
    terrain_size: float = 100.0
    wheel_radius: float = 0.15
    wheel_width: float = 0.2
    wheel_lug_height: float = 0.0
    wheel_body_names: tuple[str, ...] = (
        "front_wheel_L", "front_wheel_R",
        "middle_wheel_L", "middle_wheel_R",
        "back_wheel_L", "back_wheel_R",
    )
    wheel_joint_names: tuple[str, ...] = (
        "front_wheel_L_joint", "front_wheel_R_joint",
        "middle_wheel_L_joint", "middle_wheel_R_joint",
        "back_wheel_L_joint", "back_wheel_R_joint",
    )
    steer_joint_names: tuple[str, ...] = (
        "suspension_steer_F_L_joint", "suspension_steer_F_R_joint",
        "suspension_steer_M_L_joint", "suspension_steer_M_R_joint",
        "suspension_steer_B_L_joint", "suspension_steer_B_R_joint",
    )
    slip_reference: float = 0.3
