"""Bekker-Wong 轮地力学：软土下陷、法向力与挂钩牵引力。

移植自 mars_sim 的 TerramechanicsPlugin.cpp，注释里的 (cpp:437) 指原始行号。
全部逐元素运算，入参 shape 统一 (E, N)（E 个环境 × N 个轮），可直接向量化跑多环境。

与 mars_sim 的两处有意偏离，见 `_apply_longitudinal_limits` 与 `_motion_opposing_resistance`。
"""

from __future__ import annotations

import math

import torch

####################
## 土壤参数
####################

# 全局默认土（无分层土图时回落）。数量级参考 Wong 的月壤/火星风成沙表。
SOIL = {
    "Kc": 13600.0,        # 黏聚模量
    "Kphi": 2259100.0,    # 摩擦模量
    "n0": 0.92, "n1": 0.5,  # 下陷指数 n = n0 + n1*|s|
    "c": 462.3,           # 黏聚力 (Pa)
    "phi": 0.61,          # 内摩擦角 (rad ≈ 35°)
    "K": 0.014,           # 剪切变形模量 (m)
}

# soil_params_map 的 7 通道顺序，地形生成端和仿真消费端共用。
SOIL_KEYS = ("Kc", "Kphi", "n0", "n1", "c", "phi", "K")


####################
## mars_sim 经验参数
####################

PARAM = {
    "c1": 0.5, "c2": -0.3, "c3": 0.0,            # 最大应力角 thetam = (c1+c2*s)*theta1
    "c_T1": 0.214,
    "c_d1": -0.626, "c_d2": 0.308, "c_d3": -0.224,  # Fdp 修正
    "sinkage_max": 0.15, "sinkage_step": 0.01,   # .hh:236,240
    "Fn_Max": 5000.0,         # .hh:240
    "DampCoef": 500.0,        # 法向阻尼 per-wheel（cpp:495 Dz=5000@dt=0.001；dt=0.01 下 5000 不稳）
    "LongitudinalDampCoef": 200.0,  # 纵向速度阻尼（cpp:464,523）
    "LateralDampCoef": 200.0,       # 侧向被动阻尼（调用方用，近似 mars_sim FY_sub）
    "OmegaDeadband": 0.05,          # 低速轮速死区，避免关节抖动触发 slip=1
    "ResistVelEps": 0.05,           # 见 _motion_opposing_resistance
}

_EPS = 1.0e-6


def _calc_slip(omega: torch.Tensor, v: torch.Tensor, r_eff: float) -> torch.Tensor:
    # 滑移率 s ∈ [-1,1]（cpp:722）。r_eff = r + 0.5h。
    omega_active = omega.abs() >= PARAM["OmegaDeadband"]
    del_v = torch.where(
        omega > 0,
        r_eff * omega - v,
        r_eff * omega.abs() - v.abs(),
    )
    # denom 由 omega_active 守护（非激活置 1），drive 分支不再单独判 |v|
    denom = torch.where(omega_active, r_eff * omega, torch.ones_like(omega))
    s = torch.zeros_like(omega)
    drive = (del_v > 1.0e-3) & omega_active                         # 驱动滑转
    brake = (del_v < -1.0e-3) & omega_active & (v.abs() >= 1.0e-3)  # 制动滑移
    s = torch.where(drive, 1.0 - (v / denom).abs(), s)
    s = torch.where(brake, (denom.abs() / v.abs()) - 1.0, s)
    s = s.clamp(-1.0, 1.0)
    # 角速度与前进速度异号 → 纯滑转
    s = torch.where(((omega * v) < 0) & omega_active & (v.abs() >= 1.0e-3), torch.ones_like(s), s)
    return s


def _calc_effective_radius(s: torch.Tensor, r: float, h: float) -> torch.Tensor:
    # 等效半径 Rs（cpp:913）。h=0（光面轮）时恒为 r，整段为 no-op。
    Rs = torch.full_like(s, r + h)
    Rs = torch.where(s > 0.5, torch.full_like(s, r), Rs)
    mid = (s >= 0.15) & (s <= 0.5)
    return torch.where(mid, r + h * (0.5 - s) / 0.35, Rs)


def clamp_symmetric(x: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
    """把 x 限制到 [-limit, limit]，limit 可为 tensor。"""
    return torch.maximum(torch.minimum(x, limit), -limit)


def _motion_opposing_resistance(Fdp: torch.Tensor, v_forward: torch.Tensor) -> torch.Tensor:
    """净阻力（Fdp<0）只能阻碍运动，不能反向推动车体。

    偏离 mars_sim：Bekker 的 Fdp = 土壤推力 − 压实阻力。深陷时压实阻力占优，Fdp 变负。
    mars_sim 把这个负值当车体系恒力直接施加，车静止时会被"倒着推走"。压实阻力本质是
    随运动方向反号的，所以这里按 v 的方向调制：v≈0 时归零，车陷住就是不动，而不是倒退。
    tanh 而非 sign，避免零速附近逐步换号引起抖振。
    """
    resist = torch.clamp(-Fdp, min=0.0)
    shaped = -resist * torch.tanh(v_forward / PARAM["ResistVelEps"])
    return torch.where(Fdp >= 0.0, Fdp, shaped)


def _apply_longitudinal_limits(
    Fdp: torch.Tensor, s_flag: torch.Tensor, omega: torch.Tensor, v_forward: torch.Tensor,
) -> torch.Tensor:
    # mars_sim 动态分支：纵向符号/限幅/反向拖拽（cpp:458-472,516-525），逐行对应。
    long_damp = PARAM["LongitudinalDampCoef"]
    Fdp = torch.where((s_flag < 0) & (Fdp > 0), torch.zeros_like(Fdp), Fdp)                # cpp:458 制动却正向 → 归零
    Fdp = torch.where(s_flag < 0, Fdp - long_damp * v_forward.abs(), Fdp)                  # cpp:464 制动叠纵向阻尼
    Fdp = torch.where((s_flag >= 0) & (Fdp < -100.0), torch.full_like(Fdp, -100.0), Fdp)   # cpp:472 驱动下限
    Fdp = torch.where(omega < -PARAM["OmegaDeadband"], -Fdp, Fdp)                          # cpp:516 反向轮速翻转符号
    return torch.where((omega * v_forward) < 0, Fdp - long_damp * v_forward, Fdp)          # cpp:523 反拖阻尼


def compute_terramechanics(
    sinkage: torch.Tensor,
    omega: torch.Tensor,
    v_forward: torch.Tensor,
    vz: torch.Tensor,
    wheel: dict,
    fn_ave: float | torch.Tensor,
    state: dict,
    soil: dict | None = None,
) -> dict:
    """Bekker-Wong 轮地力（mars_sim 移植）。入参均 (E,N)：

    sinkage 几何下陷 m（垂直距离 r-(轮心z-地面z)）/ omega 轮角速度 / v_forward 轮前向速度
    vz 轮垂速（世界系，法向阻尼用）
    wheel={"r","b","h"}；fn_ave 单轮平均法向力（修正系数参考）；state={"sinkage_prev"} 跨步限幅

    返回 force_z（世界 +Z，含阻尼，非负）/ force_forward（纵向标量，调用方沿轮前向旋世界）
    / Fn（原始法向）/ sigma_m（法向应力）/ tao_m（剪应力）/ slip / sinkage / state
    """
    r = float(wheel["r"]); b = float(wheel["b"]); h = float(wheel["h"])
    r_eff = r + 0.5 * h                          # rs（cpp:410）

    sd = SOIL if soil is None else soil               # 分层土：None 回落全局；否则 per-wheel（值可为标量或 tensor）
    Kc, Kphi = sd["Kc"], sd["Kphi"]
    n0, n1 = sd["n0"], sd["n1"]
    c, phi, K = sd["c"], sd["phi"], sd["K"]
    tan_phi = torch.tan(phi) if torch.is_tensor(phi) else math.tan(phi)  # phi 标量走 math.tan（位一致），tensor 走 torch.tan
    c1, c2, c3 = PARAM["c1"], PARAM["c2"], PARAM["c3"]
    c_T1 = PARAM["c_T1"]
    cd1, cd2, cd3 = PARAM["c_d1"], PARAM["c_d2"], PARAM["c_d3"]
    z_max, z_step = PARAM["sinkage_max"], PARAM["sinkage_step"]
    Fn_max = PARAM["Fn_Max"]
    DampCoef = PARAM["DampCoef"]

    sinkage_prev = state["sinkage_prev"]

    # 滑移率（cpp:722）
    s = _calc_slip(omega, v_forward, r_eff)

    # 下陷速率限幅 + 上限（cpp:827,838）
    d = sinkage - sinkage_prev
    sinkage_lim = torch.where(d > z_step, sinkage_prev + z_step, sinkage)
    sinkage_lim = torch.where(d < -z_step, sinkage_prev - z_step, sinkage_lim)
    sinkage_lim = sinkage_lim.clamp(max=z_max)

    n = n0 + n1 * s.abs()                        # cpp:283

    # 接触几何角（cpp:849,301,307）
    theta1 = torch.acos(((r - sinkage_lim) / r).clamp(-1.0, 1.0))
    theta2 = c3 * theta1
    thetam = (c1 + c2 * s) * theta1
    theta_m2 = thetam - theta2
    theta_1m = theta1 - thetam
    theta_12 = theta1 - theta2

    # 积分系数 A/B/C（cpp:314-320）
    A = ((torch.cos(thetam) - torch.cos(theta2)) / (theta_m2 + _EPS)
         + (torch.cos(thetam) - torch.cos(theta1)) / (theta_1m + _EPS))
    B = ((torch.sin(thetam) - torch.sin(theta2)) / (theta_m2 + _EPS)
         + (torch.sin(thetam) - torch.sin(theta1)) / (theta_1m + _EPS))
    C = theta_12 / 2.0

    Rs = _calc_effective_radius(s, r, h)
    theta11 = torch.acos((r * torch.cos(theta1) / Rs).clamp(-1.0, 1.0))  # cpp:322

    # 法向应力 σ_m（cpp:323）
    sigma_m = (Kc / b + Kphi) * r**n * (torch.cos(thetam) - torch.cos(theta1))**n

    # 剪位移 jx（cpp:415-426）
    s_flag = torch.where(s >= 0, 1.0, -1.0)
    jx_drive = r_eff * ((theta11 - thetam) - (1 - s) * (torch.sin(theta11) - torch.sin(thetam)))
    jx_brake = -r_eff * ((theta11 - thetam) - (torch.sin(theta11) - torch.sin(thetam)) / (1 + 0.9 * s))
    jx = torch.where(s_flag >= 0, jx_drive, jx_brake)
    K_eff = K.clamp(min=_EPS) if torch.is_tensor(K) else max(float(K), _EPS)   # K 配置驱动（分层土），clamp 防零除
    exp_jk = 1.0 - torch.exp(-jx / K_eff)           # cpp:428
    tao_m = (c + sigma_m * tan_phi) * exp_jk    # cpp:430

    # 法向力 Fn（cpp:437-438），LimitSustainForce + 非负保险（防 theta2=0 时 A 为负带歪 passive_limit）
    Fn1 = r * b * A * sigma_m + r_eff * b * tao_m * B
    Fn = torch.clamp(Fn1, min=0.0, max=Fn_max)

    # 滚动阻力矩 Mr（cpp:440-443，土壤剪切阻力矩，非电机扭矩）
    coef_tempT = 1.0 + c_T1 * (fn_ave - Fn) / fn_ave
    Mr1 = r_eff**2 * C * (b * c + coef_tempT * Fn * tan_phi / (r * A)) * exp_jk
    Mr2 = 1.0 + r_eff * B * tan_phi * exp_jk / (r * A)
    Mr = Mr1 / Mr2

    # 纵向推力 Fdp（cpp:446-450,457）
    coef_s = cd1 + cd2 * s
    coef_t = cd3 * (fn_ave - Fn) / fn_ave
    Fdp1 = (A / C + B**2 / (A * C)) * Mr / r_eff - B * Fn / A
    Fdp = Fdp1 * (1.0 + coef_s) * (1.0 + coef_t) * s_flag

    # mars_sim 动态分支的纵向限幅/阻尼
    Fdp = _apply_longitudinal_limits(Fdp, s_flag, omega, v_forward)

    # W1.2 静态近似：轮未驱动时用有库仑上限的纵向阻尼替代 StaticModel 的位移静摩擦，
    # 避免无碰撞地面在微小坡度/初速下越滑越快。
    passive_limit = Fn * tan_phi
    passive_drag = clamp_symmetric(-PARAM["LongitudinalDampCoef"] * v_forward, passive_limit)
    Fdp = torch.where(omega.abs() < PARAM["OmegaDeadband"], passive_drag, Fdp)

    # 深陷时净阻力占优，只能阻碍运动而不能反推车体
    Fdp = _motion_opposing_resistance(Fdp, v_forward)

    # 近零沉陷硬地：接触角 theta1~sqrt(z) -> 0 时，Fdp 的 1/A 角度积分发散为非物理
    # 大值。牵引力不可能超出摩擦锥，按轮下土的内摩擦角钳正；软土工况 Fdp 远低于该
    # 包络（loose 峰值 ~30N vs cap ~83N），此钳位只在硬地端生效。
    Fdp = torch.minimum(Fdp, Fn * tan_phi)

    # 法向力（阻尼 cpp:495-508），非负 clamp 防"土壤吸住"轮子
    Fz = torch.clamp(Fn - DampCoef * vz.clamp(-5.0, 5.0), min=0.0)

    # 离地判定用几何 sinkage（非限幅值）：无碰撞下限幅滞后，必须用几何真值判接触，
    # 否则车抛起时 sinkage_lim 残留正值 → Fn 持续顶飞正反馈（mars_sim 有碰撞托底可放心用 sinkage_lim）。
    # 阈值 0.001mm：必须低于任何在用土类的静态平衡沉陷（刚性 proxy 类约 3mm，
    # 且弹跳瞬态时 z 会瞬时跌破 0.02mm），否则硬地会被间歇判为"离地"而丢失全部轮地力。
    contact = sinkage > 1.0e-6
    Fz = torch.where(contact, Fz, torch.zeros_like(Fz))
    Fdp = torch.where(contact, Fdp, torch.zeros_like(Fdp))
    sinkage_prev_new = torch.where(contact, sinkage_lim, torch.full_like(sinkage_lim, 2.0e-4))

    zero = torch.zeros_like(Fn)
    return {
        "Mr": torch.where(contact, Mr, torch.zeros_like(Mr)),
        "force_z": Fz,
        "force_forward": Fdp,
        "Fn": Fn,
        "sigma_m": torch.where(contact, sigma_m, zero),   # 承压 (Pa)，观测用
        "tao_m": torch.where(contact, tao_m, zero),       # 剪应力 (Pa)，观测用
        "tan_phi": tan_phi,       # per-wheel（分层土），供消费端侧向限幅 side_limit
        "slip": s,
        "sinkage": torch.where(contact, sinkage_lim, zero),
        "contact": contact,
        "state": {"sinkage_prev": sinkage_prev_new},
    }


def solve_static_sinkage(fn_target, r: float, b: float, h: float = 0.0, soil: dict | None = None):
    """二分法解静态（slip=0, vz=0）下 Fn(z)=fn_target 的下陷深度。

    spawn 预压用：起步即 Fn≈mg/wheel，避免 sinkage=0→Fn=0→自由落体瞬态发散。
    fn_target 标量 → 返回 float；(E,N) tensor + soil → 返回 (E,N) per-wheel sinkage_eq。
    """
    is_scalar = not torch.is_tensor(fn_target)
    z_max = PARAM["sinkage_max"]
    if is_scalar:
        ft = torch.full((1, 1), float(fn_target), dtype=torch.float32)
        lo, hi = torch.zeros_like(ft), torch.full_like(ft, z_max)
    else:
        ft, lo, hi = fn_target, torch.zeros_like(fn_target), torch.full_like(fn_target, z_max)
    zeros = torch.zeros_like(ft)
    for _ in range(40):
        mid = (lo + hi) / 2.0
        out = compute_terramechanics(
            mid, zeros, zeros, zeros,
            {"r": r, "b": b, "h": h}, fn_ave=ft,
            state={"sinkage_prev": mid.clone()},   # 绕过速率限幅
            soil=soil,
        )
        too_small = out["Fn"] < ft
        lo = torch.where(too_small, mid, lo)
        hi = torch.where(too_small, hi, mid)
    result = (lo + hi) / 2.0
    return float(result[0, 0]) if is_scalar else result


####################
## Isaac 侧几何工具
####################

def quat_apply_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = q[..., 1:4]
    q_w = q[..., 0:1]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


def root_axes_w(root_quat_w: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    """车体 +X / +Y 轴在世界系下的方向。"""
    num_envs = root_quat_w.shape[0]
    forward_b = torch.zeros(num_envs, 3, device=device)
    right_b = torch.zeros(num_envs, 3, device=device)
    forward_b[:, 0] = 1.0
    right_b[:, 1] = 1.0
    return quat_apply_wxyz(root_quat_w, forward_b), quat_apply_wxyz(root_quat_w, right_b)


def steered_wheel_axes_w(root_quat_w: torch.Tensor, steer: torch.Tensor, device):
    """按各轮转向角把车体轴旋成 per-wheel 的前向/侧向世界轴，shape (E,N,3)。"""
    root_forward_w, root_right_w = root_axes_w(root_quat_w, device)
    wheel_forward_w = (
        torch.cos(steer)[..., None] * root_forward_w[:, None, :]
        + torch.sin(steer)[..., None] * root_right_w[:, None, :]
    )
    wheel_right_w = (
        -torch.sin(steer)[..., None] * root_forward_w[:, None, :]
        + torch.cos(steer)[..., None] * root_right_w[:, None, :]
    )
    return wheel_forward_w, wheel_right_w
