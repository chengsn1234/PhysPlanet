# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Wh12 单轮土槽试验台复现（HIT 试验对比）
####################
## 链路：轮关节速度控制 ω + 台架 PI 跟踪 v=(1-s)·r_eff·ω -> Bekker-Wong 轮地力 -> 稳态沉陷/挂钩牵引力/阻力矩
## 对应 HIT 单轮试验：Earth 重力，表1 土参数，滑转率扫掠取稳态窗口
## 地面无碰撞（与整车软地仿真一致），沉陷为几何量，力全部由公式模型给出
####################

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Wh12 single-wheel soil-bin testbed (HIT comparison).")
parser.add_argument("--soil", type=str, default="fig1", choices=["fig1"],
                    help="fig1=试验土槽参数（图1：kc/kphi/n0/n1/c/phi/K）")
parser.add_argument("--load", type=float, default=80.0, help="垂直载荷 W (N)，表2-3 G1 滑转率组次 80N")
parser.add_argument("--cd", type=str, default="platform", choices=["platform", "bench"],
                    help="FDP 经验系数：platform=mars_sim 原值，bench=本台架重辨识(常数增益0.90)")
parser.add_argument("--tag", type=str, default=None, help="输出文件名 tag，默认=<soil>_<cd>")
parser.add_argument("--wheel_mass", type=float, default=5.0, help="轮体质量 (kg)，台架配重 = W/g - wheel_mass")
parser.add_argument("--omega", type=float, default=1.0, help="轮角速度指令 (rad/s)")
parser.add_argument("--settle", type=float, default=12.0, help="每相位稳定时间 (s)，exp1-1 口径 dwell 12s")
parser.add_argument("--measure", type=float, default=5.0, help="每相位测量窗 (s)，取末段稳态窗均值")
parser.add_argument("--out", type=str, default=None,
                   help="输出CSV路径（repo根相对），默认 docs/experiments/wh12_single_wheel/data/wh12_slip_sweep_<soil>.csv")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from terramechanics import compute_terramechanics, solve_static_sinkage

####################
## 试验条件（HIT 单轮试验台）
####################

WHEEL = {"r": 0.135, "b": 0.165, "h": 0.010}   # Wh12: 半径/轮宽/抓地齿高
R_EFF = WHEEL["r"] + 0.5 * WHEEL["h"]          # 模型滑移定义用等效半径 rs
G_EARTH = 9.81                                 # 试验台在 Earth 重力下

# 试验土槽参数（图1：kc=15.6kPa/m^(N-1), kphi=2407.6kPa/m^N, n0=0.895, n1=0.995,
# c=250Pa, phi=31.9deg, K=0.01m）。n1≈1 即滑转下陷指数，是 z(s) 抬头的主因。
SOIL_FIG1 = {
    "Kc": 15600.0,        # kPa/m^(n-1) -> Pa
    "Kphi": 2407600.0,    # kPa/m^n -> Pa
    "n0": 0.895, "n1": 0.995,
    "c": 250.0,           # Pa
    "phi": math.radians(31.9),
    "K": 0.01,            # m
}

# FDP 经验系数（PARAM 的 c_d1/c_d2/c_d3）：platform 为 mars_sim 原值；
# bench 为本台架重辨识值：稳态下 Fn≡W 使 c_d3 不可辨识，c_d2 也不必要，
# 实测只需一个常数增益 0.90（离线辨识，见 exp13_package/scripts/compare_metrics.py）。
CD_PLATFORM = (-0.626, 0.308, -0.224)
CD_BENCH = (-0.10, 0.0, 0.0)

SOIL_PRESETS = {"fig1": {"soil": SOIL_FIG1, "load": 80.0, "tag": "fig1"}}
SOIL_T1 = SOIL_PRESETS[args_cli.soil]["soil"]

# 经验系数覆盖（不改 terramechanics.py 的平台参数，仅本进程生效）
_cd = CD_PLATFORM if args_cli.cd == "platform" else CD_BENCH
import terramechanics as _tm
_tm.PARAM["c_d1"], _tm.PARAM["c_d2"], _tm.PARAM["c_d3"] = _cd

# 滑转率扫掠 0-0.7（0.05 步长，对齐 MARSSIM 图 x 轴从 0 到最大）
SLIP_PHASES = [round(0.05 * i, 2) for i in range(15)]

W_LOAD = args_cli.load if args_cli.load is not None else SOIL_PRESETS[args_cli.soil]["load"]
M_WHEEL = args_cli.wheel_mass
M_BASE = W_LOAD / G_EARTH - M_WHEEL            # 台架配重质量
OMEGA_CMD = args_cli.omega

####################
## 最小 URDF（carriage + wheel，无碰撞几何，力全部走公式模型）
####################

IYY = 0.5 * M_WHEEL * WHEEL["r"] ** 2
IXX = (1.0 / 12.0) * M_WHEEL * (3.0 * WHEEL["r"] ** 2 + WHEEL["h"] ** 2)
URDF_TEMPLATE = f"""<?xml version="1.0"?>
<robot name="wh12_testbed">
  <link name="carriage">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{M_BASE:.4f}"/>
      <inertia ixx="0.05" iyy="0.05" izz="0.05" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.12" rpy="0 0 0"/>
      <geometry><box size="0.10 0.10 0.24"/></geometry>
      <material name="grey"><color rgba="0.6 0.6 0.65 1"/></material>
    </visual>
  </link>
  <link name="wheel">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{M_WHEEL:.4f}"/>
      <inertia ixx="{IXX:.6f}" iyy="{IYY:.6f}" izz="{IXX:.6f}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="1.5707963 0 0"/>
      <geometry><cylinder radius="{WHEEL['r']}" length="{WHEEL['b']}"/></geometry>
      <material name="dark"><color rgba="0.25 0.25 0.28 1"/></material>
    </visual>
  </link>
  <joint name="wheel_joint" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="carriage"/>
    <child link="wheel"/>
    <axis xyz="0 1 0"/>
    <limit effort="100000" velocity="1000" lower="-1000000" upper="1000000"/>
    <dynamics damping="0" friction="0"/>
  </joint>
</robot>
"""

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
urdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wh12_assets")
os.makedirs(urdf_dir, exist_ok=True)
urdf_path = os.path.join(urdf_dir, "wh12_testbed.urdf")
with open(urdf_path, "w") as f:
    f.write(URDF_TEMPLATE)


####################
## 场景
####################

@configclass
class Wh12SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/wh12",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf_path,
            fix_base=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="velocity",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=1000.0),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, WHEEL["r"]),
            joint_pos={"wheel_joint": 0.0},
            joint_vel={"wheel_joint": 0.0},
        ),
        actuators={
            "wheel": ImplicitActuatorCfg(
                joint_names_expr=["wheel_joint"],
                effort_limit=100000.0, velocity_limit=1000.0,
                stiffness=0.0, damping=1000.0,
            ),
        },
    )


####################
## 主循环
####################

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot: Articulation = scene["robot"]
    device = robot.device
    sim_dt = sim.get_physics_dt()
    wheel_body_id = robot.find_bodies("wheel")[0][0]
    wheel_joint_id = robot.find_joints("wheel_joint")[0][0]

    # 台架 PI（丝杠水平推力），稳态时 rail 力 = -Fdp；轮体只受公式模型力 + 土壤反力矩
    kp, ki, i_clamp = 1000.0, 800.0, 200.0

    # 初始沉陷：静态解 (s=0, Fn=W)，起步即平衡避免自由落体瞬态
    z_eq = solve_static_sinkage(W_LOAD, WHEEL["r"], WHEEL["b"], h=WHEEL["h"], soil=SOIL_T1)
    print(f"[INFO]: W={W_LOAD:.0f}N m_total={M_BASE + M_WHEEL:.2f}kg z_eq={z_eq * 1000:.2f}mm")

    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] = WHEEL["r"] - z_eq
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    scene.reset()

    tm_state = {"sinkage_prev": torch.full((1, 1), z_eq, device=device)}
    quat_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    settle_steps = int(args_cli.settle / sim_dt)
    measure_steps = int(args_cli.measure / sim_dt)
    rows = []
    phase_summary = []

    print(f"[INFO]: phases (settle {args_cli.settle}s + measure {args_cli.measure}s): {SLIP_PHASES}")
    sim_time = 0.0
    for phase_i, s_cmd in enumerate(SLIP_PHASES):
        omega_cmd = OMEGA_CMD
        v_cmd = (1.0 - s_cmd) * R_EFF * OMEGA_CMD
        int_x = 0.0
        for step in range(settle_steps + measure_steps):
            omega_cur = robot.data.joint_vel[:, wheel_joint_id].unsqueeze(1)
            wheel_vel = robot.data.body_lin_vel_w[:, wheel_body_id, :]
            wheel_pos = robot.data.body_pos_w[:, wheel_body_id, :]

            sinkage = torch.clamp(WHEEL["r"] - wheel_pos[:, 2:3], min=0.0)
            v_fwd = wheel_vel[:, 0:1]
            vz = wheel_vel[:, 2:3]

            tm_out = compute_terramechanics(
                sinkage, omega_cur, v_fwd, vz,
                WHEEL, fn_ave=W_LOAD, state=tm_state, soil=SOIL_T1,
            )
            tm_state = tm_out["state"]

            e_v = v_cmd - robot.data.root_lin_vel_w[:, 0:1]
            int_x = float(np.clip(int_x + e_v[0, 0].item() * sim_dt, -i_clamp / ki, i_clamp / ki))
            f_rail = kp * e_v + ki * int_x

            forces = torch.zeros(1, robot.num_bodies, 3, device=device)
            torques = torch.zeros(1, robot.num_bodies, 3, device=device)
            forces[:, wheel_body_id, 0] = tm_out["force_forward"][:, 0]
            forces[:, wheel_body_id, 2] = tm_out["force_z"][:, 0]
            mr = float(tm_out["Mr"][0, 0])
            omega_sign = 1.0 if float(omega_cur[0, 0]) >= 0.0 else -1.0
            torques[:, wheel_body_id, 1] = -mr * omega_sign   # 土壤滚动阻力矩（cpp: MY_Local=-Mr）
            forces[:, 0, 0] = f_rail[:, 0]
            robot.set_external_force_and_torque(forces, torques, is_global=True)

            robot.set_joint_velocity_target(torch.full_like(omega_cur, omega_cmd))

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            sim_time += sim_dt

            # 轨道约束：base 只保留 x/z 平动
            rs = robot.data.root_state_w.clone()
            rs[:, 1] = 0.0
            rs[:, 3:7] = quat_id
            rs[:, 8] = 0.0
            rs[:, 10:13] = 0.0
            robot.write_root_pose_to_sim(rs[:, :7])
            robot.write_root_velocity_to_sim(rs[:, 7:])

            if step >= settle_steps:
                rows.append((
                    phase_i, s_cmd, sim_time, float(tm_out["slip"][0, 0]),
                    float(tm_out["sinkage"][0, 0]) * 1000, float(tm_out["force_forward"][0, 0]),
                    mr, float(tm_out["Fn"][0, 0]), float(tm_out["force_z"][0, 0]),
                    float(f_rail[0, 0]), float(omega_cur[0, 0]), float(v_fwd[0, 0]),
                    float(robot.data.applied_torque[0, wheel_joint_id]),
                ))
                if (step - settle_steps) % 100 == 0:
                    print(f"  [s={s_cmd:+.4f}] t={sim_time:6.1f} z={rows[-1][4]:6.2f}mm "
                          f"Fdp={rows[-1][5]:7.2f}N Mr={mr:7.2f}Nm slip={rows[-1][3]:6.3f} "
                          f"Fn={rows[-1][7]:6.1f}N rail={rows[-1][9]:7.2f}N")

        win = np.array([r for r in rows if r[0] == phase_i], dtype=float)
        mean, std = win[:, 4:13].mean(axis=0), win[:, 4:13].std(axis=0)
        inter = np.empty(18)
        inter[0::2], inter[1::2] = mean, std   # 与表头交错对齐: v, v_std, ...
        phase_summary.append((s_cmd, win[:, 3].mean(), *inter))
        print(f"[PHASE {phase_i}] s_cmd={s_cmd:+.4f} s_ach={win[:, 3].mean():+.4f} "
              f"z={mean[0]:.2f}±{std[0]:.2f}mm Fdp={mean[1]:.2f}±{std[1]:.2f}N "
              f"Mr={mean[2]:.2f}±{std[2]:.2f}Nm Fn={mean[3]:.1f}N")

    # 落盘（repo 根的 docs/experiments/exp13_package/data/）
    tag = args_cli.tag or f"{SOIL_PRESETS[args_cli.soil]['tag']}_{args_cli.cd}"
    out_path = os.path.join(
        REPO_ROOT, args_cli.out
        if args_cli.out else f"docs/experiments/exp13_package/data/wh12_slip_sweep_{tag}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = ("s_cmd,s_ach,t,slip,sinkage_mm,Fdp_N,Mr_Nm,Fn_N,Fz_N,rail_N,omega,v_fwd,joint_torque")
    np.savetxt(out_path, np.array(rows, dtype=float), delimiter=",", header=header, comments="")
    summary_path = out_path.replace(".csv", "_summary.csv")
    s_header = ("s_cmd,s_ach,z_mm,z_std,Fdp_N,Fdp_std,Mr_Nm,Mr_std,Fn_N,Fn_std,Fz_N,Fz_std,"
                "rail_N,rail_std,omega,omega_std,v,v_std,tau,tau_std")
    np.savetxt(summary_path, np.array(phase_summary, dtype=float), delimiter=",", header=s_header, comments="")
    print(f"[INFO]: saved {out_path} ({len(rows)} rows) and {summary_path}")


####################
## main
####################

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device, gravity=(0.0, 0.0, -G_EARTH))
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.2, 1.2, 0.6], target=[0.0, 0.0, 0.0])

    scene = InteractiveScene(Wh12SceneCfg(num_envs=1, env_spacing=4.0))
    sim.reset()

    print("[INFO]: Wh12 single-wheel testbed ready")
    run_simulator(sim, scene)

    simulation_app.close()


if __name__ == "__main__":
    main()
