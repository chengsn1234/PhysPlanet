# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 祝融号软地仿真
####################
## 链路：ROS 控制 -> 祝融号 6WD/6WS 控制 -> Bekker-Wong 轮地力 -> 虚拟轮传感器
## 目标：单脚本看清祝融号软地仿真流程；地面无碰撞，岩石碰撞保留
####################

import argparse
from isaaclab.app import AppLauncher


####################
## 启动参数
####################

parser = argparse.ArgumentParser(description="Zhurong terramechanics (soft-soil) test.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--terrain", type=str, default=None)
parser.add_argument("--debug", action="store_true", default=None)
parser.add_argument("--pub_freq", type=int, default=None)
parser.add_argument("--config", type=str, default=None, help="YAML config path")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from sim_utils import load_sim_config, load_terrain_size, write_terrain_context, terrain_dome_light_cfg

cfg, args_cli = load_sim_config(args_cli, "zhurong_config.yaml")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


####################
## Isaac / ROS
####################

import numpy as np
import torch
import rclpy
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")

from physplanet_assets.zhurong import ZHURONG_CFG
from zhurong.zhurong_control import ZhurongController
from zhurong.zhurong_sensor import (
    publish_height_scan_array,
    publish_height_scan_image,
    publish_semantic_visualizations,
    publish_soft_wheel_sensor_data,
    publish_wheel_sensor_data,
    setup_all_camera_publishers_from_scene,
    WHEEL_NAMES,
)
from zhurong.zhurong_control import WHEEL_BODY_NAMES, WHEEL_LUG_H, WHEEL_ORDER
from urdf_tf_publisher import URDFTFPublisher
from terramechanics import solve_static_sinkage, SOIL, SOIL_KEYS
from terramechanics import (
    bilinear_ground_z,
    bilinear_sample,
    build_terramechanics_forces,
    setup_rover_terramechanics,
    steered_wheel_axes_w,
)
from gt_utils import load_gt_maps, make_world_grid, compute_patches
from rover_sensor import publish_gt_patch

HEIGHT_OFFSET = 0.40


####################
## 场景
####################

@configclass
class ZhurongMarsSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="usd",
        collision_group=-1,
        usd_path=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "terrain_generation", "terrain", "terrain2", "terrain_only_nocollide.usd"
        ),
    )

    obstacles = AssetBaseCfg(
        prim_path="/World/terrain/obstacles",
        spawn=sim_utils.UsdFileCfg(
            visible=True,
            usd_path=os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "terrain_generation", "terrain", "terrain2", "rocks_merged.usd"
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    hidden_terrain = AssetBaseCfg(
        prim_path="/World/terrain/hidden_terrain",
        spawn=sim_utils.UsdFileCfg(
            visible=False,
            usd_path=os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "terrain_generation", "terrain", "terrain2", "terrain_merged.usd"
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=terrain_dome_light_cfg(sim_utils, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "terrain_generation", "terrain", cfg["terrain"]["name"])),
    )

    # 补光
    sphere_light = AssetBaseCfg(
        prim_path="/World/SphereLight",
        spawn=sim_utils.SphereLightCfg(
            intensity=5000.0, radius=50.0,
            color_temperature=5500.0, enable_color_temperature=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 80.0)),
    )

    robot: ArticulationCfg = ZHURONG_CFG.replace(prim_path="{ENV_REGEX_NS}/zhurong_mars_rover")

    camera_nav = CameraCfg(
        prim_path="{ENV_REGEX_NS}/zhurong_mars_rover/ZEDNav_Link/zed_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "depth", "semantic_segmentation"],
        semantic_filter=["class"],
        colorize_semantic_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 15.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    obsf_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/zhurong_mars_rover/camera_obs_F_link/zed_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "depth", "semantic_segmentation"],
        semantic_filter=["class"],
        colorize_semantic_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 8.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/zhurong_mars_rover/base_link",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 10.0)),
        ray_alignment="base",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[10.0, 10.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/terrain/hidden_terrain"],
        max_distance=100.0,
    )

    # 6 轮 contact sensor（摩擦估计）
    contact_front_L = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/zhurong_mars_rover/(front_wheel_L)",
        update_period=0.0, history_length=6, debug_vis=False,
    )
    contact_front_R = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/zhurong_mars_rover/(front_wheel_R)",
        update_period=0.0, history_length=6, debug_vis=False,
    )
    contact_middle_L = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/zhurong_mars_rover/(middle_wheel_L)",
        update_period=0.0, history_length=6, debug_vis=False,
    )
    contact_middle_R = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/zhurong_mars_rover/(middle_wheel_R)",
        update_period=0.0, history_length=6, debug_vis=False,
    )
    contact_back_L = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/zhurong_mars_rover/(back_wheel_L)",
        update_period=0.0, history_length=6, debug_vis=False,
    )
    contact_back_R = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/zhurong_mars_rover/(back_wheel_R)",
        update_period=0.0, history_length=6, debug_vis=False,
    )


####################
## 地形接触辅助
####################

def apply_wheel_friction():
    from pxr import UsdPhysics
    import isaacsim.core.utils.stage as stage_utils
    _stage = stage_utils.get_current_stage()

    applied_count = 0
    for prim in _stage.Traverse():
        path = str(prim.GetPath())
        if "/collisions/" not in path:
            continue
        for wheel_name in WHEEL_BODY_NAMES:
            if wheel_name in path:
                mat = UsdPhysics.MaterialAPI.Apply(prim)
                mat.CreateStaticFrictionAttr().Set(8.0)
                mat.CreateDynamicFrictionAttr().Set(6.0)
                mat.CreateRestitutionAttr().Set(0.1)
                applied_count += 1
                break
    print(f"[INFO]: Applied high-friction material to {applied_count} wheel collision prims")


####################
## 地形与轮地力学
####################

def load_terrain_assets(cfg, scene, robot):
    sd = cfg["sim_data"]
    lm_size = sd["size"]
    lm_res = sd["resolution"]
    height_scan_grid_shape = tuple(int(s / lm_res) + 1 for s in lm_size)

    terrain_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "terrain_generation", "terrain", cfg["terrain"]["name"]
    )

    # 地面高程：软地物理力 + height_scan 都用精图 height_gt_map，fallback 到粗图 heights.npy
    height_gt_path = os.path.join(terrain_base, "height_gt_map.npy")
    if os.path.exists(height_gt_path):
        heights_t = torch.as_tensor(np.load(height_gt_path), dtype=torch.float32, device=robot.device)
        print(f"[INFO]: Using height_gt_map for physics+scan: shape={heights_t.shape}")
    else:
        heights_path = os.path.join(terrain_base, "heights.npy")
        if not os.path.exists(heights_path):
            raise FileNotFoundError(f"heights.npy not found: {heights_path}，需重生成地形")
        heights_t = torch.as_tensor(np.load(heights_path), dtype=torch.float32, device=robot.device)

    terrain_size = load_terrain_size(terrain_base, cfg["terrain"]["size"])

    # contact sensor dict
    contact_sensors = {}
    cs_cfg = sd["contact_sensor"]
    if cs_cfg.get("enabled", True):
        for wname in WHEEL_NAMES:
            try:
                contact_sensors[wname] = scene[f"contact_{wname}"]
            except KeyError:
                print(f"[WARN]: Contact sensor contact_{wname} not found in scene")
        if contact_sensors:
            print(f"[INFO]: Contact sensors enabled: {list(contact_sensors.keys())}")
    force_threshold = cs_cfg.get("force_threshold", 0.1)

    # 分层土参数图（H,W,7），无则 None（W1 平地无 soil，回落全局 SOIL）
    soil_map_path = os.path.join(terrain_base, "soil_params_map.npy")
    if os.path.exists(soil_map_path):
        soil_params = np.load(soil_map_path)
        if soil_params.ndim != 3 or soil_params.shape[-1] != len(SOIL_KEYS):
            raise ValueError(f"soil_params_map shape must be (H,W,{len(SOIL_KEYS)}), got {soil_params.shape}: {soil_map_path}")
        soil_params_t = torch.as_tensor(soil_params, dtype=torch.float32, device=robot.device)
        print(f"[INFO]: Loaded soil params map: {soil_map_path} (shape={soil_params_t.shape})")
    else:
        soil_params_t = None
        print(f"[INFO]: No soil params map (全局 SOIL, phi={SOIL['phi']})")

    gt_cfg = cfg.get("gt_scan", {})
    return {
        "heights_t": heights_t,
        "soil_params_t": soil_params_t,
        "gt_maps": load_gt_maps(terrain_base, robot.device),
        "contact_sensors": contact_sensors,
        "terrain_size": terrain_size,
        "height_scan_grid_shape": height_scan_grid_shape,
        "gt_grid_shape": tuple(int(s / gt_cfg.get("resolution", 0.1)) + 1 for s in gt_cfg.get("size", [10.0, 10.0])),
        "force_threshold": force_threshold,
        "env_origins": scene.env_origins,
        "num_envs": scene.num_envs,
    }


def setup_terramechanics(robot, controller, cfg, num_envs):
    print(f"[INFO]: body_names: {robot.body_names}")
    wheel_joint_ids = [controller.wheel_indices[k] for k in WHEEL_ORDER]
    steer_joint_ids = [controller.steer_indices[k] for k in WHEEL_ORDER]
    tm = setup_rover_terramechanics(
        robot,
        WHEEL_BODY_NAMES,
        wheel_joint_ids,
        controller.WHEEL_RADIUS,
        controller.WHEEL_WIDTH,
        cfg["sim"]["gravity"],
        num_envs,
        wheel_lug_h=WHEEL_LUG_H,
    )
    tm["steer_joint_ids"] = steer_joint_ids
    print(
        f"[INFO]: Terramechanics ON: mass={tm['total_mass']:.1f}kg g={tm['gravity']} "
        f"fn_ave/wheel={tm['fn_ave']:.1f}N sinkage_eq={tm['sinkage_eq']:.4f}m"
    )
    return tm


def spawn_robot(sim, scene, robot, tm, terrain, cfg):
    sim_dt = sim.get_physics_dt()
    spawn_cfg = cfg["spawn"]
    num_envs = terrain["num_envs"]
    num_wheels = tm["num_wheels"]
    wheel_body_ids = tm["wheel_body_ids"]
    wheel_r = tm["wheel_r"]
    wheel_b = tm["wheel_b"]
    fn_ave = tm["fn_ave"]
    sinkage_eq = tm["sinkage_eq"]
    heights_t = terrain["heights_t"]
    terrain_size = terrain["terrain_size"]
    soil_params_t = terrain["soil_params_t"]
    env_origins = terrain["env_origins"]
    device = robot.device

    ## 1. 首次 spawn 到 terrain 中心
    root_state = robot.data.default_root_state.clone()
    for i in range(num_envs):
        root_state[i, 0] = env_origins[i, 0] + terrain_size / 2.0   # 自动 terrain 中心（固定 offset 50 在小 terrain 超界）
        root_state[i, 1] = env_origins[i, 1] + terrain_size / 2.0
        root_state[i, 2] = env_origins[i, 2] + spawn_cfg["height"]
        root_state[i, 3] = 1.0
        root_state[i, 4:7] = 0.0
        root_state[i, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    scene.reset()
    for _ in range(5):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    ## 2. 粗放：用 height map 查 base 处地面高度
    ground_z_spawn = bilinear_ground_z(root_state[:, :2], heights_t, terrain_size)
    root_state[:, 2] = ground_z_spawn + spawn_cfg["ground_clearance"]
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    ## 3. 校正：轮子放到静态下陷位置，首帧 Fn 接近 mg
    wheel_z_cur = robot.data.body_pos_w[:, wheel_body_ids, 2]
    wheel_xy = robot.data.body_pos_w[:, wheel_body_ids, :2]
    wheel_xy_flat = wheel_xy.reshape(-1, 2)
    gz_cur = bilinear_ground_z(wheel_xy_flat, heights_t, terrain_size).reshape(num_envs, num_wheels)
    if soil_params_t is not None:
        # 分层土：per-wheel sinkage_eq（不同土静态下陷不同），tm_state 同步 per-wheel
        spawn_soil = bilinear_sample(soil_params_t, wheel_xy_flat, terrain_size).reshape(num_envs, num_wheels, 7)
        soil_dict = dict(zip(SOIL_KEYS, spawn_soil.unbind(-1)))
        fn_ave_t = torch.full((num_envs, num_wheels), fn_ave, device=device)
        sinkage_eq_pw = solve_static_sinkage(fn_ave_t, wheel_r, wheel_b, soil=soil_dict)   # (E,6)
        tm["tm_state"] = {"sinkage_prev": sinkage_eq_pw.clone()}
    else:
        sinkage_eq_pw = sinkage_eq
    desired_wheel_z = gz_cur + wheel_r - sinkage_eq_pw
    root_state[:, 2] += (desired_wheel_z - wheel_z_cur).mean(dim=1)

    ## 4. 姿态贴合：6 轮地面点拟合平面，base +Z 对齐法线
    A = torch.cat([wheel_xy, torch.ones(num_envs, num_wheels, 1, device=device)], dim=-1)
    abc = torch.linalg.lstsq(A, gz_cur.unsqueeze(-1)).solution[:, :, 0]   # (E,3): a,b,c
    nx, ny = -abc[:, 0], -abc[:, 1]
    nz = torch.ones_like(nx)
    nL = torch.sqrt(nx * nx + ny * ny + nz * nz)
    qw = torch.sqrt(torch.clamp((1.0 + nz / nL) / 2.0, min=1e-8))         # base +Z→法线 最小旋转
    root_state[:, 3] = qw
    root_state[:, 4] = (-ny / nL) / (2.0 * qw)
    root_state[:, 5] = (nx / nL) / (2.0 * qw)
    root_state[:, 6] = 0.0

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    scene.reset()
    sinkage_spawn = sinkage_eq_pw if torch.is_tensor(sinkage_eq_pw) else torch.full((num_envs, num_wheels), sinkage_eq, device=device)
    print(f"[INFO]: Robot spawned, wheels pre-loaded to sinkage=[{float(sinkage_spawn.min()):.4f},{float(sinkage_spawn.max()):.4f}]m")


def compute_wheel_forces(robot, tm, terrain, controller):
    wheel_body_ids = tm["wheel_body_ids"]
    steer_joint_ids = tm["steer_joint_ids"]
    steer = robot.data.joint_pos[:, steer_joint_ids]
    wheel_forward_w, wheel_right_w = steered_wheel_axes_w(robot.data.root_quat_w, steer, robot.device)

    tm_out, dbg, forces = build_terramechanics_forces(robot, tm, terrain, wheel_forward_w, wheel_right_w)
    robot.set_external_force_and_torque(forces, torch.zeros_like(forces), body_ids=wheel_body_ids, is_global=True)
    return tm_out, dbg


def publish_wheel_sensors(scene, controller, terrain, tm, tm_out, dbg, num_envs, sim_time,
                          features, gt_cfg):
    robot = scene["robot"]
    contact_sensors = terrain["contact_sensors"]
    force_threshold = terrain["force_threshold"]
    height_scan_grid_shape = terrain["height_scan_grid_shape"]
    do_height_scan = features.get("height_scan", True)
    use_raycast = features.get("height_scan_raycast", False)
    do_wheel_sensor = features.get("wheel_sensor", True)
    do_gt = features.get("gt_publish", True)
    gt_image = features.get("gt_image", False)
    terrain_size = terrain["terrain_size"]

    # height_scan
    if do_height_scan:
        if use_raycast:
            try:
                height_scanner = scene["height_scanner"]
                for env_id in range(num_envs):
                    robot_z = height_scanner.data.pos_w[env_id, 2]
                    scan = robot_z - height_scanner.data.ray_hits_w[env_id, :, 2] - HEIGHT_OFFSET
                    scan = torch.nan_to_num(scan, nan=0.0, posinf=0.0, neginf=0.0)
                    publish_height_scan_array(controller, env_id, scan)
                    publish_height_scan_image(controller, env_id, scan, height_scan_grid_shape, sim_time=sim_time)
            except (KeyError, IndexError, AttributeError):
                pass
        else:
            sd = cfg["sim_data"]
            hs_xy = make_world_grid(robot.data.root_pos_w, robot.data.root_quat_w,
                                    sd["size"], sd["resolution"], robot.device)
            # heights_t 已加载为精图 height_gt_map（load_terrain_assets 里统一）
            patches = compute_patches(hs_xy, {"height": terrain["heights_t"]}, terrain_size)
            robot_z = robot.data.root_pos_w[:, 2:3]
            scan = robot_z - patches["height"] - HEIGHT_OFFSET
            scan = torch.nan_to_num(scan, nan=0.0, posinf=0.0, neginf=0.0)
            for env_id in range(num_envs):
                publish_height_scan_array(controller, env_id, scan[env_id])
                publish_height_scan_image(controller, env_id, scan[env_id], height_scan_grid_shape, sim_time=sim_time)

    # GT patch
    if do_gt:
        gt_xy = make_world_grid(robot.data.root_pos_w, robot.data.root_quat_w,
                                gt_cfg["size"], gt_cfg["resolution"], robot.device)
        patches = compute_patches(gt_xy, terrain["gt_maps"], terrain_size)
        for env_id in range(num_envs):
            for name, patch in patches.items():
                if name == "soil":
                    for i, key in enumerate(SOIL_KEYS):
                        publish_gt_patch(controller, env_id, patch[env_id, :, i], f"soil_{key}",
                                         terrain["gt_grid_shape"], sim_time=sim_time, publish_image=gt_image)
                else:
                    publish_gt_patch(controller, env_id, patch[env_id], name, terrain["gt_grid_shape"],
                                     sim_time=sim_time, publish_image=gt_image)

    if not do_wheel_sensor:
        return

    # 软地用公式力虚拟接触，话题格式保持和硬地一致。
    if tm_out is not None and dbg is not None:
        wheel_torque = robot.data.applied_torque[:, tm["wheel_joint_ids"]]
        for env_id in range(num_envs):
            publish_soft_wheel_sensor_data(
                controller, env_id, WHEEL_NAMES,
                dbg["forces_w"], tm_out, dbg,
                applied_torque=wheel_torque,
                sim_time=sim_time,
            )
        return

    # 回退路径：如果没有 tm_out，就发 contact sensor 原始数据。
    if contact_sensors:
        for env_id in range(num_envs):
            publish_wheel_sensor_data(
                controller, contact_sensors, env_id,
                robot=robot,
                wheel_indices=controller.wheel_indices,
                sim_time=sim_time, force_threshold=force_threshold,
            )


####################
## 主循环
####################

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, controller: ZhurongController, tf_publisher: URDFTFPublisher, cfg: dict):
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    num_envs = scene.num_envs

    ## 1. 加载地形数据和软地模型
    terrain = load_terrain_assets(cfg, scene, robot)
    write_terrain_context("zhurong", "soft", cfg["terrain"]["name"], terrain["terrain_size"], scene.env_origins)

    controller._init_joint_indices(robot)
    tm = setup_terramechanics(robot, controller, cfg, num_envs)
    total_mass = tm["total_mass"]
    gravity = tm["gravity"]

    ## 2. spawn 并预压到静态下陷
    spawn_robot(sim, scene, robot, tm, terrain, cfg)

    features = cfg.get("features", {})
    gt_cfg = cfg.get("gt_scan", {"size": [10.0, 10.0], "resolution": 0.1})

    publish_steps = max(1, int(1.0 / (sim_dt * cfg["sim"]["pub_freq"])))
    print("\n" + "=" * 60)
    print("[INFO] Zhurong terramechanics (soft-soil) test")
    print(f"  Joints: {robot.num_joints}, Bodies: {robot.num_bodies}")
    print(f"  Environments: {num_envs}, Terrain: {cfg['terrain']['name']}")
    print(f"  Ground collision: OFF (Bekker-Wong formula force)")
    print("=" * 60 + "\n")

    count = 0
    sim_time = 0.0
    while simulation_app.is_running():
        ## ROS 控制
        rclpy.spin_once(controller, timeout_sec=0.0)
        joint_pos_target, joint_vel_target = controller.compute_control_batch()
        robot.set_joint_position_target(joint_pos_target)
        robot.set_joint_velocity_target(joint_vel_target)

        ## 软地公式力替代地面碰撞
        tm_out, dbg = compute_wheel_forces(robot, tm, terrain, controller)

        ## 仿真步进
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        sim_time += sim_dt

        ## Debug 输出（含轮地力学闭环证据；前 50 步逐帧，之后每 10 步）
        if cfg["debug"] and (count < 50 or count % 10 == 0):
            tan_phi = tm_out["tan_phi"] if torch.is_tensor(tm_out["tan_phi"]) else torch.full_like(tm_out["Fn"], float(tm_out["tan_phi"]))
            pos = robot.data.root_pos_w[0]
            root_vxy = torch.linalg.norm(robot.data.root_lin_vel_w[0, :2])
            wz = float(dbg["wheel_pos_w"][0, :, 2].mean())
            gz = float(dbg["ground_z_t"][0].mean())
            print(f"[{count:4d}] z={pos[2]:.3f} wz={wz:.3f} gz={gz:.3f} "
                  f"ΣFn={float(tm_out['Fn'][0].sum()):.0f} ΣFz={float(tm_out['force_z'][0].sum()):.0f}/{total_mass * gravity:.0f} "
                  f"ΣFwd={float(tm_out['force_forward'][0].sum()):.0f} ΣFside={float(dbg['side_drag'][0].sum()):.0f} vxy={float(root_vxy):.2f} "
                  f"Fz={float(tm_out['force_z'][0, 0]):.0f} sink={float(tm_out['sinkage'][0].mean()):.4f} "
                  f"tanφ=[{float(tan_phi[0].min()):.2f},{float(tan_phi[0].max()):.2f}] "
                  f"slip={float(tm_out['slip'][0, 0]):.2f} Fwd0={float(tm_out['force_forward'][0, 0]):.1f} "
                  f"vf={float(dbg['v_fwd'][0, 0]):.2f} om={float(dbg['omega'][0, 0]):.2f} vz={float(dbg['vz'][0, 0]):.2f}")
            # 多 env 健康聚合：各 env 车在 terrain 不同世界位置 → 不同土 → ΣFn 应不同（min≠max）；mean≈mg
            if num_envs > 1:
                sum_fn = tm_out["Fn"].sum(dim=1)
                sink_env = tm_out["sinkage"].mean(dim=1)
                print(f"  [envs] ΣFn min={float(sum_fn.min()):.0f} max={float(sum_fn.max()):.0f} "
                      f"mean={float(sum_fn.mean()):.0f}/{total_mass * gravity:.0f} | "
                      f"sink min={float(sink_env.min()):.4f} max={float(sink_env.max()):.4f}")

        ## height_scan + GT patch + wheel_sensor + TF/odom 发布（降频）
        if count % publish_steps == 0:
            tf_publisher.publish_tf_all_envs(sim_time)
            controller.publish_odom(sim_time)
            publish_semantic_visualizations(controller, scene, sim_time)
            publish_wheel_sensors(scene, controller, terrain, tm, tm_out, dbg, num_envs, sim_time,
                                  features, gt_cfg)

        count += 1


####################
## main
####################

def main():
    rclpy.init()

    sim_cfg = sim_utils.SimulationCfg(dt=cfg["sim"]["dt"], device=args_cli.device, gravity=(0.0, 0.0, -cfg["sim"]["gravity"]))
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[75.0, 75.0, 4.0], target=[50.0, 70.0, -2.0])

    scene_cfg = ZhurongMarsSceneCfg(num_envs=cfg["sim"]["num_envs"], env_spacing=cfg["sim"]["env_spacing"])

    # 根据 terrain 参数覆盖 USD 路径
    terrain_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "terrain_generation", "terrain", cfg["terrain"]["name"]
    )
    scene_cfg.terrain.usd_path = os.path.join(terrain_base, "terrain_only_nocollide.usd")
    scene_cfg.obstacles.spawn.usd_path = os.path.join(terrain_base, "rocks_merged.usd")
    scene_cfg.hidden_terrain.spawn.usd_path = os.path.join(terrain_base, "terrain_merged.usd")

    # enable_cameras=false 时不创建相机
    if not args_cli.enable_cameras:
        scene_cfg.camera_nav = None
        scene_cfg.obsf_camera = None
    # height_scan_raycast=false 时不建 raycaster（查表方式不需要）
    if not cfg.get("features", {}).get("height_scan_raycast", False):
        scene_cfg.height_scanner = None

    scene = InteractiveScene(scene_cfg)

    sim.reset()

    apply_wheel_friction()

    print("[INFO]: Setup complete...")

    robot = scene["robot"]
    controller = ZhurongController(robot=robot, num_envs=cfg["sim"]["num_envs"], device=robot.device)

    urdf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "physplanet_assets", "zhurong_assets", "zhurong.urdf"
    )
    tf_publisher = URDFTFPublisher(controller, urdf_path, robot, cfg["sim"]["num_envs"])

    if args_cli.enable_cameras:
        setup_all_camera_publishers_from_scene(scene, freq=cfg["sim"]["pub_freq"])
        print(f"[INFO]: Camera publishers setup complete ({cfg['sim']['pub_freq']}Hz)")
    else:
        print("[INFO]: Cameras disabled, skipping camera publishers")

    print("[INFO]: Waiting for /cmd_vel commands...")

    run_simulator(sim, scene, controller, tf_publisher, cfg)

    controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
