# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

# Hunter 软地仿真
####################
## 链路：ROS 控制 -> Hunter 运动学 -> Bekker-Wong 轮地力 -> 虚拟轮传感器
## 目标：单脚本看清 Hunter 光面轮软地仿真流程
####################

import argparse
import os
import sys

from isaaclab.app import AppLauncher


####################
## 启动参数
####################

parser = argparse.ArgumentParser(description="Hunter terramechanics test.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--terrain", type=str, default=None)
parser.add_argument("--debug", action="store_true", default=None)
parser.add_argument("--pub_freq", type=int, default=None)
parser.add_argument("--config", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from sim_utils import load_sim_config, load_terrain_size, write_terrain_context, terrain_dome_light_cfg

cfg, args_cli = load_sim_config(args_cli, "hunter_config.yaml")

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
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")

from physplanet_assets.hunter import HUNTER_SUS_CFG
from hunter.hunter_control import (
    HunterController,
    WHEEL_BODY_NAMES,
    WHEEL_LUG_H,
    WHEEL_ORDER,
    WHEEL_RADIUS,
    WHEEL_WIDTH,
    resolve_hunter_joint_map,
)
from hunter.hunter_sensor import (
    WHEEL_NAMES,
    publish_height_scan_array,
    publish_height_scan_image,
    publish_semantic_visualizations,
    publish_soft_wheel_sensor_data,
    setup_all_camera_publishers_from_scene,
)
from terramechanics import (
    SOIL,
    SOIL_KEYS,
    bilinear_ground_z,
    bilinear_sample,
    build_terramechanics_forces,
    setup_rover_terramechanics,
    solve_static_sinkage,
    steered_wheel_axes_w,
)
from gt_utils import load_gt_maps, make_world_grid, compute_patches
from rover_sensor import publish_gt_patch
from urdf_tf_publisher import URDFTFPublisher

HEIGHT_OFFSET = 0.26878


####################
## 场景
####################

@configclass
class HunterSoftSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="usd",
        collision_group=-1,
        usd_path="",
    )

    obstacles = AssetBaseCfg(
        prim_path="/World/terrain/obstacles",
        spawn=sim_utils.UsdFileCfg(visible=True, usd_path=""),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    hidden_terrain = AssetBaseCfg(
        prim_path="/World/terrain/hidden_terrain",
        spawn=sim_utils.UsdFileCfg(visible=False, usd_path=""),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=terrain_dome_light_cfg(sim_utils, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "terrain_generation", "terrain", cfg["terrain"]["name"])),
    )

    sphere_light = AssetBaseCfg(
        prim_path="/World/SphereLight",
        spawn=sim_utils.SphereLightCfg(intensity=15000.0, radius=50, color_temperature=5500, enable_color_temperature=True),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -50.0, 80.0)),
    )

    robot: ArticulationCfg = HUNTER_SUS_CFG.replace(prim_path="{ENV_REGEX_NS}/hunter")

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/hunter/base_link/hunter_link/rear_center_link/imu/zed2_base_link/zed2_cam",
        update_period=0.2,
        height=480,
        width=640,
        data_types=["rgb", "depth", "semantic_segmentation"],
        semantic_filter=["class"],
        colorize_semantic_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/hunter/base_link",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 10.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[5.0, 5.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/terrain/hidden_terrain"],
        max_distance=100.0,
    )


####################
## 地形与轮地力学
####################

def terrain_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "terrain_generation", "terrain", cfg["terrain"]["name"],
    )


def set_terrain_usd(scene_cfg):
    base = terrain_dir()
    scene_cfg.terrain.usd_path = os.path.join(base, "terrain_only_nocollide.usd")
    scene_cfg.obstacles.spawn.usd_path = os.path.join(base, "rocks_merged.usd")
    scene_cfg.hidden_terrain.spawn.usd_path = os.path.join(base, "terrain_merged.usd")


def load_terrain_assets(scene, robot):
    base = terrain_dir()
    # 软地：物理力 + height_scan 都用精图 height_gt_map（0.05m/px），fallback 到粗图 heights.npy
    height_gt_path = os.path.join(base, "height_gt_map.npy")
    if os.path.exists(height_gt_path):
        heights_t = torch.as_tensor(np.load(height_gt_path), dtype=torch.float32, device=robot.device)
        print(f"[INFO]: Using height_gt_map for physics+scan: shape={heights_t.shape}")
    else:
        heights_t = torch.as_tensor(np.load(os.path.join(base, "heights.npy")), dtype=torch.float32, device=robot.device)

    terrain_size = load_terrain_size(base, cfg["terrain"]["size"])

    soil_params_t = None
    soil_path = os.path.join(base, "soil_params_map.npy")
    if os.path.exists(soil_path):
        soil_params = np.load(soil_path)
        if soil_params.ndim != 3 or soil_params.shape[-1] != len(SOIL_KEYS):
            raise ValueError(f"soil_params_map shape must be (H,W,{len(SOIL_KEYS)}), got {soil_params.shape}")
        soil_params_t = torch.as_tensor(soil_params, dtype=torch.float32, device=robot.device)
        print(f"[INFO]: Loaded soil params map: {soil_path} (shape={soil_params_t.shape})")
    else:
        print(f"[INFO]: No soil params map (全局 SOIL, phi={SOIL['phi']})")

    sd = cfg["sim_data"]
    gt_cfg = cfg.get("gt_scan", {})
    return {
        "heights_t": heights_t,
        "soil_params_t": soil_params_t,
        "gt_maps": load_gt_maps(base, robot.device),
        "terrain_size": terrain_size,
        "env_origins": scene.env_origins,
        "num_envs": scene.num_envs,
        "height_scan_grid_shape": tuple(int(s / sd["resolution"]) + 1 for s in sd["size"]),
        "gt_grid_shape": tuple(int(s / gt_cfg.get("resolution", 0.1)) + 1 for s in gt_cfg.get("size", [5.0, 5.0])),
    }


def hunter_wheel_joint_ids(joint_map):
    left = joint_map["left_drive_indices"]
    right = joint_map["right_drive_indices"]
    return [left[0], left[1], right[0], right[1]]


def hunter_wheel_axes_w(robot, joint_map):
    steer = torch.zeros(robot.data.joint_pos.shape[0], len(WHEEL_ORDER), device=robot.device)
    steer[:, 1] = robot.data.joint_pos[:, joint_map["steer_left_idx"]]
    steer[:, 3] = robot.data.joint_pos[:, joint_map["steer_right_idx"]]
    return steered_wheel_axes_w(robot.data.root_quat_w, steer, robot.device)


def setup_terramechanics(robot, joint_map, num_envs):
    tm = setup_rover_terramechanics(
        robot,
        WHEEL_BODY_NAMES,
        hunter_wheel_joint_ids(joint_map),
        WHEEL_RADIUS,
        WHEEL_WIDTH,
        cfg["sim"]["gravity"],
        num_envs,
        wheel_lug_h=WHEEL_LUG_H,
    )
    print(
        f"[INFO]: Terramechanics ON: mass={tm['total_mass']:.1f}kg g={tm['gravity']} "
        f"fn_ave/wheel={tm['fn_ave']:.1f}N sinkage_eq={tm['sinkage_eq']:.4f}m"
    )
    return tm


def reset_robot_state(sim, scene, robot, root_state):
    sim_dt = sim.get_physics_dt()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    scene.reset()
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)


def spawn_robot(sim, scene, robot, tm, terrain):
    num_envs = terrain["num_envs"]
    env_origins = terrain["env_origins"]
    terrain_size = terrain["terrain_size"]
    heights_t = terrain["heights_t"]
    soil_params_t = terrain["soil_params_t"]
    device = robot.device

    root_state = robot.data.default_root_state.clone()
    for i in range(num_envs):
        root_state[i, 0] = env_origins[i, 0] + terrain_size / 2.0
        root_state[i, 1] = env_origins[i, 1] + terrain_size / 2.0
        root_state[i, 2] = env_origins[i, 2] + cfg["spawn"]["height"]
        root_state[i, 3] = 1.0
        root_state[i, 4:7] = 0.0
        root_state[i, 7:] = 0.0

    reset_robot_state(sim, scene, robot, root_state)
    ground_z = bilinear_ground_z(root_state[:, :2], heights_t, terrain_size)
    root_state[:, 2] = ground_z + cfg["spawn"]["ground_clearance"]
    reset_robot_state(sim, scene, robot, root_state)

    wheel_body_ids = tm["wheel_body_ids"]
    wheel_xy = robot.data.body_pos_w[:, wheel_body_ids, :2]
    wheel_xy_flat = wheel_xy.reshape(-1, 2)
    gz = bilinear_ground_z(wheel_xy_flat, heights_t, terrain_size).reshape(num_envs, tm["num_wheels"])

    if soil_params_t is not None:
        wheel_soil = bilinear_sample(soil_params_t, wheel_xy_flat, terrain_size).reshape(num_envs, tm["num_wheels"], len(SOIL_KEYS))
        soil_dict = dict(zip(SOIL_KEYS, wheel_soil.unbind(-1)))
        fn_ave_t = torch.full((num_envs, tm["num_wheels"]), tm["fn_ave"], device=device)
        sinkage_eq = solve_static_sinkage(fn_ave_t, tm["wheel_r"], tm["wheel_b"], soil=soil_dict)
        tm["tm_state"] = {"sinkage_prev": sinkage_eq.clone()}
    else:
        sinkage_eq = torch.full((num_envs, tm["num_wheels"]), tm["sinkage_eq"], device=device)

    wheel_z = robot.data.body_pos_w[:, wheel_body_ids, 2]
    root_state[:, 2] += (gz + tm["wheel_r"] - sinkage_eq - wheel_z).mean(dim=1)
    reset_robot_state(sim, scene, robot, root_state)
    print(f"[INFO]: Robot spawned, wheels pre-loaded to sinkage=[{float(sinkage_eq.min()):.4f},{float(sinkage_eq.max()):.4f}]m")


def compute_wheel_forces(robot, tm, terrain, joint_map):
    wheel_forward_w, wheel_right_w = hunter_wheel_axes_w(robot, joint_map)
    tm_out, dbg, forces = build_terramechanics_forces(robot, tm, terrain, wheel_forward_w, wheel_right_w)
    robot.set_external_force_and_torque(forces, torch.zeros_like(forces), body_ids=tm["wheel_body_ids"], is_global=True)
    return tm_out, dbg


def _publish_sensors(scene, controller, terrain, tm, tm_out, dbg, sim_time,
                     features, gt_cfg):
    robot = scene["robot"]
    num_envs = terrain["num_envs"]
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
                    publish_height_scan_image(controller, env_id, scan, terrain["height_scan_grid_shape"], sim_time=sim_time)
            except (KeyError, IndexError, AttributeError):
                pass
        else:
            sd = cfg["sim_data"]
            hs_xy = make_world_grid(robot.data.root_pos_w, robot.data.root_quat_w,
                                    sd["size"], sd["resolution"], robot.device)
            # heights_t 已加载为精图 height_gt_map（load_terrain_assets 里统一）
            patches = compute_patches(hs_xy, {"height": terrain["heights_t"]}, terrain_size)
            robot_z = robot.data.root_pos_w[:, 2:3]  # (E,1)
            scan = robot_z - patches["height"] - HEIGHT_OFFSET  # (E,N)
            scan = torch.nan_to_num(scan, nan=0.0, posinf=0.0, neginf=0.0)
            for env_id in range(num_envs):
                publish_height_scan_array(controller, env_id, scan[env_id])
                publish_height_scan_image(controller, env_id, scan[env_id], terrain["height_scan_grid_shape"], sim_time=sim_time)

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

    # wheel sensor
    if do_wheel_sensor:
        wheel_torque = robot.data.applied_torque[:, tm["wheel_joint_ids"]]
        for env_id in range(num_envs):
            publish_soft_wheel_sensor_data(
                controller, env_id, WHEEL_NAMES,
                dbg["forces_w"], tm_out, dbg,
                applied_torque=wheel_torque, sim_time=sim_time,
            )


####################
## 主循环
####################

def run_simulator(sim, scene, controller, tf_publisher):
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    joint_map = resolve_hunter_joint_map(robot.data.joint_names)
    terrain = load_terrain_assets(scene, robot)
    write_terrain_context("hunter", "soft", cfg["terrain"]["name"], terrain["terrain_size"], scene.env_origins)
    tm = setup_terramechanics(robot, joint_map, scene.num_envs)
    spawn_robot(sim, scene, robot, tm, terrain)

    features = cfg.get("features", {})
    gt_cfg = cfg.get("gt_scan", {"size": [5.0, 5.0], "resolution": 0.1})

    publish_steps = max(1, int(1.0 / (sim_dt * cfg["sim"]["pub_freq"])))
    print("\n" + "=" * 60)
    print("[INFO] Hunter terramechanics (soft-soil) test")
    print(f"  Environments: {scene.num_envs}, Terrain: {cfg['terrain']['name']}")
    print("  Ground collision: OFF (Bekker-Wong formula force)")
    print("=" * 60 + "\n")

    count = 0
    sim_time = 0.0
    while simulation_app.is_running():
        rclpy.spin_once(controller, timeout_sec=0.0)

        joint_pos_target, joint_vel_target = controller.compute_control_batch()
        robot.set_joint_position_target(joint_pos_target)
        robot.set_joint_velocity_target(joint_vel_target)

        tm_out, dbg = compute_wheel_forces(robot, tm, terrain, joint_map)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        sim_time += sim_dt

        if cfg["debug"] and (count < 50 or count % 10 == 0):
            pos = robot.data.root_pos_w[0]
            root_vxy = torch.linalg.norm(robot.data.root_lin_vel_w[0, :2])
            print(
                f"[{count:4d}] z={pos[2]:.3f} "
                f"ΣFn={float(tm_out['Fn'][0].sum()):.0f} "
                f"ΣFz={float(tm_out['force_z'][0].sum()):.0f}/{tm['total_mass'] * tm['gravity']:.0f} "
                f"ΣFwd={float(tm_out['force_forward'][0].sum()):.0f} "
                f"ΣFside={float(dbg['side_drag'][0].sum()):.0f} "
                f"vxy={float(root_vxy):.2f} sink={float(tm_out['sinkage'][0].mean()):.4f}"
            )

        if count % publish_steps == 0:
            tf_publisher.publish_tf_all_envs(sim_time)
            controller.publish_odom(sim_time)
            publish_semantic_visualizations(controller, scene, sim_time)
            _publish_sensors(scene, controller, terrain, tm, tm_out, dbg, sim_time,
                             features, gt_cfg)
        count += 1


####################
## main
####################

def main():
    rclpy.init()

    sim_cfg = sim_utils.SimulationCfg(
        dt=cfg["sim"]["dt"],
        device=args_cli.device,
        gravity=(0.0, 0.0, -cfg["sim"]["gravity"]),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[75.0, 75.0, 4.0], target=[50.0, 70.0, -2.0])

    scene_cfg = HunterSoftSceneCfg(num_envs=cfg["sim"]["num_envs"], env_spacing=cfg["sim"]["env_spacing"])
    set_terrain_usd(scene_cfg)
    features = cfg.get("features", {})
    use_raycast = features.get("height_scan_raycast", False)
    if use_raycast:
        scene_cfg.height_scanner.debug_vis = bool(cfg["debug"])
    else:
        scene_cfg.height_scanner = None
    if not args_cli.enable_cameras:
        scene_cfg.camera = None

    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    controller = HunterController(num_envs=cfg["sim"]["num_envs"], robot=robot, device=robot.device)
    urdf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "physplanet_assets", "hunter_assets", "hunter_se_suspension_description.urdf",
    )
    tf_publisher = URDFTFPublisher(controller, urdf_path, robot, cfg["sim"]["num_envs"])

    if args_cli.enable_cameras:
        setup_all_camera_publishers_from_scene(scene, freq=cfg["sim"]["pub_freq"])

    run_simulator(sim, scene, controller, tf_publisher)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
