# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Hunter 火星地形硬地仿真
####################
## 链路：ROS 控制 -> Hunter Ackermann 控制 -> PhysX 地形/岩石接触 -> height_scan/wheel_sensor
## 目标：验证 Hunter 在有碰撞火星地形上的控制、传感器和 ROS 发布
####################

import argparse

from isaaclab.app import AppLauncher


####################
## 启动参数
####################

parser = argparse.ArgumentParser(description="Spawning an interactive robot scene with ROS integration.")
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

cfg, args_cli = load_sim_config(args_cli, "hunter_config.yaml")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


####################
## Isaac / ROS
####################

import torch
import numpy as np
import rclpy

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg

from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from physplanet_assets.hunter import HUNTER_SUS_CFG

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")

from hunter.hunter_control import HunterController
from hunter.hunter_control import WHEEL_RADIUS as HUNTER_WHEEL_RADIUS, resolve_hunter_joint_map
from hunter.hunter_sensor import (
    publish_height_scan_array,
    publish_height_scan_image,
    publish_semantic_visualizations,
    publish_wheel_sensor_data,
    setup_all_camera_publishers_from_scene,
    WHEEL_NAMES as HUNTER_WHEEL_NAMES,
)
from urdf_tf_publisher import URDFTFPublisher
from terramechanics import bilinear_ground_z
from gt_utils import load_gt_maps, make_world_grid, compute_patches, SOIL_KEYS
from rover_sensor import publish_gt_patch

HEIGHT_OFFSET = 0.26878

####################
## 场景
####################

@configclass
class RoverMarsSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="usd",
        collision_group=-1,
        usd_path=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "terrain_generation", "terrain", "terrain2", "terrain_only.usd"
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
    # 补光 SphereLight（参照 RLPhysPlanet）
    sphere_light = AssetBaseCfg(
        prim_path="/World/SphereLight",
        spawn=sim_utils.SphereLightCfg(
            intensity=15000.0, radius=50, color_temperature=5500, enable_color_temperature=True
        ),
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
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    
    # 高度扫描
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

    # 4 轮 contact sensor（摩擦估计）
    # 注：GPU 管线不支持 mesh collider 的 contact filter，filter_prim_paths_expr 无效
    contact_re_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/hunter/(re_left_link)",
        update_period=0.0, history_length=6, debug_vis=False,
    )
    contact_fr_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/hunter/(fr_left_link)",
        update_period=0.0, history_length=6, debug_vis=False,
    )
    contact_re_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/hunter/(re_right_link)",
        update_period=0.0, history_length=6, debug_vis=False,
    )
    contact_fr_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/hunter/(fr_right_link)",
        update_period=0.0, history_length=6, debug_vis=False,
    )


####################
## 主循环
####################

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, controller: HunterController, tf_publisher: URDFTFPublisher, cfg: dict):
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    joint_map = resolve_hunter_joint_map(robot.data.joint_names)

    _spawn = cfg["spawn"]
    terrain_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "terrain_generation", "terrain", cfg["terrain"]["name"],
    )
    terrain_size = load_terrain_size(terrain_base, cfg["terrain"]["size"])
    terrain_offset_x = _spawn.get("offset_x")
    terrain_offset_y = _spawn.get("offset_y")
    terrain_offset_x = terrain_size / 2.0 if terrain_offset_x is None else terrain_offset_x
    terrain_offset_y = terrain_size / 2.0 if terrain_offset_y is None else terrain_offset_y
    spawn_height = _spawn.get("height", 15.0)
    ground_clearance = _spawn.get("ground_clearance", 0.3)
    print(f"[INFO]: Terrain size={terrain_size}m, spawn=({terrain_offset_x}, {terrain_offset_y}), height={spawn_height}")

    ## 1. spawn 到 terrain 中心，再用 height scanner 校正高度
    root_state = robot.data.default_root_state.clone()
    for i in range(scene.num_envs):
        root_state[i, 0] = scene.env_origins[i, 0] + terrain_offset_x
        root_state[i, 1] = scene.env_origins[i, 1] + terrain_offset_y
        root_state[i, 2] = scene.env_origins[i, 2] + spawn_height
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

    # 读取 height_scanner 获取每辆车下方的地形高度
    try:
        hs = scene["height_scanner"]
        ray_hits = hs.data.ray_hits_w
        for i in range(scene.num_envs):
            valid_hits = ray_hits[i, :, 2]
            valid_hits = valid_hits[valid_hits > -1e3]
            if len(valid_hits) > 0:
                ground_z = valid_hits.max().item()
            else:
                ground_z = scene.env_origins[i, 2]
            root_state[i, 2] = ground_z + ground_clearance
    except (KeyError, IndexError, AttributeError):
        # height_scanner 关闭时，用 heights.npy 双线性插值（同软地脚本）
        heights_t = torch.as_tensor(np.load(os.path.join(terrain_base, "heights.npy")),
                                    dtype=torch.float32, device=robot.device)
        gz = bilinear_ground_z(root_state[:, :2], heights_t, terrain_size)
        root_state[:, 2] = gz + ground_clearance

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    scene.reset()
    print("[INFO]: Robot spawned on terrain")

    count = 0
    num_envs = scene.num_envs
    sim_time = 0.0

    ## 2. 传感器发布配置
    features = cfg.get("features", {})
    do_height_scan = features.get("height_scan", True)
    use_raycast = features.get("height_scan_raycast", False)
    do_wheel_sensor = features.get("wheel_sensor", True)
    do_gt = features.get("gt_publish", True)
    gt_image = features.get("gt_image", False)
    gt_cfg = cfg.get("gt_scan", {"size": [5.0, 5.0], "resolution": 0.1})
    sd = cfg["sim_data"]
    height_scan_grid_shape = tuple(int(s / sd["resolution"]) + 1 for s in sd["size"])
    gt_grid_shape = tuple(int(s / gt_cfg.get("resolution", 0.1)) + 1 for s in gt_cfg.get("size", [5.0, 5.0]))
    gt_maps = load_gt_maps(terrain_base, robot.device)
    # 硬地 height_scan 查表用粗图 heights.npy（和碰撞 mesh 一致），启动时加载一次
    heights_t = torch.as_tensor(np.load(os.path.join(terrain_base, "heights.npy")),
                                dtype=torch.float32, device=robot.device)
    publish_steps = max(1, int(1.0 / (sim_dt * cfg["sim"]["pub_freq"])))

    contact_sensors = {}
    for wname in HUNTER_WHEEL_NAMES:
        try:
            contact_sensors[wname] = scene[f"contact_{wname}"]
        except KeyError:
            pass
    if contact_sensors:
        print(f"[INFO]: Contact sensors active: {list(contact_sensors.keys())}")
    cs_cfg = sd.get("contact_sensor", {})
    force_threshold = cs_cfg.get("force_threshold", 0.1)

    hunter_wheel_indices = {}
    jm = joint_map
    left_idx = jm["left_drive_indices"]
    right_idx = jm["right_drive_indices"]
    hunter_wheel_indices["re_left"] = left_idx[0]
    hunter_wheel_indices["fr_left"] = left_idx[1]
    hunter_wheel_indices["re_right"] = right_idx[0]
    hunter_wheel_indices["fr_right"] = right_idx[1]

    ## 3. ROS 控制 + PhysX 硬地接触 + 传感器发布
    while simulation_app.is_running():
        rclpy.spin_once(controller, timeout_sec=0.0)

        joint_pos_target, joint_vel_target = controller.compute_control_batch()

        robot.set_joint_velocity_target(joint_vel_target)
        robot.set_joint_position_target(joint_pos_target)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        sim_time += sim_dt  # 累积仿真时间
        count += 1

        # Publish sensor data (降频)
        if count % publish_steps == 0:
            tf_publisher.publish_tf_all_envs(sim_time)
            controller.publish_odom(sim_time)
            publish_semantic_visualizations(controller, scene, sim_time)
            if do_height_scan:
                if use_raycast:
                    try:
                        height_scanner = scene["height_scanner"]
                        for env_id in range(num_envs):
                            height_data = height_scanner.data.ray_hits_w[env_id]
                            if height_data is not None:
                                robot_z = height_scanner.data.pos_w[env_id, 2]
                                scan = robot_z - height_scanner.data.ray_hits_w[env_id, :, 2] - HEIGHT_OFFSET
                                scan = torch.nan_to_num(scan, nan=0.0, posinf=0.0, neginf=0.0)
                                publish_height_scan_array(controller, env_id, scan)
                                publish_height_scan_image(controller, env_id, scan, height_scan_grid_shape, sim_time=sim_time)
                    except (KeyError, IndexError, AttributeError):
                        pass
                else:
                    hs_xy = make_world_grid(robot.data.root_pos_w, robot.data.root_quat_w,
                                            sd["size"], sd["resolution"], robot.device)
                    patches = compute_patches(hs_xy, {"height": heights_t}, terrain_size)
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
                patches = compute_patches(gt_xy, gt_maps, terrain_size)
                for env_id in range(num_envs):
                    for name, patch in patches.items():
                        if name == "soil":
                            for i, key in enumerate(SOIL_KEYS):
                                publish_gt_patch(controller, env_id, patch[env_id, :, i], f"soil_{key}",
                                                 gt_grid_shape, sim_time=sim_time, publish_image=gt_image)
                        else:
                            publish_gt_patch(controller, env_id, patch[env_id], name, gt_grid_shape,
                                             sim_time=sim_time, publish_image=gt_image)

            # 轮子接触力 + slip + torque 发布
            if do_wheel_sensor and contact_sensors:
                for env_id in range(num_envs):
                    publish_wheel_sensor_data(
                        controller, contact_sensors, env_id,
                        robot=robot,
                        wheel_indices=hunter_wheel_indices,
                        wheel_radius=HUNTER_WHEEL_RADIUS,
                        sim_time=sim_time, force_threshold=force_threshold,
                    )


####################
## main
####################

def main():
    rclpy.init()

    sim_cfg = sim_utils.SimulationCfg(dt=cfg["sim"]["dt"], device=args_cli.device, gravity=(0.0, 0.0, -cfg["sim"]["gravity"]))
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[75.0, 75.0, 4.0], target=[50.0, 70.0, -2.0])

    scene_cfg = RoverMarsSceneCfg(num_envs=cfg["sim"]["num_envs"], env_spacing=cfg["sim"]["env_spacing"])

    # 根据 terrain 参数覆盖 USD 路径
    terrain_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "terrain_generation", "terrain", cfg["terrain"]["name"]
    )
    scene_cfg.terrain.usd_path = os.path.join(terrain_base, "terrain_only.usd")
    scene_cfg.obstacles.spawn.usd_path = os.path.join(terrain_base, "rocks_merged.usd")
    scene_cfg.hidden_terrain.spawn.usd_path = os.path.join(terrain_base, "terrain_merged.usd")

    # height_scan_raycast=false 时不建 raycaster（查表方式不需要）
    if not cfg.get("features", {}).get("height_scan_raycast", False):
        scene_cfg.height_scanner = None
    elif cfg["debug"]:
        scene_cfg.height_scanner.debug_vis = True

    # enable_cameras=false 时不创建相机
    if not args_cli.enable_cameras:
        scene_cfg.camera = None

    scene = InteractiveScene(scene_cfg)
    write_terrain_context("hunter", "hard", cfg["terrain"]["name"],
                          load_terrain_size(terrain_base, cfg["terrain"]["size"]), scene.env_origins)
    sim.reset()

    # 给所有轮子 collision mesh 加高摩擦
    from pxr import UsdPhysics
    import isaacsim.core.utils.stage as stage_utils
    _stage = stage_utils.get_current_stage()
    for prim in _stage.Traverse():
        path = str(prim.GetPath())
        if any(w in path for w in ["re_left_link", "re_right_link", "fr_left_link", "fr_right_link"]):
            link_name = path.split("/")[-1]
            if link_name not in ("fr_left_link", "fr_right_link", "re_left_link", "re_right_link"):
                continue
            for child in prim.GetChildren():
                for sub in child.GetAllChildren():
                    if sub.GetTypeName() == "Mesh":
                        mat = UsdPhysics.MaterialAPI.Apply(sub)
                        mat.CreateStaticFrictionAttr().Set(8.0)
                        mat.CreateDynamicFrictionAttr().Set(6.0)
                        mat.CreateRestitutionAttr().Set(0.1)
    print("[INFO]: Applied high-friction material to all wheels")

    robot = scene["robot"]
    controller = HunterController(num_envs=cfg["sim"]["num_envs"], robot=robot, device=robot.device)

    urdf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "physplanet_assets", "hunter_assets", "hunter_se_suspension_description.urdf"
    )
    tf_publisher = URDFTFPublisher(controller, urdf_path, robot, cfg["sim"]["num_envs"])

    # 相机发布可选，默认由启动参数控制。
    if args_cli.enable_cameras:
        setup_all_camera_publishers_from_scene(scene, freq=cfg["sim"]["pub_freq"])

    print(f"[INFO]: Setup complete ({cfg['sim']['num_envs']} envs, terrain={cfg['terrain']['name']})")

    run_simulator(sim, scene, controller, tf_publisher, cfg)

    controller.destroy_node()
    rclpy.shutdown()



if __name__ == "__main__":
    main()
    simulation_app.close()
