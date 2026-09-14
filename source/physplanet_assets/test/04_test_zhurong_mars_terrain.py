# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 祝融号火星地形硬地仿真
####################
## 链路：ROS 控制 -> 祝融号 6WD/6WS 控制 -> PhysX 地形/岩石接触 -> height_scan/wheel_sensor
## 目标：验证祝融号在有碰撞火星地形上的控制、传感器和 ROS 发布
####################

import argparse
from isaaclab.app import AppLauncher


####################
## 启动参数
####################

parser = argparse.ArgumentParser(description="Zhurong rover on Mars terrain with ROS2 control.")
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

import torch
import numpy as np
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
    publish_wheel_sensor_data,
    publish_semantic_visualizations,
    setup_all_camera_publishers_from_scene,
    WHEEL_NAMES,
)
from urdf_tf_publisher import URDFTFPublisher
from terramechanics import bilinear_ground_z
from gt_utils import load_gt_maps, make_world_grid, compute_patches, SOIL_KEYS
from rover_sensor import publish_gt_patch

HEIGHT_OFFSET = 0.40

# 祝融号轮子link名称（用于摩擦配置）
ZHURONG_WHEEL_LINKS = [
    "front_wheel_L", "front_wheel_R",
    "middle_wheel_L", "middle_wheel_R",
    "back_wheel_L", "back_wheel_R"
]


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

    # obs-l camera (底盘左侧) — 暂时禁用
    # obsl_camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/zhurong_mars_rover/camera_obs_L_link/zed_camera",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["depth"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=12.0,
    #         focus_distance=400.0,
    #         horizontal_aperture=20.955,
    #         clipping_range=(0.1, 8.0)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )

    # # obs-r camera (底盘右侧) — 暂时禁用
    # obsr_camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/zhurong_mars_rover/camera_obs_R_link/zed_camera",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["depth"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=12.0,
    #         focus_distance=400.0,
    #         horizontal_aperture=20.955,
    #         clipping_range=(0.1, 8.0)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )

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
    # 注：GPU 管线不支持 mesh collider 的 contact filter，filter_prim_paths_expr 无效
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
        # 轮子 collision 是 cylinder，在 collisions/ 子目录下
        if "/collisions/" not in path:
            continue
        for wheel_name in ZHURONG_WHEEL_LINKS:
            if wheel_name in path:
                mat = UsdPhysics.MaterialAPI.Apply(prim)
                mat.CreateStaticFrictionAttr().Set(8.0)
                mat.CreateDynamicFrictionAttr().Set(6.0)
                mat.CreateRestitutionAttr().Set(0.1)
                applied_count += 1
                break
    print(f"[INFO]: Applied high-friction material to {applied_count} wheel collision prims")


####################
## 主循环
####################

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, controller: ZhurongController, tf_publisher: URDFTFPublisher, cfg: dict):
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    spawn_cfg = cfg["spawn"]
    terrain_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "terrain_generation", "terrain", cfg["terrain"]["name"],
    )
    terrain_size = load_terrain_size(terrain_base, cfg["terrain"]["size"])
    offset_x = spawn_cfg.get("offset_x")
    offset_y = spawn_cfg.get("offset_y")
    offset_x = terrain_size / 2.0 if offset_x is None else offset_x
    offset_y = terrain_size / 2.0 if offset_y is None else offset_y

    # 初始化 controller 的关节索引
    controller._init_joint_indices(robot)

    # 从 sim_data 配置推导 grid_shape
    sd = cfg["sim_data"]
    lm_size = sd["size"]
    lm_res = sd["resolution"]
    height_scan_grid_shape = tuple(int(s / lm_res) + 1 for s in lm_size)

    # 构建 contact sensor dict
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

    ## 1. spawn 到配置位置，再用 height scanner 校正高度
    root_state = robot.data.default_root_state.clone()
    for i in range(scene.num_envs):
        root_state[i, 0] = scene.env_origins[i, 0] + offset_x
        root_state[i, 1] = scene.env_origins[i, 1] + offset_y
        root_state[i, 2] = scene.env_origins[i, 2] + spawn_cfg["height"]
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

    # 读取 height_scanner 获取地形高度
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
            root_state[i, 2] = ground_z + spawn_cfg["ground_clearance"]
    except (KeyError, IndexError, AttributeError):
        # height_scanner 关闭时，用 heights.npy 双线性插值（同软地脚本）
        heights_t = torch.as_tensor(np.load(os.path.join(terrain_base, "heights.npy")),
                                    dtype=torch.float32, device=robot.device)
        gz = bilinear_ground_z(root_state[:, :2], heights_t, terrain_size)
        root_state[:, 2] = gz + spawn_cfg["ground_clearance"]

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    scene.reset()
    print("[INFO]: Robot spawned on terrain")

    count = 0
    num_envs = scene.num_envs
    sim_time = 0.0
    publish_steps = max(1, int(1.0 / (sim_dt * cfg["sim"]["pub_freq"])))
    print("\n" + "=" * 60)
    print("[INFO] Zhurong rover on Mars terrain")
    print(f"  Joints: {robot.num_joints}")
    print(f"  Environments: {num_envs}")
    print(f"  Terrain: {cfg['terrain']['name']}")
    print(f"  Height scan grid: {height_scan_grid_shape}")
    print(f"  ROS topics: /cmd_vel, /env_N/cmd_vel")
    print("=" * 60 + "\n")

    features = cfg.get("features", {})
    do_height_scan = features.get("height_scan", True)
    use_raycast = features.get("height_scan_raycast", False)
    do_wheel_sensor = features.get("wheel_sensor", True)
    do_gt = features.get("gt_publish", True)
    gt_image = features.get("gt_image", False)
    gt_cfg = cfg.get("gt_scan", {"size": [10.0, 10.0], "resolution": 0.1})
    gt_grid_shape = tuple(int(s / gt_cfg.get("resolution", 0.1)) + 1 for s in gt_cfg.get("size", [10.0, 10.0]))
    gt_maps = load_gt_maps(terrain_base, robot.device)
    # 硬地 height_scan 查表用粗图 heights.npy（和碰撞 mesh 一致），启动时加载一次
    heights_t = torch.as_tensor(np.load(os.path.join(terrain_base, "heights.npy")),
                                dtype=torch.float32, device=robot.device)

    while simulation_app.is_running():
        ## ROS 控制
        rclpy.spin_once(controller, timeout_sec=0.0)

        joint_pos_target, joint_vel_target = controller.compute_control_batch()

        robot.set_joint_position_target(joint_pos_target)
        robot.set_joint_velocity_target(joint_vel_target)

        ## PhysX 硬地接触
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        sim_time += sim_dt

        # Debug 输出
        if cfg["debug"] and count % 100 == 0:
            pos = robot.data.root_pos_w[0]
            vel_b = robot.data.root_lin_vel_b[0]  # 车体坐标系速度
            print(f"[{count}] pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), vel: {vel_b[0]:.2f} m/s")
            print(f"  cmd: linear={controller.linear_x[0]:.2f}, angular={controller.angular_z[0]:.2f}")

        if count % publish_steps == 0:
            ## TF/odom + height_scan + GT patch + wheel_sensor 发布（降频）
            tf_publisher.publish_tf_all_envs(sim_time)
            controller.publish_odom(sim_time)
            publish_semantic_visualizations(controller, scene, sim_time)
            ## height_scan 发布
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
                    hs_xy = make_world_grid(robot.data.root_pos_w, robot.data.root_quat_w,
                                            sd["size"], sd["resolution"], robot.device)
                    patches = compute_patches(hs_xy, {"height": heights_t}, terrain_size)
                    robot_z = robot.data.root_pos_w[:, 2:3]
                    scan = robot_z - patches["height"] - HEIGHT_OFFSET
                    scan = torch.nan_to_num(scan, nan=0.0, posinf=0.0, neginf=0.0)
                    for env_id in range(num_envs):
                        publish_height_scan_array(controller, env_id, scan[env_id])
                        publish_height_scan_image(controller, env_id, scan[env_id], height_scan_grid_shape, sim_time=sim_time)

            ## GT patch 发布
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

            if do_wheel_sensor and contact_sensors:
                for env_id in range(num_envs):
                    publish_wheel_sensor_data(
                        controller, contact_sensors, env_id,
                        robot=robot,
                        wheel_indices=controller.wheel_indices,
                        sim_time=sim_time, force_threshold=force_threshold,
                    )

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
    scene_cfg.terrain.usd_path = os.path.join(terrain_base, "terrain_only.usd")
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
    write_terrain_context("zhurong", "hard", cfg["terrain"]["name"],
                          load_terrain_size(terrain_base, cfg["terrain"]["size"]), scene.env_origins)

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

    # 相机发布可选，默认由启动参数控制。
    if args_cli.enable_cameras:
        setup_all_camera_publishers_from_scene(scene, freq=cfg["sim"]["pub_freq"])
        print(f"[INFO]: Camera publishers setup complete ({cfg['sim']['pub_freq']}Hz)")
        print("[INFO]: Semantic topics: /env_N/zhurong_<camera>_semantic[_labels/_viz]")
    else:
        print("[INFO]: Cameras disabled, skipping camera publishers")

    print("[INFO]: Waiting for /cmd_vel commands...")

    run_simulator(sim, scene, controller, tf_publisher, cfg)

    controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
