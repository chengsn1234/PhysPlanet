# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 祝融号硬地控制测试
####################
## 链路：ROS cmd_vel -> 祝融号 6WD/6WS 控制 -> PhysX 平地接触 -> TF/camera
## 目标：验证祝融号基础控制和 ROS 发布，不包含火星地形和软地力学
####################

import argparse
from isaaclab.app import AppLauncher


####################
## 启动参数
####################

parser = argparse.ArgumentParser(description="Zhurong rover ROS2 control test.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--debug", action="store_true", help="Enable debug output.")
parser.add_argument("--pub_freq", type=int, default=10, help="ROS publishing frequency (Hz).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


####################
## Isaac / ROS
####################

import os
import rclpy
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")

from physplanet_assets.zhurong import ZHURONG_CFG
import sys
UTILS_DIR = os.path.join(os.path.dirname(__file__), "..", "utils")
sys.path.append(UTILS_DIR)
from zhurong.zhurong_control import ZhurongController
from zhurong.zhurong_sensor import setup_all_camera_publishers_from_scene
from urdf_tf_publisher import URDFTFPublisher


####################
## 场景
####################

@configclass
class ZhurongTestSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=500.0, color_temperature=6500.0),
    )

    robot: ArticulationCfg = ZHURONG_CFG.replace(prim_path="{ENV_REGEX_NS}/zhurong_mars_rover")

    # nav camera (ZED相机在PTZ桅杆末端, 用于远距离导航)
    camera_nav = CameraCfg(
        prim_path="{ENV_REGEX_NS}/zhurong_mars_rover/ZEDNav_Link/zed_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 15.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    # obs-f camera (底盘前方低处, 看近场障碍物)
    obsf_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/zhurong_mars_rover/camera_obs_F_link/zed_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 8.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    # obs-l camera (底盘左侧) — 暂时禁用（对齐 07）
    # obsl_camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/zhurong_mars_rover/camera_obs_L_link/zed_camera",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["depth"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=12.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 8.0)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )

    # obs-r camera (底盘右侧) — 暂时禁用（对齐 07）
    # obsr_camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/zhurong_mars_rover/camera_obs_R_link/zed_camera",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["depth"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=12.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 8.0)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )


####################
## 主循环
####################

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, controller: ZhurongController, tf_publisher: URDFTFPublisher, args_cli):
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    # 初始化 controller 的关节索引
    controller._init_joint_indices(robot)

    count = 0
    num_envs = scene.num_envs
    sim_time = 0.0  # 维护仿真时间，用于统一所有消息的时间戳

    print("\n" + "=" * 60)
    print("[INFO] Zhurong rover ROS2 control test")
    print(f"  Joints: {robot.num_joints}")
    print(f"  Environments: {num_envs}")
    print(f"  ROS topics: /cmd_vel, /env_N/cmd_vel")
    print("=" * 60 + "\n")

    while simulation_app.is_running():
        # 处理 ROS 消息
        rclpy.spin_once(controller, timeout_sec=0.0)

        # 发布完整 TF tree（使用 sim_time）
        tf_publisher.publish_tf_all_envs(sim_time)

        # 批量计算控制指令
        joint_pos_target, joint_vel_target = controller.compute_control_batch()

        # 应用控制
        robot.set_joint_position_target(joint_pos_target)
        robot.set_joint_velocity_target(joint_vel_target)

        # 仿真步进
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        sim_time += sim_dt  # 累积仿真时间

        # Debug 输出
        if args_cli.debug and count % 100 == 0:
            pos = robot.data.root_pos_w[0]
            vel = robot.data.root_lin_vel_w[0]
            print(f"[{count}] pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), vel: {vel[0]:.2f} m/s")
            print(f"  cmd: linear={controller.linear_x[0]:.2f}, angular={controller.angular_z[0]:.2f}")

        count += 1


####################
## main
####################

def main():
    rclpy.init()

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, gravity=(0.0, 0.0, -3.71), device=args_cli.device)  # 火星重力
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, 3.0, 2.0], target=[0.0, 0.0, 0.5])

    scene_cfg = ZhurongTestSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0)

    # enable_cameras=false 时不创建相机
    if not args_cli.enable_cameras:
        scene_cfg.camera_nav = None
        scene_cfg.obsf_camera = None

    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    controller = ZhurongController(robot=robot, num_envs=args_cli.num_envs, device=robot.device)

    urdf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "physplanet_assets", "zhurong_assets", "zhurong.urdf"
    )
    tf_publisher = URDFTFPublisher(controller, urdf_path, robot, args_cli.num_envs)

    # 相机发布可选，默认由启动参数控制。
    if args_cli.enable_cameras:
        setup_all_camera_publishers_from_scene(scene, freq=args_cli.pub_freq)
        print(f"[INFO]: Camera publishers setup complete ({args_cli.pub_freq}Hz)")
    else:
        print("[INFO]: Cameras disabled, skipping camera publishers")

    print("[INFO]: Setup complete. Waiting for /cmd_vel commands...")

    run_simulator(sim, scene, controller, tf_publisher, args_cli)

    controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
