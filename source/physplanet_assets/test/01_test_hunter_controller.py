# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Hunter 硬地控制测试
####################
## 链路：ROS cmd_vel -> Hunter Ackermann 控制 -> PhysX 平地接触 -> TF/odom/camera
## 目标：验证 Hunter 基础控制和 ROS 发布，不包含火星地形和软地力学
####################

import argparse
from isaaclab.app import AppLauncher


####################
## 启动参数
####################

parser = argparse.ArgumentParser(description="Hunter rover ROS2 control test.")
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
import sys
import rclpy
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")

from physplanet_assets.hunter import HUNTER_SUS_CFG
UTILS_DIR = os.path.join(os.path.dirname(__file__), "..", "utils")
sys.path.append(UTILS_DIR)
from hunter.hunter_control import HunterController
from hunter.hunter_sensor import setup_all_camera_publishers_from_scene
from urdf_tf_publisher import URDFTFPublisher


####################
## 场景
####################

@configclass
class HunterTestSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=500.0, color_temperature=6500.0),
    )

    robot: ArticulationCfg = HUNTER_SUS_CFG.replace(prim_path="{ENV_REGEX_NS}/hunter")

    # 相机（zed2，挂在 base_link 链路末端）
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/hunter/base_link/hunter_link/rear_center_link/imu/zed2_base_link/zed2_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )


####################
## 主循环
####################

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, controller: HunterController, tf_publisher: URDFTFPublisher, args_cli):
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    count = 0
    num_envs = scene.num_envs
    sim_time = 0.0

    print("\n" + "=" * 60)
    print("[INFO] Hunter rover ROS2 control test")
    print(f"  Joints: {robot.num_joints}")
    print(f"  Environments: {num_envs}")
    print(f"  ROS topics: /cmd_vel, /env_N/cmd_vel")
    print("=" * 60 + "\n")

    while simulation_app.is_running():
        # 处理 ROS 消息
        rclpy.spin_once(controller, timeout_sec=0.0)

        # 发布完整 TF tree + odom（使用 sim_time）
        tf_publisher.publish_tf_all_envs(sim_time)
        controller.publish_odom(sim_time)

        # 批量计算控制指令
        joint_pos_target, joint_vel_target = controller.compute_control_batch()

        # 应用控制
        robot.set_joint_position_target(joint_pos_target)
        robot.set_joint_velocity_target(joint_vel_target)

        # 仿真步进
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        sim_time += sim_dt

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

    scene_cfg = HunterTestSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0)

    # enable_cameras=false 时不创建相机
    if not args_cli.enable_cameras:
        scene_cfg.camera = None

    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    controller = HunterController(num_envs=args_cli.num_envs, robot=robot, device=robot.device)

    urdf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "physplanet_assets", "hunter_assets", "hunter_se_suspension_description.urdf"
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
