from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from terrain_launch import make_actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("env_id", default_value="0"),
        DeclareLaunchArgument("context_file", default_value="~/isaaclab_ws/physplanet/ros2_ws/active_terrain.json"),
        DeclareLaunchArgument("ros2_config_file", default_value="",
                              description="ros2_config.yaml 路径（空=用包内默认 config/ros2_config.yaml）"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=os.path.expanduser(
            "~/isaaclab_ws/physplanet/ros2_ws/rviz/hunter_display.rviz")),
        OpaqueFunction(function=lambda context: make_actions(context, "hunter")),
    ])
