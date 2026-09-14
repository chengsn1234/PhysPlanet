#!/usr/bin/env python3
"""
发布地形 Mesh 为 RViz MarkerArray（地形 DAE 带贴图 + 石头 STL）。

用法：
    ros2 run physplanet_utils terrain_marker_publisher --terrain terrain2
"""

import argparse
import os
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


class TerrainMarkerPublisher(Node):
    def __init__(self, terrain_name: str = "terrain2"):
        super().__init__("terrain_marker_publisher")

        self.terrain_name = terrain_name
        self.publisher = self.create_publisher(MarkerArray, "/terrain_marker", 10)

        # 地形目录（相对 repo root 解析，realpath 解析 symlink）
        script_dir = os.path.dirname(os.path.realpath(__file__))
        repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
        base_dir = os.path.join(repo_root, "source/physplanet_assets/terrain_generation/terrain", terrain_name)

        markers = []

        # 1. 地形 DAE（带贴图）
        dae_path = os.path.join(base_dir, "terrain.dae")
        if os.path.exists(dae_path):
            terrain_marker = Marker()
            terrain_marker.header.frame_id = "odom"
            terrain_marker.ns = "terrain"
            terrain_marker.id = 0
            terrain_marker.type = Marker.MESH_RESOURCE
            terrain_marker.action = Marker.ADD
            terrain_marker.mesh_resource = f"file://{dae_path}"
            terrain_marker.mesh_use_embedded_materials = True
            terrain_marker.scale.x = 1.0
            terrain_marker.scale.y = 1.0
            terrain_marker.scale.z = 1.0
            terrain_marker.pose.orientation.w = 1.0
            markers.append(terrain_marker)
            self.get_logger().info(f"Terrain DAE: {dae_path}")

        # 2. 石头 STL（简化版，避免卡死）
        rocks_stl_path = os.path.join(base_dir, "rocks_simplified.stl")
        if not os.path.exists(rocks_stl_path):
            rocks_stl_path = os.path.join(base_dir, "rocks.stl")
        if os.path.exists(rocks_stl_path):
            size_mb = os.path.getsize(rocks_stl_path) / (1024 * 1024)
            if size_mb < 50:
                rocks_marker = Marker()
                rocks_marker.header.frame_id = "odom"
                rocks_marker.ns = "rocks"
                rocks_marker.id = 1
                rocks_marker.type = Marker.MESH_RESOURCE
                rocks_marker.action = Marker.ADD
                rocks_marker.mesh_resource = f"file://{rocks_stl_path}"
                rocks_marker.mesh_use_embedded_materials = False
                rocks_marker.scale.x = 1.0
                rocks_marker.scale.y = 1.0
                rocks_marker.scale.z = 1.0
                rocks_marker.pose.orientation.w = 1.0
                rocks_marker.color.r = 0.5
                rocks_marker.color.g = 0.4
                rocks_marker.color.b = 0.35
                rocks_marker.color.a = 1.0
                markers.append(rocks_marker)
                self.get_logger().info(f"Rocks STL: {rocks_stl_path} ({size_mb:.1f}MB)")
            else:
                self.get_logger().warn(f"Rocks STL too large ({size_mb:.1f}MB), skipping. Run with --rock_k 0.02 to reduce.")

        if not markers:
            self.get_logger().error("No terrain files found")
            return

        self.markers = markers

        # 定时发布
        self.timer = self.create_timer(1.0, self.publish_markers)
        self.get_logger().info("Terrain marker publisher started")

    def publish_markers(self):
        msg = MarkerArray()
        for marker in self.markers:
            marker.header.stamp = self.get_clock().now().to_msg()
            msg.markers.append(marker)
        self.publisher.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain", type=str, default="terrain2", help="Terrain name")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = TerrainMarkerPublisher(args.terrain)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
