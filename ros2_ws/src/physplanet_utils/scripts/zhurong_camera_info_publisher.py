#!/usr/bin/env python3
"""Publish CameraInfo and static TF for zhurong depth camera."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TransformStamped
import tf2_ros


def main():
    rclpy.init()
    node = Node('zhurong_camera_info_publisher')
    prefix = f"env_{node.declare_parameter('env_id', 0).value}"

    # Camera intrinsics: focal_length=12.0, horizontal_aperture=20.955, 480x640
    fx = 12.0 * 640 / 20.955
    fy = fx
    cx, cy = 320.0, 240.0

    msg = CameraInfo()
    msg.header.frame_id = f'{prefix}/zhurong_camera_nav'
    msg.height = 480
    msg.width = 640
    msg.distortion_model = 'plumb_bob'
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

    pub = node.create_publisher(CameraInfo, f'/{prefix}/zhurong_camera_nav_info', 10)

    # Static TF: env_0/ZEDNav_Link -> env_0/zhurong_camera_nav
    # 标准光学坐标系变换: roll=-90, yaw=-90
    tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(node)
    tf_msg = TransformStamped()
    tf_msg.header.frame_id = f'{prefix}/ZEDNav_Link'
    tf_msg.child_frame_id = f'{prefix}/zhurong_camera_nav'
    tf_msg.transform.rotation.x = -0.5
    tf_msg.transform.rotation.y = 0.5
    tf_msg.transform.rotation.z = -0.5
    tf_msg.transform.rotation.w = 0.5
    tf_static_broadcaster.sendTransform(tf_msg)

    def timer_cb():
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)

    node.create_timer(0.1, timer_cb)
    node.get_logger().info(
        f'Publishing CameraInfo (fx={fx:.1f}) + TF: env_0/ZEDNav_Link -> env_0/zhurong_camera_nav')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
