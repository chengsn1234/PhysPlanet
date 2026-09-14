#!/usr/bin/env python3
"""Publish CameraInfo and static TF for hunter camera optical frame."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TransformStamped
import tf2_ros

# focal_length=12.0, horizontal_aperture=20.955, 640x480
FX = 12.0 * 640 / 20.955
FY = FX
CX, CY = 320.0, 240.0
W, H = 640, 480

# Camera optical frame offset from zed2_base_link
# Standard optical frame transform: roll=-90, yaw=-90 (no tilt compensation)
POS_X, POS_Y, POS_Z = 0.0, 0.0, 0.0
Q_X, Q_Y, Q_Z, Q_W = -0.5, 0.5, -0.5, 0.5


def main():
    rclpy.init()
    node = Node("hunter_camera_info_publisher")
    env_id = node.declare_parameter("env_id", 0).value
    prefix = f"env_{env_id}"

    tf_broadcaster = tf2_ros.StaticTransformBroadcaster(node)

    parent_frame = f"{prefix}/zed2_base_link"
    child_frame = f"{prefix}/hunter_camera_nav"
    topic = f"/{prefix}/camera_info"

    info_msg = CameraInfo()
    info_msg.header.frame_id = child_frame
    info_msg.height = H
    info_msg.width = W
    info_msg.distortion_model = "plumb_bob"
    info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    info_msg.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
    info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info_msg.p = [FX, 0.0, CX, 0.0, 0.0, FY, CY, 0.0, 0.0, 0.0, 1.0, 0.0]

    pub = node.create_publisher(CameraInfo, topic, 10)

    tf_msg = TransformStamped()
    tf_msg.header.frame_id = parent_frame
    tf_msg.child_frame_id = child_frame
    tf_msg.transform.translation.x = POS_X
    tf_msg.transform.translation.y = POS_Y
    tf_msg.transform.translation.z = POS_Z
    tf_msg.transform.rotation.x = Q_X
    tf_msg.transform.rotation.y = Q_Y
    tf_msg.transform.rotation.z = Q_Z
    tf_msg.transform.rotation.w = Q_W
    tf_broadcaster.sendTransform(tf_msg)
    node.get_logger().info(f"TF: {parent_frame} -> {child_frame}")

    def timer_cb():
        info_msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(info_msg)

    node.create_timer(0.1, timer_cb)
    node.get_logger().info(f"Publishing CameraInfo (fx={FX:.1f}) for hunter camera")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
