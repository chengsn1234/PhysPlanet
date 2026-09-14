#!/usr/bin/env python3
"""Publish CameraInfo + static TF for zhurong obs_f/obs_l/obs_r cameras."""
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

# 标准光学坐标系变换 (roll=-90, yaw=-90)
Q_X, Q_Y, Q_Z, Q_W = -0.5, 0.5, -0.5, 0.5

# Camera definitions: (link_name, camera_name)
CAMERAS = [
    ("camera_obs_F_link", "obsf"),
]


def make_camera_info(frame_id):
    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.height = H
    msg.width = W
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [FX, 0.0, CX, 0.0, 0.0, FY, CY, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def make_static_tf(parent_link, child_frame):
    tf_msg = TransformStamped()
    tf_msg.header.frame_id = parent_link
    tf_msg.child_frame_id = child_frame
    tf_msg.transform.rotation.x = Q_X
    tf_msg.transform.rotation.y = Q_Y
    tf_msg.transform.rotation.z = Q_Z
    tf_msg.transform.rotation.w = Q_W
    return tf_msg


def main():
    rclpy.init()
    node = Node("zhurong_obs_camera_info_publisher")
    prefix = f"env_{node.declare_parameter('env_id', 0).value}"

    tf_broadcaster = tf2_ros.StaticTransformBroadcaster(node)
    pubs = {}
    info_msgs = {}

    for link_name, cam_name in CAMERAS:
        parent_frame = f"{prefix}/{link_name}"
        child_frame = f"{prefix}/zhurong_{cam_name}_camera"
        topic = f"/{prefix}/zhurong_{cam_name}_camera_info"

        info_msgs[cam_name] = make_camera_info(child_frame)
        pubs[cam_name] = node.create_publisher(CameraInfo, topic, 10)

        tf_msg = make_static_tf(parent_frame, child_frame)
        tf_broadcaster.sendTransform(tf_msg)
        node.get_logger().info(f"TF: {parent_frame} -> {child_frame}")

    def timer_cb():
        stamp = node.get_clock().now().to_msg()
        for cam_name in info_msgs:
            info_msgs[cam_name].header.stamp = stamp
            pubs[cam_name].publish(info_msgs[cam_name])

    node.create_timer(0.1, timer_cb)
    node.get_logger().info(
        f"Publishing CameraInfo (fx={FX:.1f}) for obs_f/obs_l/obs_r + static TFs")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
