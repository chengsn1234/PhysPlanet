#!/usr/bin/env python3
"""Convert zhurong obs_f/obs_l/obs_r depth images to PointCloud2 for elevation_mapping."""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField


CAMERAS = ["obsf"]

# Filter out near-field self points from the obsf camera before generating the
# point cloud. Raise this a little more if the front wheels still leak through.
OBS_F_MIN_DEPTH = 0.55


class ObsDepthToPointCloud(Node):
    def __init__(self):
        super().__init__("zhurong_obs_depth_to_pointcloud")
        prefix = f"env_{self.declare_parameter('env_id', 0).value}"
        self.stride = 4

        self.fx = {}
        self.fy = {}
        self.cx = {}
        self.cy = {}
        self.pc_pubs = {}

        depth_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        for cam_name in CAMERAS:
            self.fx[cam_name] = None
            self.fy[cam_name] = None
            self.cx[cam_name] = None
            self.cy[cam_name] = None

            self.create_subscription(
                CameraInfo, f"/{prefix}/zhurong_{cam_name}_camera_info",
                lambda msg, cn=cam_name: self._info_cb(msg, cn), 10)

            self.pc_pubs[cam_name] = self.create_publisher(
                PointCloud2, f"/{prefix}/zhurong_{cam_name}_camera_points", 10)

            self.create_subscription(
                Image, f"/{prefix}/zhurong_{cam_name}_camera_depth",
                lambda msg, cn=cam_name: self._depth_cb(msg, cn), depth_qos)

        self.get_logger().info("Waiting for obs CameraInfo...")

    def _info_cb(self, msg, cam_name):
        if self.fx[cam_name] is not None:
            return
        self.fx[cam_name] = msg.k[0]
        self.fy[cam_name] = msg.k[4]
        self.cx[cam_name] = msg.k[2]
        self.cy[cam_name] = msg.k[5]
        self.get_logger().info(
            f"[{cam_name}] CameraInfo: fx={self.fx[cam_name]:.1f}")

    def _depth_cb(self, msg, cam_name):
        if self.fx[cam_name] is None:
            return

        h, w = msg.height, msg.width
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)

        s = self.stride
        u = np.arange(0, w, s, dtype=np.float32)
        v = np.arange(0, h, s, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        depth = depth[::s, ::s]

        fx = self.fx[cam_name]
        fy = self.fy[cam_name]
        cx = self.cx[cam_name]
        cy = self.cy[cam_name]

        x = (uu - cx) * depth / fx
        y = (vv - cy) * depth / fy
        z = depth

        min_depth = OBS_F_MIN_DEPTH if cam_name == "obsf" else 0.1
        mask = np.isfinite(z) & (z > min_depth)
        pts = np.stack([x, y, z], axis=-1)[mask]

        pc_msg = PointCloud2()
        pc_msg.header = msg.header
        pc_msg.height = 1
        pc_msg.width = pts.shape[0]
        pc_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pc_msg.is_bigendian = False
        pc_msg.point_step = 12
        pc_msg.row_step = pc_msg.point_step * pts.shape[0]
        pc_msg.data = pts.astype(np.float32).tobytes()
        pc_msg.is_dense = True

        self.pc_pubs[cam_name].publish(pc_msg)


def main():
    rclpy.init()
    node = ObsDepthToPointCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
