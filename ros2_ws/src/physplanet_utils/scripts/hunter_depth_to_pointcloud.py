#!/usr/bin/env python3
"""Convert hunter depth image to PointCloud2 for elevation_mapping."""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField


class DepthToPointCloud(Node):
    def __init__(self):
        super().__init__('hunter_depth_to_pointcloud')
        prefix = f"env_{self.declare_parameter('env_id', 0).value}"
        self.fx = self.fy = self.cx = self.cy = None
        self.stride = 4  # 降采样：每4个像素取1个，640x480 -> ~19k点

        self.create_subscription(
            CameraInfo, f'/{prefix}/camera_info',
            self._info_cb, 10)

        self.pc_pub = self.create_publisher(
            PointCloud2, f'/{prefix}/camera_points', 10)

        depth_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(
            Image, f'/{prefix}/camera_depth',
            self._depth_cb, depth_qos)

        self.get_logger().info('Waiting for CameraInfo...')

    def _info_cb(self, msg):
        if self.fx is not None:
            return
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.get_logger().info(
            f'Got CameraInfo: fx={self.fx:.1f}, fy={self.fy:.1f}')

    def _depth_cb(self, msg):
        if self.fx is None:
            return

        h, w = msg.height, msg.width
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)

        s = self.stride
        u = np.arange(0, w, s, dtype=np.float32)
        v = np.arange(0, h, s, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        depth = depth[::s, ::s]

        x = (uu - self.cx) * depth / self.fx
        y = (vv - self.cy) * depth / self.fy
        z = depth

        mask = np.isfinite(z) & (z > 0.1)
        pts = np.stack([x, y, z], axis=-1)[mask]

        pc_msg = PointCloud2()
        pc_msg.header = msg.header
        pc_msg.height = 1
        pc_msg.width = pts.shape[0]
        pc_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pc_msg.is_bigendian = False
        pc_msg.point_step = 12
        pc_msg.row_step = pc_msg.point_step * pts.shape[0]
        pc_msg.data = pts.astype(np.float32).tobytes()
        pc_msg.is_dense = True

        self.pc_pub.publish(pc_msg)


def main():
    rclpy.init()
    node = DepthToPointCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
