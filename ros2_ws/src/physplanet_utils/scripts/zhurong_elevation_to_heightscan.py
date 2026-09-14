#!/usr/bin/env python3
"""Convert elevation_map (GridMap) to height_scan (Float32MultiArray) for zhurong policy node."""
import numpy as np
import rclpy
from rclpy.node import Node
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from scipy.ndimage import zoom
import tf2_ros


class ElevationToHeightScan(Node):
    def __init__(self):
        super().__init__('zhurong_elevation_to_heightscan')

        self.declare_parameter('map_topic', '/elevation_mapping_node/elevation_map_raw')
        self.declare_parameter('scan_topic', '/env_0/height_scan')
        self.declare_parameter('scan_size', 101)
        self.declare_parameter('sensor_height_offset', 0.40)  # zhurong base_link 离地高度(轮心-0.25+轮半径0.15)
        self.declare_parameter('base_frame', 'env_0/base_link')
        self.declare_parameter('map_frame', 'odom')

        self.scan_size = int(self.get_parameter('scan_size').value)
        self.target_len = self.scan_size * self.scan_size
        self.sensor_height_offset = float(self.get_parameter('sensor_height_offset').value)
        self.base_frame = self.get_parameter('base_frame').value
        self.map_frame = self.get_parameter('map_frame').value

        map_topic = self.get_parameter('map_topic').value
        scan_topic = self.get_parameter('scan_topic').value

        self.pub = self.create_publisher(Float32MultiArray, scan_topic, 10)
        self.img_pub = self.create_publisher(Image, '/env_0/height_scan_image', 10)
        self.sub = self.create_subscription(GridMap, map_topic, self._cb, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            f'{map_topic} -> {scan_topic} (size={self.target_len})')

    def _cb(self, msg: GridMap):
        try:
            idx = list(msg.layers).index('elevation')
        except ValueError:
            return

        data = np.array(msg.data[idx].data, dtype=np.float32)
        rows = msg.data[idx].layout.dim[0].size
        cols = msg.data[idx].layout.dim[1].size
        grid = data.reshape(rows, cols)

        # handle circular buffer offset
        row_off = msg.outer_start_index
        col_off = msg.inner_start_index
        if row_off != 0 or col_off != 0:
            grid = np.roll(np.roll(grid, -row_off, axis=0), -col_off, axis=1)

        # elevation_map: row=X, col=Y -> IsaacLab: row=Y, col=X
        grid = grid.T

        # convert to relative height: robot_z - terrain_z - offset
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
            robot_z = tf.transform.translation.z
            grid = robot_z - grid - self.sensor_height_offset
        except Exception:
            return

        # resize with interpolation (NaN preserved)
        if grid.shape != (self.scan_size, self.scan_size):
            valid = np.isfinite(grid)
            grid_filled = np.where(valid, grid, 0.0)
            scale_r = self.scan_size / grid.shape[0]
            scale_c = self.scan_size / grid.shape[1]
            resized = zoom(grid_filled, (scale_r, scale_c), order=1)
            valid_resized = zoom(valid.astype(np.float32), (scale_r, scale_c), order=1) > 0.5
            resized[~valid_resized] = np.nan
            grid = resized

        # publish Float32MultiArray
        out = Float32MultiArray()
        out.data = grid.flatten().astype(np.float32).tolist()
        self.pub.publish(out)

        # publish Image for RViz visualization
        valid = np.isfinite(grid)
        if np.any(valid):
            vmin, vmax = np.nanmin(grid), np.nanmax(grid)
            if vmax - vmin < 1e-6:
                vmax = vmin + 1.0
            norm = (grid - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(grid)
        norm = np.where(valid, norm, 0.0)
        gray = (norm * 255).clip(0, 255).astype(np.uint8)

        img = Image()
        img.header.stamp = msg.header.stamp
        img.height = self.scan_size
        img.width = self.scan_size
        img.encoding = 'mono8'
        img.step = self.scan_size
        img.data = gray.tobytes()
        self.img_pub.publish(img)


def main():
    rclpy.init()
    node = ElevationToHeightScan()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
