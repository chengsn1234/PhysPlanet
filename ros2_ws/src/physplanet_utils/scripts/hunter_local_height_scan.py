#!/usr/bin/env python3
"""Crop a robot-centric 5x5m local height scan from a larger elevation map.

Canonical output convention (to match docs/tasks/04 sim height_scan):
  - array shape is (num_y, num_x)
  - row -> y
  - col -> x
  - value  = robot_z - terrain_z - offset

Incoming GridMap convention from this repo's elevation_mapping publisher:
  - row -> x
  - col -> y
"""
import math

import numpy as np
import rclpy
from grid_map_msgs.msg import GridMap
from rclpy.node import Node
from scipy.ndimage import map_coordinates
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import tf2_ros


def quat_to_yaw_xyzw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def height_scan_grid_to_mono8(height_grid: np.ndarray) -> np.ndarray:
    """Use the same debug-image convention as the sim height scan publisher."""
    finite = np.isfinite(height_grid)
    if np.any(finite):
        vals = height_grid[finite]
        vmin = float(np.percentile(vals, 2.0))
        vmax = float(np.percentile(vals, 98.0))
        if vmax - vmin < 1.0e-6:
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
        if vmax - vmin < 1.0e-6:
            vmax = vmin + 1.0
        norm = (height_grid - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(height_grid)
    norm = np.where(finite, norm, 0.0)
    gray = (norm * 255.0).clip(0, 255).astype(np.uint8)
    return np.flipud(np.fliplr(gray.T))


def grid_map_layer_to_matrix(msg: GridMap, layer_index: int) -> np.ndarray:
    """Decode a GridMap layer with the same storage-order rule as grid_map_ros.

    ANYbotics/grid_map uses the first layout label to indicate storage order:
    `row_index` means row-major, `column_index` means column-major. The current
    elevation map publisher labels the message as column-major, so a plain
    row-major NumPy reshape would silently transpose the spatial axes.
    """
    array = msg.data[layer_index]
    data = np.array(array.data, dtype=np.float32)
    dims = {dim.label: int(dim.size) for dim in array.layout.dim}
    rows = dims.get("row_index")
    cols = dims.get("column_index")
    if rows is None or cols is None:
        raise ValueError("GridMap layer layout must contain row_index and column_index")

    first_label = array.layout.dim[0].label
    if first_label == "row_index":
        return data.reshape(rows, cols)
    if first_label == "column_index":
        return data.reshape(cols, rows).T
    raise ValueError(f"Unexpected GridMap storage label: {first_label}")


class HunterLocalHeightScan(Node):
    def __init__(self):
        super().__init__("hunter_local_height_scan")

        self.declare_parameter("map_topic", "/elevation_mapping_node/elevation_map_raw")
        self.declare_parameter("map_layer", "inpaint")
        self.declare_parameter("scan_topic", "/env_0/local_height_scan")
        self.declare_parameter("image_topic", "/env_0/local_height_scan_image")
        self.declare_parameter("scan_size", 101)
        self.declare_parameter("crop_length", 5.0)
        self.declare_parameter("sensor_height_offset", 0.26878)
        self.declare_parameter("base_frame", "env_0/base_link")
        self.declare_parameter("map_frame", "odom")

        self.scan_size = int(self.get_parameter("scan_size").value)
        self.crop_length = float(self.get_parameter("crop_length").value)
        self.sensor_height_offset = float(self.get_parameter("sensor_height_offset").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.map_frame = str(self.get_parameter("map_frame").value)

        map_topic = str(self.get_parameter("map_topic").value)
        self.map_layer = str(self.get_parameter("map_layer").value)
        scan_topic = str(self.get_parameter("scan_topic").value)
        image_topic = str(self.get_parameter("image_topic").value)

        self.pub = self.create_publisher(Float32MultiArray, scan_topic, 10)
        self.img_pub = self.create_publisher(Image, image_topic, 10)
        self.sub = self.create_subscription(GridMap, map_topic, self._cb, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Canonical output grid: row -> y, col -> x.
        axis = np.linspace(-0.5 * self.crop_length, 0.5 * self.crop_length, self.scan_size, dtype=np.float32)
        local_x, local_y = np.meshgrid(axis, axis, indexing="xy")
        self.local_x = local_x
        self.local_y = local_y

        self.get_logger().info(
            f"{map_topic}[{self.map_layer}] -> {scan_topic} ({self.crop_length:.2f}m, {self.scan_size}x{self.scan_size})"
        )

    def _cb(self, msg: GridMap):
        try:
            idx = list(msg.layers).index(self.map_layer)
        except ValueError:
            return

        try:
            grid = grid_map_layer_to_matrix(msg, idx)
        except ValueError as exc:
            self.get_logger().warn(str(exc))
            return

        row_off = msg.outer_start_index
        col_off = msg.inner_start_index
        if row_off != 0 or col_off != 0:
            grid = np.roll(np.roll(grid, -row_off, axis=0), -col_off, axis=1)

        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
        except Exception:
            return

        robot_x = float(tf.transform.translation.x)
        robot_y = float(tf.transform.translation.y)
        robot_z = float(tf.transform.translation.z)
        q = tf.transform.rotation
        yaw = quat_to_yaw_xyzw(float(q.x), float(q.y), float(q.z), float(q.w))

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        world_x = robot_x + cos_yaw * self.local_x - sin_yaw * self.local_y
        world_y = robot_y + sin_yaw * self.local_x + cos_yaw * self.local_y

        resolution = float(msg.info.resolution)
        center_x = float(msg.info.pose.position.x)
        center_y = float(msg.info.pose.position.y)
        # Match ANYbotics grid_map indexing exactly:
        #   position = map_center + (half_length - 0.5 * res) - res * index
        # so index 0 is the (+x, +y) corner of the map, not the (-x, -y) corner.
        #
        # For the current publisher, the layer matrix is organized as:
        #   axis 0 -> row_index -> x
        #   axis 1 -> column_index -> y
        # and both axes decrease in map coordinates as the matrix indices increase.
        half_len_x = 0.5 * float(msg.info.length_x)
        half_len_y = 0.5 * float(msg.info.length_y)
        sample_rows = (center_x + half_len_x - 0.5 * resolution - world_x) / resolution
        sample_cols = (center_y + half_len_y - 0.5 * resolution - world_y) / resolution
        inside = (
            (sample_rows >= 1.0)
            & (sample_rows <= (grid.shape[0] - 2.0))
            & (sample_cols >= 1.0)
            & (sample_cols <= (grid.shape[1] - 2.0))
        )

        valid = np.isfinite(grid).astype(np.float32)
        grid_filled = np.where(np.isfinite(grid), grid, 0.0)
        coords = np.stack([sample_rows, sample_cols], axis=0)
        sampled = map_coordinates(grid_filled, coords, order=1, mode="nearest")
        valid_weight = map_coordinates(valid, coords, order=1, mode="nearest")
        sampled[(~inside) | (valid_weight < 0.999)] = np.nan

        local_height = robot_z - sampled - self.sensor_height_offset
        local_height = np.nan_to_num(local_height, nan=0.0, posinf=0.0, neginf=0.0)

        out = Float32MultiArray()
        out.data = local_height.flatten().astype(np.float32).tolist()
        self.pub.publish(out)

        gray = height_scan_grid_to_mono8(local_height)

        img = Image()
        img.header.stamp = msg.header.stamp
        img.header.frame_id = self.base_frame
        img.height = self.scan_size
        img.width = self.scan_size
        img.encoding = "mono8"
        img.step = self.scan_size
        img.data = gray.tobytes()
        self.img_pub.publish(img)


def main():
    rclpy.init()
    node = HunterLocalHeightScan()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
