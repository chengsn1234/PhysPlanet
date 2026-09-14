# 通用 ROS 传感器发布
####################
## 功能：height scan / wheel force / wheel scalar 的通用 ROS topic 发布
## 车型差异：topic 名和 frame 映射由 hunter_sensor / zhurong_sensor 提供
####################

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Float64, Header

from sim_utils import sim_time_to_ros_time


_publishers = {}

SEMANTIC_LAYER_COLORS = [
    (225, 95, 60), (235, 170, 65), (125, 185, 90), (70, 135, 190), (155, 105, 185),
]


####################
## Publisher 缓存
####################

def get_or_create_publisher(node, topic_name, msg_type, queue_size=1):
    key = (id(node), topic_name, msg_type)
    if key not in _publishers:
        _publishers[key] = node.create_publisher(msg_type, topic_name, queue_size)
    return _publishers[key]


def _semantic_color(label):
    if label == "BACKGROUND":
        return (0, 0, 0)
    if label == "rock":
        return (220, 220, 220)
    if label.startswith("layer_") and label[6:].isdigit():
        return SEMANTIC_LAYER_COLORS[(int(label[6:]) - 1) % len(SEMANTIC_LAYER_COLORS)]
    return (180, 70, 180)


def publish_semantic_visualizations(node, scene, camera_topics, sim_time=None):
    """Publish colorized semantic previews for the configured cameras."""
    for camera_name, (topic_prefix, frame_prefix) in camera_topics.items():
        try:
            camera = scene[camera_name]
            if "semantic_segmentation" not in camera.cfg.data_types:
                continue
            images = camera.data.output["semantic_segmentation"]
            infos = camera.data.info
        except KeyError:
            continue

        for env_id, image in enumerate(images):
            ids = image[..., 0].detach().cpu().numpy()
            rgb = np.zeros((*ids.shape, 3), dtype=np.uint8)
            labels = infos[env_id].get("semantic_segmentation", {}).get("idToLabels", {})
            for semantic_id, metadata in labels.items():
                rgb[ids == int(semantic_id)] = _semantic_color(metadata.get("class", "UNLABELLED"))

            msg = Image()
            msg.header = Header()
            msg.header.stamp = sim_time_to_ros_time(sim_time).to_msg()
            msg.header.frame_id = f"env_{env_id}/{frame_prefix}"
            msg.height, msg.width = rgb.shape[:2]
            msg.encoding = "rgb8"
            msg.step = msg.width * 3
            msg.data = rgb.tobytes()
            get_or_create_publisher(node, f"/env_{env_id}/{topic_prefix}_semantic_viz", Image).publish(msg)


####################
## Height Scan
####################

def height_scan_grid_to_mono8(height_grid: np.ndarray) -> np.ndarray:
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


def publish_height_scan_array(node, env_id, height_scan, topic_name=None):
    if height_scan is None:
        return
    if topic_name is None:
        topic_name = f"/env_{env_id}/sim_height_scan"

    publisher = get_or_create_publisher(node, topic_name, Float32MultiArray)
    msg = Float32MultiArray()
    if hasattr(height_scan, "detach"):
        msg.data = height_scan.detach().cpu().float().tolist()
    else:
        msg.data = np.asarray(height_scan, dtype=np.float32).reshape(-1).tolist()
    publisher.publish(msg)


def publish_height_scan_image(node, env_id, height_scan, grid_shape, topic_name=None, sim_time: float = None):
    if height_scan is None:
        return
    if topic_name is None:
        topic_name = f"/env_{env_id}/sim_height_scan_image"

    publisher = get_or_create_publisher(node, topic_name, Image)
    if hasattr(height_scan, "detach"):
        grid = height_scan.detach().cpu().float().numpy().reshape(grid_shape)
    else:
        grid = np.asarray(height_scan, dtype=np.float32).reshape(grid_shape)

    msg = Image()
    msg.header = Header()
    msg.header.stamp = sim_time_to_ros_time(sim_time).to_msg()
    msg.header.frame_id = f"env_{env_id}/base_link"
    msg.height = int(grid_shape[0])
    msg.width = int(grid_shape[1])
    msg.encoding = "mono8"
    msg.step = int(grid_shape[1])
    msg.data = height_scan_grid_to_mono8(grid).tobytes()
    publisher.publish(msg)


####################
## GT Patch
####################

def publish_gt_patch(node, env_id, patch, name, grid_shape, sim_time=None, publish_image=False):
    """发布一个 GT patch。
    Float32MultiArray → /env_{id}/gt_{name}
    publish_image=True 时额外发 mono8 → /env_{id}/gt_{name}_image
    """
    if patch is None:
        return
    topic = f"/env_{env_id}/gt_{name}"
    publisher = get_or_create_publisher(node, topic, Float32MultiArray)
    msg = Float32MultiArray()
    if hasattr(patch, "detach"):
        msg.data = patch.detach().cpu().float().tolist()
    else:
        msg.data = np.asarray(patch, dtype=np.float32).reshape(-1).tolist()
    publisher.publish(msg)

    if publish_image:
        img_topic = f"/env_{env_id}/gt_{name}_image"
        img_publisher = get_or_create_publisher(node, img_topic, Image)
        if hasattr(patch, "detach"):
            grid = patch.detach().cpu().float().numpy().reshape(grid_shape)
        else:
            grid = np.asarray(patch, dtype=np.float32).reshape(grid_shape)
        img = Image()
        img.header = Header()
        img.header.stamp = sim_time_to_ros_time(sim_time).to_msg()
        img.header.frame_id = f"env_{env_id}/base_link"
        img.height = int(grid_shape[0])
        img.width = int(grid_shape[1])
        img.encoding = "mono8"
        img.step = int(grid_shape[1])
        img.data = height_scan_grid_to_mono8(grid).tobytes()
        img_publisher.publish(img)


####################
## Wheel Topics
####################

def publish_wheel_contact_forces(node, env_id, wheel_name, net_force_w, mr, wheel_frame_map, sim_time: float = None):
    if net_force_w is None:
        return
    topic = f"/env_{env_id}/wheel_force_{wheel_name}"
    publisher = get_or_create_publisher(node, topic, WrenchStamped)

    msg = WrenchStamped()
    msg.header = Header()
    msg.header.stamp = sim_time_to_ros_time(sim_time).to_msg()
    msg.header.frame_id = f"env_{env_id}/{wheel_frame_map.get(wheel_name, 'base_link')}"

    if hasattr(net_force_w, "detach"):
        f = net_force_w.detach().cpu().float()
    else:
        f = np.asarray(net_force_w, dtype=np.float32)
    msg.wrench.force.x = float(f[0])
    msg.wrench.force.y = float(f[1])
    msg.wrench.force.z = float(f[2])
    msg.wrench.torque.z = float(mr)
    publisher.publish(msg)


def publish_wheel_torque(node, env_id, wheel_name, mr, sim_time: float = None):
    if mr is None:
        return
    topic = f"/env_{env_id}/wheel_torque_{wheel_name}"
    publisher = get_or_create_publisher(node, topic, Float64)
    msg = Float64()
    msg.data = float(mr)
    publisher.publish(msg)


def publish_wheel_slip(node, env_id, wheel_name, slip, sim_time: float = None):
    if slip is None:
        return
    topic = f"/env_{env_id}/wheel_slip_{wheel_name}"
    publisher = get_or_create_publisher(node, topic, Float64)
    msg = Float64()
    msg.data = float(slip)
    publisher.publish(msg)


def publish_wheel_scalar(node, env_id, wheel_name, field, value):
    if value is None:
        return
    topic = f"/env_{env_id}/wheel_{field}_{wheel_name}"
    publisher = get_or_create_publisher(node, topic, Float64)
    msg = Float64()
    msg.data = float(value)
    publisher.publish(msg)
