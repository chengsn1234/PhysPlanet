# Hunter 传感器发布
####################
## 功能：相机、height scan、硬地接触和软地轮传感器话题发布
## 车型差异：Hunter 的 topic 名、frame 名和轮子顺序
####################

from __future__ import annotations

import numpy as np
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata
import omni.syntheticdata._syntheticdata as sd
import torch
from omni.syntheticdata import SyntheticData

from sim_utils import compute_slip_ratio
from rover_sensor import (
    publish_height_scan_array,
    publish_height_scan_image,
    publish_semantic_visualizations as _publish_semantic_visualizations,
    publish_wheel_scalar,
    publish_wheel_slip,
    publish_wheel_torque,
    publish_wheel_contact_forces as _publish_wheel_contact_forces,
)

WHEEL_NAMES = ["re_left", "fr_left", "re_right", "fr_right"]

WHEEL_STEER_MAP = {
    "re_left": "re_left_link",
    "fr_left": "fr_steer_left_link",
    "re_right": "re_right_link",
    "fr_right": "fr_steer_right_link",
}

CAMERA_TOPIC_MAP = {"camera": ("camera", "hunter_camera_nav")}

def publish_wheel_contact_forces(node, env_id, wheel_name, net_force_w, mr, sim_time: float = None):
    return _publish_wheel_contact_forces(node, env_id, wheel_name, net_force_w, mr, WHEEL_STEER_MAP, sim_time)


def publish_soft_wheel_sensor_data(node, env_id, wheel_names, forces_w, tm_out, dbg, applied_torque=None, sim_time: float = None):
    # 软地无 PhysX ground contact，用公式力补齐每轮虚拟传感器话题。
    for i, wname in enumerate(wheel_names):
        force_w = forces_w[env_id, i]
        torque = 0.0 if applied_torque is None else applied_torque[env_id, i]

        publish_wheel_contact_forces(node, env_id, wname, force_w, torque, sim_time)
        publish_wheel_slip(node, env_id, wname, tm_out["slip"][env_id, i], sim_time)
        publish_wheel_torque(node, env_id, wname, torque, sim_time)

        publish_wheel_scalar(node, env_id, wname, "sinkage", tm_out["sinkage"][env_id, i])
        publish_wheel_scalar(node, env_id, wname, "fn", tm_out["Fn"][env_id, i])
        publish_wheel_scalar(node, env_id, wname, "fz", tm_out["force_z"][env_id, i])
        publish_wheel_scalar(node, env_id, wname, "fwd", tm_out["force_forward"][env_id, i])
        publish_wheel_scalar(node, env_id, wname, "side", dbg["side_drag"][env_id, i])
        publish_wheel_scalar(node, env_id, wname, "v_fwd", dbg["v_fwd"][env_id, i])
        publish_wheel_scalar(node, env_id, wname, "omega", dbg["omega"][env_id, i])


def publish_wheel_sensor_data(node, contact_sensors, env_id,
                              robot, wheel_indices, wheel_radius=0.135,
                              sim_time: float = None, force_threshold: float = 0.1):
    # 批量发布 4 轮原始数据（force+slip+torque），不计算 μ（μ 由 friction_identifier 节点订阅辨识）。
    # net_forces_w 含所有碰撞体噪声，由辨识端滑动窗口处理；applied_torque 近似摩擦力矩，仅稳态匀速滑移成立。
    root_vel_b = robot.data.root_lin_vel_b[env_id]

    for wname, sensor in contact_sensors.items():
        # GPU 管线对 mesh collider 不支持 contact filter，force_matrix_w 全零
        # 直接用 net_forces_w，噪声由辨识端滑动窗口均值处理
        try:
            normal_force = sensor.data.net_forces_w[env_id, 0, :]
        except (IndexError, AttributeError):
            continue
        if hasattr(normal_force, "detach"):
            fn = float(torch.norm(normal_force))
        else:
            fn = float(np.linalg.norm(normal_force))
        if fn < force_threshold:
            continue

        widx = wheel_indices[wname]
        mr = float(robot.data.applied_torque[env_id, widx])
        omega = float(robot.data.joint_vel[env_id, widx])
        slip = compute_slip_ratio(omega, root_vel_b, wheel_radius)

        publish_wheel_slip(node, env_id, wname, slip, sim_time)
        publish_wheel_contact_forces(node, env_id, wname, normal_force, mr, sim_time)
        publish_wheel_torque(node, env_id, wname, mr, sim_time)


def setup_camera_publishers_from_render_product(render_product_path, freq=10, env_id=0, data_types=None):
    # 直接基于已有 render product 建 ROS 图像发布。
    if data_types is None:
        data_types = ["rgb", "depth"]
    step_size = int(60 / freq)
    frame_id = f"env_{env_id}/hunter_camera_nav"

    if "rgb" in data_types:
        rv_rgb = SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
        writer_rgb = rep.writers.get(rv_rgb + "ROS2PublishImage")
        writer_rgb.initialize(frameId=frame_id, nodeNamespace="", queueSize=1, topicName=f"env_{env_id}/camera_rgb")
        writer_rgb.attach([render_product_path])
        gate_path_rgb = SyntheticData._get_node_path(rv_rgb + "IsaacSimulationGate", render_product_path)
        og.Controller.attribute(gate_path_rgb + ".inputs:step").set(step_size)

    if "depth" in data_types:
        rv_depth = SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.DistanceToImagePlane.name)
        writer_depth = rep.writers.get(rv_depth + "ROS2PublishImage")
        writer_depth.initialize(frameId=frame_id, nodeNamespace="", queueSize=1, topicName=f"env_{env_id}/camera_depth")
        writer_depth.attach([render_product_path])
        gate_path_depth = SyntheticData._get_node_path(rv_depth + "IsaacSimulationGate", render_product_path)
        og.Controller.attribute(gate_path_depth + ".inputs:step").set(step_size)

    if "semantic_segmentation" in data_types:
        semantic_topic = f"env_{env_id}/camera_semantic"
        semantic_writer = rep.writers.get("ROS2PublishSemanticSegmentation")
        semantic_writer.initialize(frameId=frame_id, nodeNamespace="", queueSize=1, topicName=semantic_topic)
        semantic_writer.attach([render_product_path])

        label_writer = rep.writers.get("SemanticSegmentationSDROS2PublishSemanticLabels")
        label_writer.initialize(nodeNamespace="", queueSize=1, topicName=f"{semantic_topic}_labels")
        label_writer.attach([render_product_path])

        rv_semantic = SyntheticData.convert_sensor_type_to_rendervar("SemanticSegmentation")
        gate_path_semantic = SyntheticData._get_node_path(rv_semantic + "IsaacSimulationGate", render_product_path)
        og.Controller.attribute(gate_path_semantic + ".inputs:step").set(step_size)


def setup_all_camera_publishers_from_scene(scene, freq=50):
    # 对 scene 里现有的 Hunter 相机统一建发布器。
    camera_sensor = scene["camera"]
    for env_id in range(scene.num_envs):
        render_product_path = camera_sensor._render_product_paths[env_id]
        setup_camera_publishers_from_render_product(render_product_path, freq, env_id, camera_sensor.cfg.data_types)
        print(f"[INFO]: Setup ROS publishers for environment {env_id} using existing camera sensor")
    return camera_sensor


def publish_semantic_visualizations(node, scene, sim_time=None):
    _publish_semantic_visualizations(node, scene, CAMERA_TOPIC_MAP, sim_time)
