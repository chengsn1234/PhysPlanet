# URDF TF 发布
####################
## 功能：解析 URDF joint tree，并用 Isaac joint state 发布完整 TF
## 输出：/tf、/clock、/robot_description_env_N、/env_N/robot_pose
####################

import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from rclpy.qos import QoSProfile, DurabilityPolicy

from sim_utils import sim_time_to_ros_time


@dataclass
class JointInfo:
    name: str
    joint_type: str  # 'revolute', 'continuous', 'fixed', 'prismatic'
    parent_link: str
    child_link: str
    origin_xyz: np.ndarray  # (3,)
    origin_rpy: np.ndarray  # (3,) roll, pitch, yaw
    axis: np.ndarray  # (3,) rotation axis for revolute/continuous


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    # roll-pitch-yaw -> quaternion [x, y, z, w]
    cr, cp, cy = np.cos(roll / 2), np.cos(pitch / 2), np.cos(yaw / 2)
    sr, sp, sy = np.sin(roll / 2), np.sin(pitch / 2), np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


def axis_angle_to_quat(axis: np.ndarray, angle: float) -> Tuple[float, float, float, float]:
    # axis-angle -> quaternion [x, y, z, w]
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    half = angle / 2
    s = np.sin(half)
    w = np.cos(half)
    x, y, z = axis * s
    return (x, y, z, w)


def quat_multiply(q1: Tuple, q2: Tuple) -> Tuple:
    # quaternion multiply, [x, y, z, w]
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return (x, y, z, w)


class URDFTFPublisher:
    def __init__(self, node: Node, urdf_path: str, robot=None, num_envs: int = 1):
        # node: TF 广播用 ROS2 节点；urdf_path: URDF 路径；robot: IsaacLab Articulation；num_envs: 环境数
        self.node = node
        self.robot = robot
        self.num_envs = num_envs
        self.tf_broadcaster = TransformBroadcaster(node)

        # Clock publisher for sim_time synchronization
        self.clock_publisher = node.create_publisher(Clock, '/clock', 10)

        # Parse URDF
        self.joints: Dict[str, JointInfo] = {}
        self.link_tree: Dict[str, str] = {}  # child_link -> parent_link
        self.root_link: Optional[str] = None
        self.urdf_content: str = ""  # Store original URDF content

        self._parse_urdf(urdf_path)

        # Build joint index mapping if robot provided
        self.joint_name_to_idx: Dict[str, int] = {}
        if robot is not None:
            self.joint_name_to_idx = {name: i for i, name in enumerate(robot.joint_names)}

        # Publishers for robot_description (one per environment)
        self.robot_description_publishers: List = []
        self.robot_pose_publishers: List = []
        self._setup_robot_description_publishers()
        self._setup_robot_pose_publishers()

        print(f"[URDFTFPublisher] Parsed {len(self.joints)} joints from {urdf_path}")
        print(f"  Root link: {self.root_link}")
        print(f"  Robot joints in simulation: {len(self.joint_name_to_idx)}")

    def _parse_urdf(self, urdf_path: str):
        # 解析 URDF joint/link 层级。
        with open(urdf_path, 'r') as f:
            self.urdf_content = f.read()

        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Find all links to identify root (link with no parent)
        all_links = set()
        child_links = set()

        for link in root.findall('link'):
            all_links.add(link.get('name'))

        for joint in root.findall('joint'):
            name = joint.get('name')
            joint_type = joint.get('type', 'fixed')

            parent = joint.find('parent')
            child = joint.find('child')
            if parent is None or child is None:
                continue

            parent_link = parent.get('link')
            child_link = child.get('link')
            child_links.add(child_link)

            # Parse origin
            origin = joint.find('origin')
            if origin is not None:
                xyz = origin.get('xyz', '0 0 0')
                rpy = origin.get('rpy', '0 0 0')
                origin_xyz = np.array([float(x) for x in xyz.split()])
                origin_rpy = np.array([float(x) for x in rpy.split()])
            else:
                origin_xyz = np.zeros(3)
                origin_rpy = np.zeros(3)

            # Parse axis
            axis_elem = joint.find('axis')
            if axis_elem is not None:
                axis_xyz = axis_elem.get('xyz', '0 0 1')
                axis = np.array([float(x) for x in axis_xyz.split()])
            else:
                axis = np.array([0.0, 0.0, 1.0])

            self.joints[name] = JointInfo(
                name=name,
                joint_type=joint_type,
                parent_link=parent_link,
                child_link=child_link,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis=axis,
            )
            self.link_tree[child_link] = parent_link

        # Root link is the one that's never a child
        self.root_link = (all_links - child_links).pop() if len(all_links - child_links) == 1 else 'base_link'

    def _setup_robot_description_publishers(self):
        # Latched QoS for robot_description (persists for late subscribers)
        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        for env_id in range(self.num_envs):
            topic = f"/robot_description_env_{env_id}"
            pub = self.node.create_publisher(String, topic, qos)
            self.robot_description_publishers.append(pub)

        # Publish robot_description for each environment
        self.publish_robot_descriptions()

    def _setup_robot_pose_publishers(self):
        for env_id in range(self.num_envs):
            topic = f"/env_{env_id}/robot_pose"
            pub = self.node.create_publisher(PoseStamped, topic, 10)
            self.robot_pose_publishers.append(pub)

    def publish_robot_descriptions(self):
        for env_id, pub in enumerate(self.robot_description_publishers):
            # Add TF prefix to URDF
            modified_urdf = self._add_tf_prefix(self.urdf_content, env_id)
            msg = String()
            msg.data = modified_urdf
            pub.publish(msg)

    def _add_tf_prefix(self, urdf_content: str, env_id: int) -> str:
        # 给 link/joint 加 env 前缀，避免多环境 TF 名冲突。
        prefix = f"env_{env_id}/"

        # Parse and modify URDF
        root = ET.fromstring(urdf_content)

        # Modify link names
        for link in root.findall('link'):
            name = link.get('name')
            if name:
                link.set('name', prefix + name)

        # Modify joint names and link references
        for joint in root.findall('joint'):
            name = joint.get('name')
            if name:
                joint.set('name', prefix + name)

            parent = joint.find('parent')
            if parent is not None:
                link = parent.get('link')
                if link:
                    parent.set('link', prefix + link)

            child = joint.find('child')
            if child is not None:
                link = child.get('link')
                if link:
                    child.set('link', prefix + link)

        return ET.tostring(root, encoding='unicode')

    def publish_tf(self, env_id: int = 0, sim_time: float = None):
        # 发布单 env 的完整 TF 树；sim_time 仿真秒，必填
        if self.robot is None:
            return

        # 转换 sim_time 为 ROS 时间戳
        stamp = sim_time_to_ros_time(sim_time).to_msg()

        # Get robot base pose
        positions = self.robot.data.root_pos_w  # (num_envs, 3)
        quaternions = self.robot.data.root_quat_w  # (num_envs, 4) [w, x, y, z]

        # Get joint positions
        joint_pos = self.robot.data.joint_pos  # (num_envs, num_joints)

        transforms = []

        # 1. Publish odom -> base_link (root)
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "odom"
        t.child_frame_id = f"env_{env_id}/{self.root_link}"

        t.transform.translation.x = float(positions[env_id, 0])
        t.transform.translation.y = float(positions[env_id, 1])
        t.transform.translation.z = float(positions[env_id, 2])

        # Quaternion: Isaac [w, x, y, z] -> ROS [x, y, z, w]
        t.transform.rotation.x = float(quaternions[env_id, 1])
        t.transform.rotation.y = float(quaternions[env_id, 2])
        t.transform.rotation.z = float(quaternions[env_id, 3])
        t.transform.rotation.w = float(quaternions[env_id, 0])

        transforms.append(t)

        # 2. Publish all joints
        for joint_name, joint_info in self.joints.items():
            if joint_info.joint_type == 'fixed':
                # Fixed joint: use origin directly
                t = TransformStamped()
                t.header.stamp = stamp
                t.header.frame_id = f"env_{env_id}/{joint_info.parent_link}"
                t.child_frame_id = f"env_{env_id}/{joint_info.child_link}"

                t.transform.translation.x = joint_info.origin_xyz[0]
                t.transform.translation.y = joint_info.origin_xyz[1]
                t.transform.translation.z = joint_info.origin_xyz[2]

                qx, qy, qz, qw = rpy_to_quat(*joint_info.origin_rpy)
                t.transform.rotation.x = qx
                t.transform.rotation.y = qy
                t.transform.rotation.z = qz
                t.transform.rotation.w = qw

                transforms.append(t)

            elif joint_info.joint_type in ('revolute', 'continuous'):
                # Revolute/continuous: apply joint angle
                if joint_name not in self.joint_name_to_idx:
                    continue

                joint_idx = self.joint_name_to_idx[joint_name]
                angle = float(joint_pos[env_id, joint_idx])

                t = TransformStamped()
                t.header.stamp = stamp
                t.header.frame_id = f"env_{env_id}/{joint_info.parent_link}"
                t.child_frame_id = f"env_{env_id}/{joint_info.child_link}"

                t.transform.translation.x = joint_info.origin_xyz[0]
                t.transform.translation.y = joint_info.origin_xyz[1]
                t.transform.translation.z = joint_info.origin_xyz[2]

                q_origin = rpy_to_quat(*joint_info.origin_rpy)
                q_joint = axis_angle_to_quat(joint_info.axis, angle)
                qx, qy, qz, qw = quat_multiply(q_origin, q_joint)

                t.transform.rotation.x = qx
                t.transform.rotation.y = qy
                t.transform.rotation.z = qz
                t.transform.rotation.w = qw

                transforms.append(t)

            elif joint_info.joint_type == 'prismatic':
                # Prismatic: apply linear displacement along axis
                if joint_name not in self.joint_name_to_idx:
                    continue

                joint_idx = self.joint_name_to_idx[joint_name]
                displacement = float(joint_pos[env_id, joint_idx])

                t = TransformStamped()
                t.header.stamp = stamp
                t.header.frame_id = f"env_{env_id}/{joint_info.parent_link}"
                t.child_frame_id = f"env_{env_id}/{joint_info.child_link}"

                t.transform.translation.x = joint_info.origin_xyz[0] + joint_info.axis[0] * displacement
                t.transform.translation.y = joint_info.origin_xyz[1] + joint_info.axis[1] * displacement
                t.transform.translation.z = joint_info.origin_xyz[2] + joint_info.axis[2] * displacement

                qx, qy, qz, qw = rpy_to_quat(*joint_info.origin_rpy)
                t.transform.rotation.x = qx
                t.transform.rotation.y = qy
                t.transform.rotation.z = qz
                t.transform.rotation.w = qw

                transforms.append(t)

        self.tf_broadcaster.sendTransform(transforms)

    def publish_robot_pose(self, env_id: int = 0, sim_time: float = None):
        # 发布单 env 的根位姿 PoseStamped；sim_time 必填
        if self.robot is None or env_id >= len(self.robot_pose_publishers):
            return

        # 转换 sim_time 为 ROS 时间戳
        stamp = sim_time_to_ros_time(sim_time).to_msg()

        positions = self.robot.data.root_pos_w  # (num_envs, 3)
        quaternions = self.robot.data.root_quat_w  # (num_envs, 4) [w, x, y, z]

        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "odom"
        msg.pose.position.x = float(positions[env_id, 0])
        msg.pose.position.y = float(positions[env_id, 1])
        msg.pose.position.z = float(positions[env_id, 2])
        msg.pose.orientation.x = float(quaternions[env_id, 1])
        msg.pose.orientation.y = float(quaternions[env_id, 2])
        msg.pose.orientation.z = float(quaternions[env_id, 3])
        msg.pose.orientation.w = float(quaternions[env_id, 0])
        self.robot_pose_publishers[env_id].publish(msg)

    def publish_tf_all_envs(self, sim_time: float = None):
        # 发布所有 env 的 TF 树 + /clock；sim_time 必填
        # 发布 /clock topic，让所有 ROS2 节点使用 sim_time
        clock_msg = Clock()
        clock_msg.clock = sim_time_to_ros_time(sim_time).to_msg()
        self.clock_publisher.publish(clock_msg)

        for env_id in range(self.num_envs):
            self.publish_tf(env_id, sim_time)
            self.publish_robot_pose(env_id, sim_time)
