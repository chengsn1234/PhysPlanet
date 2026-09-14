# 通用 ROS 控制基类
####################
## 功能：cmd_vel / wheel_omega 订阅、odom/TF 发布、控制状态缓存
## 车型差异：子类只负责 joint 解析和 compute_control_batch
####################

from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from tf2_ros import TransformBroadcaster

from sim_utils import sim_time_to_ros_time


class RoverController(Node):
    def __init__(self, node_name, robot=None, num_envs=1, device="cuda:0", wheel_count=None):
        super().__init__(node_name)
        self.num_envs = num_envs
        self.device = device
        self.robot = robot
        self.wheel_count = wheel_count  # 子类填(zhurong 6 / hunter 4),用于 wheel_omega_cb 校验

        self.linear_x = [0.0] * num_envs
        self.angular_z = [0.0] * num_envs
        # per-wheel 角速度覆写:None=走运动学,list=各轮角速度
        self._wheel_omega_override: list[None | list[float]] = [None] * num_envs

        self.tf_broadcaster = TransformBroadcaster(self)
        self._odom_publishers = {}

        self.create_subscription(Twist, "/cmd_vel", self._global_cb, 10)
        for i in range(num_envs):
            self.create_subscription(Twist, f"/env_{i}/cmd_vel", lambda msg, env_id=i: self._env_cb(msg, env_id), 10)
            self.create_subscription(Float32MultiArray, f"/env_{i}/wheel_omega_cmd", lambda msg, env_id=i: self._wheel_omega_cb(msg, env_id), 10)
            self._odom_publishers[i] = self.create_publisher(Odometry, f"/env_{i}/odom", 10)

    def _global_cb(self, msg):
        # 全局 cmd_vel 应用到所有 env。
        for i in range(self.num_envs):
            self.linear_x[i] = msg.linear.x
            self.angular_z[i] = msg.angular.z
            self._wheel_omega_override[i] = None

    def _env_cb(self, msg, env_id):
        # 单 env cmd_vel 覆盖。
        self.linear_x[env_id] = msg.linear.x
        self.angular_z[env_id] = msg.angular.z
        self._wheel_omega_override[env_id] = None

    def _wheel_omega_cb(self, msg, env_id):
        if self.wheel_count is not None and len(msg.data) != self.wheel_count:
            self.get_logger().warn(
                f"wheel_omega_cmd expects {self.wheel_count} values, got {len(msg.data)}, clearing override"
            )
            self._wheel_omega_override[env_id] = None
            return
        self._wheel_omega_override[env_id] = list(msg.data)

    def publish_tf(self, sim_time: float = None):
        # 发布所有 env 的 odom -> base_link TF;sim_time 仿真秒,必填
        if self.robot is None:
            return
        stamp = sim_time_to_ros_time(sim_time).to_msg()
        positions = self.robot.data.root_pos_w
        quaternions = self.robot.data.root_quat_w

        transforms = []
        for env_id in range(self.num_envs):
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = "odom"
            transform.child_frame_id = f"env_{env_id}/base_link"

            transform.transform.translation.x = float(positions[env_id, 0])
            transform.transform.translation.y = float(positions[env_id, 1])
            transform.transform.translation.z = float(positions[env_id, 2])

            # Isaac 使用 [w, x, y, z],ROS 使用 [x, y, z, w]。
            transform.transform.rotation.x = float(quaternions[env_id, 1])
            transform.transform.rotation.y = float(quaternions[env_id, 2])
            transform.transform.rotation.z = float(quaternions[env_id, 3])
            transform.transform.rotation.w = float(quaternions[env_id, 0])
            transforms.append(transform)

        self.tf_broadcaster.sendTransform(transforms)

    def publish_odom(self, sim_time: float = None):
        # 发布每个 env 的 Odometry(位姿+速度)。
        if self.robot is None:
            return
        stamp = sim_time_to_ros_time(sim_time).to_msg()
        positions = self.robot.data.root_pos_w
        quaternions = self.robot.data.root_quat_w
        lin_vels = self.robot.data.root_lin_vel_w
        ang_vels = self.robot.data.root_ang_vel_w
        for env_id in range(self.num_envs):
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = "odom"
            odom.child_frame_id = f"env_{env_id}/base_link"
            odom.pose.pose.position.x = float(positions[env_id, 0])
            odom.pose.pose.position.y = float(positions[env_id, 1])
            odom.pose.pose.position.z = float(positions[env_id, 2])
            odom.pose.pose.orientation.x = float(quaternions[env_id, 1])
            odom.pose.pose.orientation.y = float(quaternions[env_id, 2])
            odom.pose.pose.orientation.z = float(quaternions[env_id, 3])
            odom.pose.pose.orientation.w = float(quaternions[env_id, 0])
            odom.twist.twist.linear.x = float(lin_vels[env_id, 0])
            odom.twist.twist.linear.y = float(lin_vels[env_id, 1])
            odom.twist.twist.linear.z = float(lin_vels[env_id, 2])
            odom.twist.twist.angular.x = float(ang_vels[env_id, 0])
            odom.twist.twist.angular.y = float(ang_vels[env_id, 1])
            odom.twist.twist.angular.z = float(ang_vels[env_id, 2])
            self._odom_publishers[env_id].publish(odom)

    def compute_control_batch(self):
        # 子类覆写:返回 (joint_pos_target, joint_vel_target)
        raise NotImplementedError("subclass must override compute_control_batch")
