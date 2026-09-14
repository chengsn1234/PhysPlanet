#!/usr/bin/env python3
"""独立摩擦辨识节点。

双模式：
1. 持续模式（默认）：滑动窗口均值，持续发布 μ
2. 序列模式（触发）：收到 start_identification → 运动序列 → 稳态采集 → 辨识

参考 mars_physics_learning 的 ParamIdentifier + Planner 结构，
简化为刚性地形 Coulomb 模型。
"""

import argparse
import csv
import math
import os
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, Float32MultiArray, Float64, String

import yaml


# ── Coulomb 辨识（纯函数，不依赖仿真代码） ──────────────────────

def compute_mu_coulomb(mr, fn, wheel_radius, slip, slip_threshold=0.05):
    """Coulomb: μ = |τ| / (r · Fn) when slipping."""
    if abs(slip) < slip_threshold or fn < 1e-6:
        return 0.0
    return abs(mr) / (wheel_radius * fn)


def compute_traction_score(mr, fn, wheel_radius):
    """Equivalent traction score: |τ| / (r · Fn), independent of slip."""
    if fn < 1e-6:
        return 0.0
    return abs(mr) / (wheel_radius * fn)


def iqr_filter(values, k=1.5):
    """IQR 异常值过滤。全异常则返回原列表。"""
    if len(values) < 4:
        return list(values)
    arr = np.array(values, dtype=np.float64)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    if iqr < 1e-12:
        return list(values)
    lo, hi = q1 - k * iqr, q3 + k * iqr
    filtered = [v for v in values if lo <= v <= hi]
    return filtered if filtered else list(values)


def aggregate_values(values, method="median"):
    """聚合：median 抗异常值，mean 兼容旧逻辑。"""
    if not values:
        return 0.0
    arr = np.array(values)
    if method == "median":
        return float(np.median(arr))
    return float(np.mean(arr))


# ── 辨识节点 ─────────────────────────────────────────────────────

class FrictionIdentifier(Node):

    def __init__(self, config_path=None, robot_type="zhurong"):
        super().__init__("friction_identifier")

        # 加载配置
        if config_path is None:
            from ament_index_python.packages import get_package_share_directory
            config_path = os.path.join(
                get_package_share_directory("physplanet_utils"), "config", "friction_identifier.yaml"
            )
        with open(config_path, "r") as f:
            self._cfg = yaml.safe_load(f)

        self._robot_type = robot_type
        robot_cfg = self._cfg.get(robot_type, {})
        self._wheels = robot_cfg.get("wheels", [])
        self._front_wheels = robot_cfg.get("front_wheels", self._wheels[:2])
        self._motion_steps = robot_cfg.get("motion_steps", [])
        self._wheel_radius = float(robot_cfg.get("wheel_radius", self._cfg.get("wheel_radius", 0.15)))
        self._slip_threshold = float(self._cfg.get("slip_threshold", 0.05))
        self._force_threshold = float(self._cfg.get("force_threshold", 0.1))
        self._window_size = int(self._cfg.get("window_size", 50))
        # env_id：ROS 参数优先（launch 传入），否则回落 YAML 默认值
        self.declare_parameter("env_id", int(self._cfg.get("env_id", 0)))
        self._env_id = int(self.get_parameter("env_id").value)

        prog_cfg = self._cfg.get("progressive", {})

        # 序列模式参数
        self._min_samples = int(prog_cfg.get("min_samples_per_step", self._cfg.get("min_samples_per_step", 10)))
        self._steady_window = int(self._cfg.get("steady_window", 50))
        self._steady_vel_threshold = float(self._cfg.get("steady_vel_threshold", 0.01))
        self._sample_interval = float(self._cfg.get("sample_interval", 0.2))

        # 新增配置（带默认值，不破坏现有行为）
        self._sequence_mode = str(self._cfg.get("sequence_mode", "fixed"))
        self._prog_omega_min = float(prog_cfg.get("omega_min", 3.0))
        self._prog_omega_step = float(prog_cfg.get("omega_step", 1.0))
        self._prog_omega_max = float(prog_cfg.get("omega_max", 15.0))
        self._prog_base_rear_omega = float(prog_cfg.get("base_rear_omega", 3.0))
        self._prog_max_duration = float(prog_cfg.get("max_duration", 5.0))
        self._prog_safety_slip = float(prog_cfg.get("safety_slip", 1.0))
        qual_cfg = self._cfg.get("quality", {})
        self._qual_sliding_ratio = float(qual_cfg.get("sliding_ratio", 0.7))
        self._qual_partial_ratio = float(qual_cfg.get("partial_ratio", 0.2))
        self._qual_iqr_k = float(qual_cfg.get("iqr_multiplier", 1.5))
        self._qual_aggregate = str(qual_cfg.get("aggregate", "median"))

        # 持续模式：滑动窗口
        self._mu_windows = {w: deque(maxlen=self._window_size) for w in self._wheels}

        # slip 缓存：force callback 驱动计算 μ
        self._slip_cache = {w: 0.0 for w in self._wheels}

        # 序列模式：采集状态
        self._samples = {}          # {wheel: [(slip, fn, mr, observable_mu, traction_score), ...]}
        self._all_samples = {}
        self._collecting = False
        self._last_sample_time = {w: 0.0 for w in self._wheels}
        self._sequence_running = False

        # 稳态检测
        self._vel_history = []
        self._steady = False

        # CSV
        self._csv_writers = {}
        self._csv_dir = None
        self._csv_files = []
        csv_path = self._cfg.get("csv_path", "")
        if csv_path:
            # 相对路径 → 基于 physplanet_utils 包目录解析
            if not os.path.isabs(csv_path):
                from ament_index_python.packages import get_package_share_directory
                csv_path = os.path.join(
                    get_package_share_directory("physplanet_utils"), csv_path
                )
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._csv_dir = os.path.join(csv_path, stamp)
            os.makedirs(self._csv_dir, exist_ok=True)
            for w in self._wheels:
                f_path = os.path.join(self._csv_dir, f"raw_{w}.csv")
                fh = open(f_path, "w", newline="", buffering=1)
                writer = csv.writer(fh)
                writer.writerow(["timestamp", "slip", "fn", "mr", "observable_mu", "traction_score"])
                self._csv_writers[w] = writer
                self._csv_files.append(fh)
            self.get_logger().info(f"Writing friction CSV logs to: {self._csv_dir}")

        # 发布器
        self._omega_pub = self.create_publisher(Float32MultiArray, f"/env_{self._env_id}/wheel_omega_cmd", 10)
        self._mu_pubs = {}
        self._quality_pubs = {}
        for w in self._wheels:
            self._mu_pubs[w] = self.create_publisher(Float64, f"/env_{self._env_id}/mu_identified_{w}", 10)
            self._quality_pubs[w] = self.create_publisher(String, f"/env_{self._env_id}/mu_quality_{w}", 10)

        # 订阅 odom（序列模式稳态检测）
        self.create_subscription(Odometry, f"/env_{self._env_id}/odom", self._odom_cb, 10)

        # 订阅 wheel_force + wheel_slip（force 驱动，slip 缓存）
        # Float64 无 header，ApproximateTimeSynchronizer 无法同步，故分开设订阅
        for w in self._wheels:
            self.create_subscription(
                WrenchStamped, f"/env_{self._env_id}/wheel_force_{w}",
                lambda msg, wname=w: self._force_cb(msg, wname), 10,
            )
            self.create_subscription(
                Float64, f"/env_{self._env_id}/wheel_slip_{w}",
                lambda msg, wname=w: self._slip_cb(msg, wname), 10,
            )

        # 序列触发
        self.create_subscription(Empty, f"/env_{self._env_id}/start_identification", self._start_cb, 10)

        # 持续发布定时器（1Hz）
        self.create_timer(1.0, self._publish_windowed_mu)

        self.get_logger().info(
            f"FrictionIdentifier ready: robot={robot_type}, wheels={self._wheels}, "
            f"env_id={self._env_id}, window={self._window_size}, steps={len(self._motion_steps)}, "
            f"sequence_mode={self._sequence_mode}"
        )

    # ── 回调 ──────────────────────────────────────────────────

    def _odom_cb(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self._vel_history.append([vx, vy])
        if len(self._vel_history) > self._steady_window:
            self._vel_history = self._vel_history[-self._steady_window:]
        if len(self._vel_history) < self._steady_window:
            self._steady = False
            return
        recent = np.array(self._vel_history[-self._steady_window:])
        self._steady = bool(np.all(np.std(recent, axis=0) < self._steady_vel_threshold))

    def _slip_cb(self, msg, wheel_name):
        """缓存最新 slip ratio。"""
        self._slip_cache[wheel_name] = msg.data

    def _force_cb(self, force_msg, wheel_name):
        """force 驱动：取缓存的 slip，计算 μ。

        注意：μ 是稳态滑移下基于 applied_torque 的近似辨识。
        - net_forces_w 是 IsaacLab 定义的纯法向接触力（||F|| = Fn），不含切向/摩擦力
        - applied_torque 近似摩擦力矩，仅在稳态匀速滑移时成立
        """
        # net_forces_w 是纯法向力（不含摩擦），||F|| = Fn
        fn = math.sqrt(
            force_msg.wrench.force.x ** 2
            + force_msg.wrench.force.y ** 2
            + force_msg.wrench.force.z ** 2
        )
        mr = abs(force_msg.wrench.torque.z)
        slip = self._slip_cache.get(wheel_name, 0.0)

        if fn < self._force_threshold:
            return

        mu = compute_mu_coulomb(mr, fn, self._wheel_radius, slip, self._slip_threshold)
        traction_score = compute_traction_score(mr, fn, self._wheel_radius)

        # 持续模式：存滑动窗口
        if wheel_name in self._mu_windows:
            self._mu_windows[wheel_name].append((slip, fn, mr, mu, traction_score))

        # 序列模式：采集时额外存 _samples
        if self._collecting:
            now = time.time()
            if now - self._last_sample_time.get(wheel_name, 0.0) < self._sample_interval:
                return
            self._last_sample_time[wheel_name] = now

            if wheel_name not in self._samples:
                self._samples[wheel_name] = []
            sample = (slip, fn, mr, mu, traction_score)
            self._samples[wheel_name].append(sample)
            if wheel_name not in self._all_samples:
                self._all_samples[wheel_name] = []
            self._all_samples[wheel_name].append(sample)

            # CSV
            w = self._csv_writers.get(wheel_name)
            if w:
                w.writerow([
                    f"{now:.3f}", f"{slip:.4f}", f"{fn:.2f}", f"{mr:.3f}",
                    f"{mu:.4f}", f"{traction_score:.4f}",
                ])

    # ── 持续模式：滑动窗口发布 ──────────────────────────────────

    def _publish_windowed_mu(self):
        """定时发布滑动窗口内 μ 均值。"""
        for w in self._wheels:
            samples = [s for s in self._mu_windows[w] if s[3] > 0.0]
            if not samples:
                continue
            mu_avg = float(np.mean([s[3] for s in samples]))
            mu_msg = Float64()
            mu_msg.data = mu_avg
            self._mu_pubs[w].publish(mu_msg)

    # ── 序列模式 ──────────────────────────────────────────────

    def _start_cb(self, msg):
        """收到触发 → 执行运动序列辨识。"""
        if self._sequence_running:
            self.get_logger().warn("Sequence already running, skip")
            return
        self._sequence_running = True
        threading.Thread(target=self._sequence_worker, daemon=True).start()

    def _sequence_worker(self):
        try:
            self._run_sequence()
        finally:
            self._sequence_running = False

    def _run_sequence(self):
        """根据 sequence_mode 分发到对应模式。"""
        if self._sequence_mode == "progressive":
            self._run_progressive_sequence()
        else:
            self._run_fixed_sequence()

    def _run_fixed_sequence(self):
        """固定步序列：每 step 变速 → 等稳态 → 采集 → 辨识。"""
        self.get_logger().info(f"Starting fixed sequence: {len(self._motion_steps)} steps")

        self._samples = {}
        self._all_samples = {}

        for i, step in enumerate(self._motion_steps):
            self._samples = {}
            front = step.get("front_wheels", [])
            rear = step.get("rear_wheels", [])
            max_dur = float(step.get("max_duration", 5.0))

            # 组装 wheel_omega_cmd
            omega_cmd = self._build_omega_cmd(front, rear)
            self.get_logger().info(f"Step {i+1}/{len(self._motion_steps)}: omega_cmd={omega_cmd}")

            # 发送轮速
            self._publish_omega(omega_cmd)

            # 等稳态
            self._vel_history = []
            self._steady = False
            t0 = time.time()
            while not self._steady and (time.time() - t0) < max_dur:
                time.sleep(0.02)

            if not self._steady:
                self.get_logger().warn(f"Step {i+1}: steady timeout, collect anyway")
            else:
                self.get_logger().info(f"Step {i+1}: steady, collecting...")

            # 采集
            self._collecting = True
            self._last_sample_time = {w: 0.0 for w in self._wheels}
            t0 = time.time()
            while (time.time() - t0) < max_dur:
                time.sleep(0.02)
                counts = [len(self._samples.get(w, [])) for w in self._wheels]
                if counts and min(counts) >= self._min_samples:
                    break
            self._collecting = False

            counts = {w: len(self._samples.get(w, [])) for w in self._wheels}
            self.get_logger().info(f"Step {i+1}: collected {counts}")

        # 停车
        self._publish_omega([0.0] * len(self._wheels))
        self.get_logger().info("All steps done, stopping")

        # 辨识
        self._identify_and_publish(self._all_samples)

        # flush CSV
        for fh in self._csv_files:
            fh.flush()

    def _run_progressive_sequence(self):
        """Progressive 序列：前轮 omega 递增直到滑移，后轮固定。"""
        self.get_logger().info(
            f"Starting progressive sequence: omega [{self._prog_omega_min} -> {self._prog_omega_max}], "
            f"step={self._prog_omega_step}"
        )

        self._samples = {}
        self._all_samples = {}

        omega = self._prog_omega_min
        step = 0
        max_steps = int((self._prog_omega_max - self._prog_omega_min) / self._prog_omega_step) + 1

        while step < max_steps and omega <= self._prog_omega_max:
            self._samples = {}

            # 组装 omega_cmd：前轮=omega，后轮=base_rear_omega
            front = [omega] * len(self._front_wheels)
            rear = [self._prog_base_rear_omega] * (len(self._wheels) - len(self._front_wheels))
            omega_cmd = self._build_omega_cmd(front, rear)
            self.get_logger().info(f"Progressive step {step+1}: omega={omega:.1f}, cmd={omega_cmd}")

            # 发送轮速
            self._publish_omega(omega_cmd)

            # 等稳态
            self._vel_history = []
            self._steady = False
            t0 = time.time()
            while not self._steady and (time.time() - t0) < self._prog_max_duration:
                # 不在 worker 线程 spin：主线程 rclpy.spin 已处理回调，重复 spin 同一 node 非线程安全
                time.sleep(0.02)

            if not self._steady:
                self.get_logger().warn(f"Progressive step {step+1}: steady timeout, collect anyway")
            else:
                self.get_logger().info(f"Progressive step {step+1}: steady, collecting...")

            # 采集
            self._collecting = True
            self._last_sample_time = {w: 0.0 for w in self._wheels}
            t0 = time.time()
            while (time.time() - t0) < self._prog_max_duration:
                time.sleep(0.02)
                counts = [len(self._samples.get(w, [])) for w in self._wheels]
                if counts and min(counts) >= self._min_samples:
                    break
            self._collecting = False

            counts = {w: len(self._samples.get(w, [])) for w in self._wheels}
            self.get_logger().info(f"Progressive step {step+1}: collected {counts}")

            # _force_cb 采集时已写入 _all_samples，无需重复 merge（否则样本被计两次）
            # 安全检查：平均 |slip| 过大
            avg_slip = 0.0
            n = 0
            for w in self._wheels:
                for s in self._samples.get(w, []):
                    avg_slip += abs(s[0])
                    n += 1
            if n > 0:
                avg_slip /= n
            if avg_slip > self._prog_safety_slip:
                self.get_logger().warn(f"Progressive: avg slip={avg_slip:.3f} > safety={self._prog_safety_slip}, stopping")
                break

            # 检查前轮是否已充分滑移
            front_wheels = self._front_wheels
            front_all_sliding = True
            for w in front_wheels:
                w_samples = self._samples.get(w, [])
                if not w_samples:
                    front_all_sliding = False
                    break
                slip_arr = np.array([abs(s[0]) for s in w_samples])
                valid_ratio = float(np.mean(slip_arr >= self._slip_threshold))
                if valid_ratio < self._qual_sliding_ratio or len(w_samples) < self._min_samples:
                    front_all_sliding = False
                    break
            if front_all_sliding:
                self.get_logger().info("Progressive: front wheels sufficiently sliding, stopping early")
                break

            omega += self._prog_omega_step
            step += 1

        # 停车
        self._publish_omega([0.0] * len(self._wheels))
        self.get_logger().info("Progressive sequence done, stopping")

        # 辨识
        self._identify_and_publish(self._all_samples)

        # flush CSV
        for fh in self._csv_files:
            fh.flush()

    def _build_omega_cmd(self, front_wheels, rear_wheels):
        """根据 robot_type 构建 wheel_omega_cmd 数组。

        Zhurong: [FL, FR, ML, MR, BL, BR] → front=[FL,FR], rear=[ML,MR,BL,BR]
        Hunter:  [re_left, fr_left, re_right, fr_right] → front=[fr_left,fr_right], rear=[re_left,re_right]
        """
        if self._robot_type == "zhurong":
            # [FL, FR, ML, MR, BL, BR]
            fl, fr = front_wheels[0], front_wheels[1]
            ml, mr, bl, br = rear_wheels[0], rear_wheels[1], rear_wheels[2], rear_wheels[3]
            return [fl, fr, ml, mr, bl, br]
        elif self._robot_type == "hunter":
            # [re_left, fr_left, re_right, fr_right]
            return [rear_wheels[0], front_wheels[0], rear_wheels[1], front_wheels[1]]
        else:
            return []

    def _publish_omega(self, omega_list):
        msg = Float32MultiArray()
        msg.data = [float(v) for v in omega_list]
        self._omega_pub.publish(msg)

    # ── 辨识 + 发布 ──────────────────────────────────────────

    def _sample_stats(self, samples):
        """Return aggregate stats for one wheel's samples."""
        if not samples:
            return None
        slip_values = np.array([s[0] for s in samples], dtype=np.float32)
        fn_values = np.array([s[1] for s in samples], dtype=np.float32)
        mr_values = np.array([s[2] for s in samples], dtype=np.float32)
        observable_values = [s[3] for s in samples if s[3] > 0.0]
        traction_values = [s[4] for s in samples]
        valid_mask = np.abs(slip_values) >= self._slip_threshold
        valid_ratio = float(np.mean(valid_mask)) if len(valid_mask) else 0.0
        if valid_ratio >= self._qual_sliding_ratio:
            quality = "sliding"
        elif valid_ratio >= self._qual_partial_ratio:
            quality = "partial_slip"
        else:
            quality = "traction_only"
        # IQR 过滤 + 聚合
        filtered_observable = iqr_filter(observable_values, k=self._qual_iqr_k) if observable_values else []
        filtered_traction = iqr_filter(traction_values, k=self._qual_iqr_k) if traction_values else []
        return {
            "count": len(samples),
            "slip_avg": float(np.mean(slip_values)),
            "fn_avg": float(np.mean(fn_values)),
            "mr_avg": float(np.mean(mr_values)),
            "observable_mu": aggregate_values(filtered_observable, method=self._qual_aggregate) if filtered_observable else 0.0,
            "traction_score": aggregate_values(filtered_traction, method="mean") if filtered_traction else 0.0,
            "valid_ratio": valid_ratio,
            "quality": quality,
        }

    def _log_sample_summary(self, label, sample_dict, publish=False, write_csv=False):
        """Log per-wheel sample statistics and optionally publish/write final results."""
        self.get_logger().info(f"{label} result:")
        for w in self._wheels:
            stats = self._sample_stats(sample_dict.get(w, []))
            if stats is None:
                self.get_logger().warn(f"{label} {w}: no samples")
                continue

            self.get_logger().info(
                f"{w}: slip={stats['slip_avg']:.4f}, fn={stats['fn_avg']:.2f}, "
                f"mr={stats['mr_avg']:.3f} -> mu_obs={stats['observable_mu']:.4f}, "
                f"trac={stats['traction_score']:.4f} "
                f"({stats['quality']}, valid={stats['valid_ratio'] * 100.0:.0f}%, n={stats['count']})"
            )

            if publish:
                mu_msg = Float64()
                if stats["quality"] in ("sliding", "partial_slip"):
                    mu_msg.data = float(stats["observable_mu"])
                else:
                    mu_msg.data = float(stats["traction_score"])
                self._mu_pubs[w].publish(mu_msg)
                # 发布质量标记
                q_msg = String()
                q_msg.data = stats["quality"]
                self._quality_pubs[w].publish(q_msg)

            if write_csv and self._csv_dir:
                f_path = os.path.join(self._csv_dir, f"result_{w}.csv")
                with open(f_path, "w", newline="") as fh:
                    writer = csv.writer(fh)
                    writer.writerow([
                        "label", "count", "slip_avg", "valid_slip_ratio", "quality",
                        "fn_avg", "mr_avg", "mu_sliding", "traction_score",
                        "published_mu",
                    ])
                    published_mu = stats["observable_mu"] if stats["quality"] in ("sliding", "partial_slip") else stats["traction_score"]
                    writer.writerow([
                        label, stats["count"], f"{stats['slip_avg']:.4f}",
                        f"{stats['valid_ratio']:.4f}", stats["quality"],
                        f"{stats['fn_avg']:.2f}", f"{stats['mr_avg']:.3f}",
                        f"{stats['observable_mu']:.4f}", f"{stats['traction_score']:.4f}",
                        f"{published_mu:.4f}",
                    ])

    def _identify_and_publish(self, sample_dict):
        """Publish and log the overall sequence result."""
        self._log_sample_summary("Overall", sample_dict, publish=True, write_csv=True)


# ── main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["zhurong", "hunter"], default="zhurong")
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = FrictionIdentifier(config_path=args.config, robot_type=args.robot)

    rclpy.spin(node)

    # cleanup
    for fh in node._csv_files:
        fh.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
