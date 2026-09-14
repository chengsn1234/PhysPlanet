#!/usr/bin/env python3
"""摩擦辨识全话题监听 + 离线分析。

用法:
  conda activate ros2_env
  source /opt/ros/humble/setup.bash
  source ~/isaaclab_ws/physplanet/ros2_ws/install/setup.bash
  python friction_monitor.py [--env 0] [--duration 60] [--robot hunter]
"""

import argparse
import csv
import math
import os
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64

WHEELS = {
    "hunter": ["re_left", "fr_left", "re_right", "fr_right"],
    "zhurong": ["front_L", "front_R", "middle_L", "middle_R", "back_L", "back_R"],
}
WHEEL_RADIUS = {"hunter": 0.135, "zhurong": 0.15}


class FrictionMonitor(Node):
    def __init__(self, env_id=0, robot="hunter"):
        super().__init__("friction_monitor")
        self.env_id = env_id
        self.robot = robot
        self.wheels = WHEELS[robot]
        self.wheel_radius = WHEEL_RADIUS[robot]

        # {wheel: [(wall_time, sim_time, fx, fy, fz, mr, slip, mu, vx, vy), ...]}
        self.records = defaultdict(list)
        # odom cache
        self._vx = 0.0
        self._vy = 0.0
        self._odom_stamp = 0.0

        # force cache (slip/torque arrived separately)
        self._force_cache = {}   # wheel -> (fn, mr, stamp)
        self._slip_cache = {}    # wheel -> slip
        self._mu_cache = {}      # wheel -> mu

        # 订阅
        self.create_subscription(Odometry, f"/env_{env_id}/odom", self._odom_cb, 10)
        for w in self.wheels:
            self.create_subscription(
                WrenchStamped, f"/env_{env_id}/wheel_force_{w}",
                lambda msg, wn=w: self._force_cb(msg, wn), 10,
            )
            self.create_subscription(
                Float64, f"/env_{env_id}/wheel_slip_{w}",
                lambda msg, wn=w: self._slip_cb(msg, wn), 10,
            )
            self.create_subscription(
                Float64, f"/env_{env_id}/mu_identified_{w}",
                lambda msg, wn=w: self._mu_cb(msg, wn), 10,
            )

        self.start_time = time.time()
        self.get_logger().info(
            f"Monitoring env_{env_id} ({robot}), wheels={self.wheels}, "
            f"radius={self.wheel_radius}"
        )

    def _odom_cb(self, msg):
        self._vx = msg.twist.twist.linear.x
        self._vy = msg.twist.twist.linear.y
        self._odom_stamp = time.time()

    def _force_cb(self, msg, wheel):
        fn = math.sqrt(
            msg.wrench.force.x ** 2 + msg.wrench.force.y ** 2 + msg.wrench.force.z ** 2
        )
        mr = abs(msg.wrench.torque.z)
        slip = self._slip_cache.get(wheel, 0.0)
        mu = self._mu_cache.get(wheel, 0.0)
        wall_t = time.time() - self.start_time
        sim_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._force_cache[wheel] = (fn, mr, sim_t)
        self.records[wheel].append(
            (wall_t, sim_t,
             msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
             fn, mr, slip, mu, self._vx, self._vy)
        )

    def _slip_cb(self, msg, wheel):
        self._slip_cache[wheel] = msg.data

    def _mu_cb(self, msg, wheel):
        self._mu_cache[wheel] = msg.data

    # ── 输出 ──────────────────────────────────────────────

    def save_csv(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for w in self.wheels:
            rows = self.records.get(w, [])
            if not rows:
                continue
            path = os.path.join(out_dir, f"monitor_{w}.csv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "wall_t", "sim_t", "fx", "fy", "fz", "fn", "mr",
                    "slip", "mu_identified", "vx", "vy",
                ])
                writer.writerows(rows)
            self.get_logger().info(f"Saved {len(rows)} rows -> {path}")

    def print_analysis(self):
        """打印每轮统计分析。"""
        print("\n" + "=" * 80)
        print("摩擦数据统计分析")
        print("=" * 80)

        for w in self.wheels:
            rows = self.records.get(w, [])
            if not rows:
                print(f"\n[{w}] 无数据")
                continue

            arr = np.array(rows)
            wall_t, sim_t = arr[:, 0], arr[:, 1]
            fx, fy, fz = arr[:, 2], arr[:, 3], arr[:, 4]
            fn, mr = arr[:, 5], arr[:, 6]
            slip = arr[:, 7]
            mu = arr[:, 8]
            vx, vy = arr[:, 9], arr[:, 10]

            # 计算 Coulomb μ: |mr| / (r * fn)
            mu_raw = np.where(fn > 1e-3, mr / (self.wheel_radius * fn), 0.0)

            duration = wall_t[-1] - wall_t[0] if len(wall_t) > 1 else 0
            dt = np.diff(wall_t)
            hz = 1.0 / np.mean(dt) if len(dt) > 0 and np.mean(dt) > 0 else 0

            print(f"\n{'─' * 60}")
            print(f"[{w}] 采样数={len(rows)}, 时长={duration:.1f}s, 频率≈{hz:.1f}Hz")
            print(f"{'─' * 60}")

            # 1) Fn 分析
            print(f"  Fn (法向力):")
            print(f"    mean={np.mean(fn):.2f}  std={np.std(fn):.2f}  "
                  f"min={np.min(fn):.2f}  max={np.max(fn):.2f}  "
                  f"median={np.median(fn):.2f}")
            print(f"    P5={np.percentile(fn,5):.2f}  P95={np.percentile(fn,95):.2f}")

            # 2) 力分量分析（看有没有异常大分量）
            print(f"  Fx: mean={np.mean(fx):.2f} std={np.std(fx):.2f} "
                  f"min={np.min(fx):.2f} max={np.max(fx):.2f}")
            print(f"  Fy: mean={np.mean(fy):.2f} std={np.std(fy):.2f} "
                  f"min={np.min(fy):.2f} max={np.max(fy):.2f}")
            print(f"  Fz: mean={np.mean(fz):.2f} std={np.std(fz):.2f} "
                  f"min={np.min(fz):.2f} max={np.max(fz):.2f}")

            # 3) 力矩分析
            print(f"  MR (驱动力矩):")
            print(f"    mean={np.mean(mr):.3f}  std={np.std(mr):.3f}  "
                  f"min={np.min(mr):.3f}  max={np.max(mr):.3f}")

            # 4) Slip 分析
            print(f"  Slip:")
            print(f"    mean={np.mean(slip):.4f}  std={np.std(slip):.4f}  "
                  f"min={np.min(slip):.4f}  max={np.max(slip):.4f}")

            # 5) μ 分析 — 原始 Coulomb
            sliding_mask = np.abs(slip) >= 0.05
            n_sliding = int(np.sum(sliding_mask))
            print(f"  μ_raw (Coulomb |mr|/(r·Fn)):")
            print(f"    全部: mean={np.mean(mu_raw):.4f} std={np.std(mu_raw):.4f} "
                  f"min={np.min(mu_raw):.4f} max={np.max(mu_raw):.4f}")
            if n_sliding > 0:
                mu_slide = mu_raw[sliding_mask]
                print(f"    滑移(|slip|>=0.05, n={n_sliding}): "
                      f"mean={np.mean(mu_slide):.4f} std={np.std(mu_slide):.4f}")
            else:
                print(f"    无滑移样本 (|slip|>=0.05: {n_sliding})")

            # 6) μ 辨识节点输出
            mu_nonzero = mu[mu > 0]
            if len(mu_nonzero) > 0:
                print(f"  μ_identified (辨识节点):")
                print(f"    mean={np.mean(mu_nonzero):.4f} std={np.std(mu_nonzero):.4f} "
                      f"min={np.min(mu_nonzero):.4f} max={np.max(mu_nonzero):.4f}")

            # 7) 速度
            print(f"  Vx: mean={np.mean(vx):.3f} std={np.std(vx):.3f}")

            # 8) Fn-MR 相关性
            if np.std(fn) > 1e-6 and np.std(mr) > 1e-6:
                corr = np.corrcoef(fn, mr)[0, 1]
                print(f"  Fn-MR 相关系数: {corr:.3f}")

            # 9) Fn 振动分析（连续帧间变化率）
            if len(fn) > 2:
                fn_diff = np.abs(np.diff(fn))
                print(f"  Fn 帧间跳变: mean={np.mean(fn_diff):.2f} "
                      f"max={np.max(fn_diff):.2f} "
                      f"(占mean Fn比例: {np.mean(fn_diff)/max(np.mean(fn),1e-6)*100:.1f}%)")

        # 全局对比
        print(f"\n{'=' * 80}")
        print("跨轮对比")
        print(f"{'=' * 80}")
        summary = {}
        for w in self.wheels:
            rows = self.records.get(w, [])
            if not rows:
                continue
            arr = np.array(rows)
            fn, mr, slip = arr[:, 5], arr[:, 6], arr[:, 7]
            mu_raw = np.where(fn > 1e-3, mr / (self.wheel_radius * fn), 0.0)
            summary[w] = {
                "fn_mean": np.mean(fn), "fn_std": np.std(fn),
                "mr_mean": np.mean(mr),
                "mu_mean": np.mean(mu_raw), "mu_std": np.std(mu_raw),
                "slip_mean": np.mean(slip),
            }

        if summary:
            header = f"{'Wheel':>10} {'Fn_mean':>8} {'Fn_std':>8} {'MR_mean':>8} {'μ_mean':>7} {'μ_std':>7} {'slip':>7}"
            print(header)
            print("-" * len(header))
            for w, s in summary.items():
                print(f"{w:>10} {s['fn_mean']:8.2f} {s['fn_std']:8.2f} "
                      f"{s['mr_mean']:8.3f} {s['mu_mean']:7.4f} "
                      f"{s['mu_std']:7.4f} {s['slip_mean']:7.4f}")

        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=int, default=0)
    parser.add_argument("--robot", choices=["hunter", "zhurong"], default="hunter")
    parser.add_argument("--duration", type=float, default=0, help="监听秒数, 0=无限")
    parser.add_argument("--csv", type=str, default="", help="CSV输出目录")
    args = parser.parse_args()

    rclpy.init()
    node = FrictionMonitor(env_id=args.env, robot=args.robot)

    stop_flag = False

    def sig_handler(sig, frame):
        nonlocal stop_flag
        print("\n[Signal] Ctrl+C, stopping...")
        stop_flag = True

    signal.signal(signal.SIGINT, sig_handler)

    print(f"[INFO] 监听中... (Ctrl+C 停止并输出分析{'，' + str(args.duration) + 's 后自动停止' if args.duration > 0 else ''})")

    t0 = time.time()
    while not stop_flag:
        rclpy.spin_once(node, timeout_sec=0.1)
        if args.duration > 0 and (time.time() - t0) > args.duration:
            print(f"[INFO] 达到 {args.duration}s，停止")
            break

    # 输出分析
    node.print_analysis()

    # 保存 CSV
    if args.csv:
        node.save_csv(args.csv)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_dir = os.path.join("friction_monitor_data", stamp)
        node.save_csv(default_dir)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
