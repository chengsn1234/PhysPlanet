# 仿真通用工具
####################
## 功能：加载 yaml 配置、仿真时间转换、基础 slip ratio
## 约定：CLI 参数覆盖 yaml；AppLauncher 参数同步回 args_cli
####################

from __future__ import annotations

import json
import os
import tempfile

from rclpy.time import Time
import yaml


CLI_MAP = {
    "num_envs": ("sim", "num_envs"),
    "terrain": ("terrain", "name"),
    "pub_freq": ("sim", "pub_freq"),
    "debug": ("debug",),
}


####################
## 配置加载
####################

def load_sim_config(args_cli, config_file, config_dir=None):
    # 加载 yaml 配置，并用 CLI 参数覆盖。
    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

    config_path = getattr(args_cli, "config", None) or os.path.join(config_dir, config_file)
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        print(f"[INFO] Loaded config: {config_path}")
    else:
        print(f"[WARN] Config not found: {config_path}, using defaults")
        cfg = {}

    for cli_key, cfg_path in CLI_MAP.items():
        val = getattr(args_cli, cli_key, None)
        if val is None:
            continue
        d = cfg
        for k in cfg_path[:-1]:
            d = d.setdefault(k, {})
        d[cfg_path[-1]] = val

    args_cli.num_envs = cfg.get("sim", {}).get("num_envs", 4)
    args_cli.debug = cfg.get("debug", False)
    args_cli.terrain = cfg.get("terrain", {}).get("name", "terrain2")
    args_cli.pub_freq = cfg.get("sim", {}).get("pub_freq", 10)

    if not getattr(args_cli, "headless", False):
        args_cli.headless = cfg.get("sim", {}).get("headless", False)
    if not getattr(args_cli, "enable_cameras", False):
        args_cli.enable_cameras = cfg.get("sim", {}).get("enable_cameras", True)

    sim = cfg.get("sim", {})
    if sim.get("rendering") and sim.get("rendering_kit"):
        args_cli.experience = sim["rendering_kit"]

    # features 段：仿真侧传感器/发布的显式开关（缺省全开）
    features = cfg.setdefault("features", {})
    features.setdefault("height_scan", True)
    features.setdefault("height_scan_raycast", False)
    features.setdefault("gt_publish", True)
    features.setdefault("gt_image", False)
    features.setdefault("wheel_sensor", True)

    return cfg, args_cli


def load_terrain_size(terrain_dir, fallback):
    size_path = os.path.join(terrain_dir, "terrain_size.txt")
    if os.path.exists(size_path):
        with open(size_path) as f:
            return float(f.read().strip())
    return float(fallback)


def terrain_dome_light_cfg(sim_utils, terrain_dir):
    """Create the Mars or lunar dome light selected by the generated terrain profile."""
    profile_path = os.path.join(terrain_dir, "terrain_profile.json")
    sky = "mars"
    if os.path.isfile(profile_path):
        with open(profile_path) as f:
            sky = json.load(f).get("sky", sky)
    if sky == "lunar":
        # Keep the hard directional-style lunar key light, but add a weak cool
        # fill so shadowed terrain and rover details remain visible.
        return sim_utils.DomeLightCfg(color=(0.08, 0.09, 0.12), intensity=60.0)

    texture_path = os.path.join(
        os.path.dirname(os.path.dirname(terrain_dir)), "models", "textures", "mars", "mars_sky.png")
    return sim_utils.DomeLightCfg(
        color_temperature=4500.0, intensity=100.0, enable_color_temperature=True,
        texture_file=texture_path, texture_format="latlong")


def write_terrain_context(robot, mode, terrain_name, terrain_size, env_origins):
    """写给 ROS launch 的当前仿真地形信息。"""
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    runtime_dir = os.path.join(repo_dir, "ros2_ws")
    context = {
        "robot": robot, "mode": mode, "terrain_name": terrain_name,
        "terrain_size": float(terrain_size), "num_envs": int(len(env_origins)),
        "env_origins_xy": [[float(p[0]), float(p[1])] for p in env_origins.detach().cpu().tolist()],
    }
    path = os.path.join(runtime_dir, "active_terrain.json")
    fd, tmp_path = tempfile.mkstemp(prefix="active_terrain.", suffix=".json", dir=runtime_dir)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(context, f, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    print(f"[INFO] ROS terrain context: {path} ({robot}/{mode}, {terrain_name})")


####################
## ROS 时间 / 基础计算
####################

def sim_time_to_ros_time(sim_time: float) -> Time:
    sec = int(sim_time)
    nanosec = int((sim_time - sec) * 1e9)
    return Time(seconds=sec, nanoseconds=nanosec)


def compute_slip_ratio(omega, root_vel_b, wheel_radius):
    # slip ratio s = (ω·r - v) / max(|ω·r|, |v|);root_vel_b[0] 为车体前向速度
    # wheel_radius 必传(车型不同:zhurong 0.15 / hunter 0.135),不设默认值避免车型常量散落通用工具
    omega_r = float(omega) * wheel_radius
    v = float(root_vel_b[0])
    denom = max(abs(omega_r), abs(v))
    if denom < 1.0e-6:
        return 0.0
    return (omega_r - v) / denom
