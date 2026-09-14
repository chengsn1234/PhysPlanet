# -*- coding: UTF-8 -*-
"""Sim Launcher WebUI — simulation config + ROS 2 config + one-click launch.

Run with the isaac-sim python (same environment as the terrain generation webui):
    ~/IsaacLab/isaaclab.sh -p source/physplanet_assets/test/sim_webui.py
open http://localhost:7861 in a browser.
"""

import os
import re
import signal
import subprocess
import threading

import yaml

# 过滤终端控制序列（Isaac Sim 进度条 ESC[3g / ESCH 等）
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1bH")

# ---- pydantic 兼容 hack（同地形 webui）----------------------------------------
try:
    import gradio_client.utils as _gr_client_utils

    _orig_json_schema_to_python_type = _gr_client_utils._json_schema_to_python_type

    def _json_schema_to_python_type_compat(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_json_schema_to_python_type(schema, defs)

    _gr_client_utils._json_schema_to_python_type = _json_schema_to_python_type_compat
except Exception:
    pass

import gradio as gr

# ---- 路径常量 -----------------------------------------------------------------
# test/sim_webui.py → repo 根需 3 级 ..
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_ASSETS = os.path.join(_REPO, "source", "physplanet_assets")
_TEST_DIR = os.path.join(_ASSETS, "test")
_CONFIG_DIR = os.path.join(_ASSETS, "config")
_TERRAIN_DIR = os.path.join(_ASSETS, "terrain_generation", "terrain")
_ROS2_WS = os.path.join(_REPO, "ros2_ws")
_ROS2_CONFIG_PATH = os.path.join(_ROS2_WS, "src", "physplanet_utils", "config", "ros2_config.yaml")
_ISAACLAB_SH = os.path.expanduser("~/IsaacLab/isaaclab.sh")
_CONDA_SH = os.path.expanduser("~/anaconda3/etc/profile.d/conda.sh")

# 车型 → test 脚本映射
_SCRIPT_MAP = {
    ("Hunter", "Hard-ground (PhysX)"):        "03_test_hunter_mars_terrain.py",
    ("Zhurong", "Hard-ground (PhysX)"):       "04_test_zhurong_mars_terrain.py",
    ("Hunter", "Soft soil (Bekker\u2013Wong)"):  "05_test_hunter_terramechanics.py",
    ("Zhurong", "Soft soil (Bekker\u2013Wong)"): "06_test_zhurong_terramechanics.py",
}
_CONFIG_MAP = {"Hunter": "hunter_config.yaml", "Zhurong": "zhurong_config.yaml"}


# ---- 辅助 --------------------------------------------------------------------
def _scan_terrains():
    if not os.path.isdir(_TERRAIN_DIR):
        return []
    return sorted(d for d in os.listdir(_TERRAIN_DIR)
                  if os.path.isdir(os.path.join(_TERRAIN_DIR, d)))


def _load_yaml(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _default_sim_config(robot):
    path = os.path.join(_CONFIG_DIR, _CONFIG_MAP[robot])
    return _load_yaml(path), path


def _default_ros2_config():
    return _load_yaml(_ROS2_CONFIG_PATH), _ROS2_CONFIG_PATH


def _dump_yaml_preserve(path, cfg):
    """写 YAML，保留注释（先读原文件结构再覆盖值）。"""
    from ruamel.yaml import YAML
    _ry = YAML()
    _ry.preserve_quotes = True
    _ry.indent(mapping=2, sequence=4, offset=2)
    with open(path) as f:
        orig = _ry.load(f)
    if orig is None:
        orig = {}
    for section, vals in cfg.items():
        if isinstance(vals, dict) and isinstance(orig.get(section), dict):
            orig[section].update(vals)
        else:
            orig[section] = vals
    with open(path, "w") as f:
        _ry.dump(orig, f)


# ---- subprocess 管理 ---------------------------------------------------------
_proc_lock = threading.Lock()
_sim_proc = None
_ros2_proc = None


def _stream_stdout(proc, log_list, stop_event):
    """后台线程：读 stdout，过滤终端控制序列，按 \\r/\\n 分行追加到 log_list。
    Isaac Sim 进度条用 \\r 刷新，readline 会卡住，所以逐块读。"""
    if proc is None or proc.stdout is None:
        return
    buf = ""
    try:
        while not stop_event.is_set():
            # 用 select 超时读取，避免永久阻塞
            import select
            ready, _, _ = select.select([proc.stdout], [], [], 0.5)
            if not ready:
                if proc.poll() is not None:
                    break
                continue
            chunk = proc.stdout.read(256)
            if not chunk:
                break
            buf += chunk
            # \r 和 \n 都当行分隔
            lines = buf.split("\n")
            buf = lines.pop()  # 最后一段可能不完整，留着
            for raw_line in lines:
                # \r 刷新的进度行只取最后一次
                parts = raw_line.split("\r")
                clean = _ANSI_RE.sub("", parts[-1]).strip()
                if clean:
                    log_list.append(clean)
        # flush 残留
        if buf.strip():
            clean = _ANSI_RE.sub("", buf.split("\r")[-1]).strip()
            if clean:
                log_list.append(clean)
    except (ValueError, OSError):
        pass


def _kill_proc(proc):
    """强杀整个进程组（isaaclab.sh 有多层子进程，SIGTERM 杀不干净）。"""
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _clean_env():
    """清除 isaac-sim python 注入的 PYTHONHOME/PYTHONPATH。

    WebUI 跑在 isaac-sim python 下，sys.path / 环境变量指向 3.11。
    subprocess 里的 conda / isaaclab.sh 各自管理 python 环境，
    继承这些变量会导致 SRE module mismatch（conda 是 3.12 加载了 3.11 的 re）。
    """
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH"):
        env.pop(key, None)
    env["PYTHONUNBUFFERED"] = "1"  # 禁用子进程输出缓冲，日志实时刷新
    return env


def _wait_first_output(log_list, timeout=5.0):
    """等待子进程首行输出（最多 timeout 秒），用于返回前让 UI 有内容显示。"""
    timer = threading.Event()
    elapsed = 0.0
    while elapsed < timeout:
        if len(log_list) > 1:  # 第一行是 $ cmd，第二行起是真实输出
            break
        timer.wait(0.2)
        elapsed += 0.2


# ---- 日志共享（定义在使用之前）------------------------------------------------
_sim_logs = []
_ros2_logs = []


def _logs_to_text(logs, max_lines=200):
    if not logs:
        return ""
    return "\n".join(logs[-max_lines:])


# ---- 仿真侧 ------------------------------------------------------------------
def _load_terrain_size(terrain_name):
    """从 terrain_size.txt 或 config fallback 读地形尺寸。"""
    size_path = os.path.join(_TERRAIN_DIR, terrain_name, "terrain_size.txt")
    if os.path.isfile(size_path):
        with open(size_path) as f:
            return float(f.read().strip())
    # fallback: 从 config YAML
    return None


def save_sim_config(robot, physics, terrain, num_envs, env_spacing, headless, enable_cameras,
                    pub_freq, features_cb, debug):
    cfg, path = _default_sim_config(robot)
    cfg.setdefault("terrain", {})["name"] = terrain
    cfg.setdefault("sim", {})["num_envs"] = int(num_envs)
    cfg["sim"]["env_spacing"] = float(env_spacing)
    cfg["sim"]["headless"] = bool(headless)
    cfg["sim"]["enable_cameras"] = bool(enable_cameras)
    cfg["sim"]["pub_freq"] = int(pub_freq)
    feats = cfg.setdefault("features", {})
    for key in ("height_scan", "height_scan_raycast", "gt_publish", "gt_image", "wheel_sensor"):
        feats[key] = key in features_cb
    cfg["debug"] = bool(debug)
    _dump_yaml_preserve(path, cfg)
    script = _SCRIPT_MAP[(robot, physics)]
    return f"✅ Saved {path}\n   script={script}\n   terrain={terrain}  num_envs={num_envs}"


def start_sim(robot, physics, terrain, num_envs, env_spacing, headless, enable_cameras, pub_freq, features_cb, debug):
    """先保存配置到 YAML（UI 状态→唯一真相源），再启动仿真子进程。
    返回 (log_text, status_text) — 绑定到 [sim_log, sim_status]。"""
    global _sim_proc
    with _proc_lock:
        if _sim_proc is not None and _sim_proc.poll() is None:
            return _logs_to_text(_sim_logs), "⚠️ Simulation already running — stop it first"

    # 边界检查：GridCloner 是 2D 网格排列（ceil(sqrt(N)) × ceil(sqrt(N))）
    # 所有 env 必须在 terrain 范围内：grid_side × spacing ≤ terrain_size
    num_envs = int(num_envs)
    env_spacing = float(env_spacing)
    terrain_size = _load_terrain_size(terrain)
    if terrain_size is not None:
        import math
        grid_side = math.ceil(math.sqrt(num_envs))
        total_span = grid_side * env_spacing
        if total_span > terrain_size:
            max_grid_side = int(terrain_size // env_spacing)
            max_envs = max_grid_side * max_grid_side
            return ("", f"❌ {num_envs} envs form a {grid_side}x{grid_side} grid; "
                    f"span {total_span:.0f} m > terrain size {terrain_size} m."
                    f"with env_spacing={env_spacing} m at most {max_envs} envs fit")

    # 先把 UI 当前状态写入 YAML，确保子进程读到的就是用户选的
    save_sim_config(robot, physics, terrain, num_envs, env_spacing, headless, enable_cameras, pub_freq, features_cb, debug)
    script = _SCRIPT_MAP[(robot, physics)]
    script_path = os.path.join(_TEST_DIR, script)
    # 不传 headless/enable_cameras CLI 参数——YAML 已写入 UI 状态，load_sim_config 会读
    cmd_args = f'--terrain {terrain} --num_envs {int(num_envs)}'
    full_cmd = (
        f'source {_CONDA_SH} && conda activate env_isaaclab && '
        f'source {_ROS2_WS}/install/setup.bash && '
        f'source /opt/ros/ros_py311/install/setup.bash && '
        f'{_ISAACLAB_SH} -p {script_path} {cmd_args}'
    )
    _sim_logs.clear()
    _sim_logs.append(f"$ {full_cmd}")
    try:
        _sim_proc = subprocess.Popen(
            ["stdbuf", "-oL", "-eL", "bash", "-c", full_cmd],
            preexec_fn=os.setsid,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            env=_clean_env(),
        )
    except Exception as exc:
        _sim_logs.append(f"❌ Failed to start: {exc}")
        return _logs_to_text(_sim_logs), f"❌ Failed to start: {exc}"

    stop_event = threading.Event()
    t = threading.Thread(target=_stream_stdout, args=(_sim_proc, _sim_logs, stop_event), daemon=True)
    t.start()
    _wait_first_output(_sim_logs)
    return _logs_to_text(_sim_logs), "🚀 Starting simulation…"


def stop_sim():
    """返回 (log_text, status_text) — 绑定到 [sim_log, sim_status]。"""
    global _sim_proc
    with _proc_lock:
        if _sim_proc is None or _sim_proc.poll() is not None:
            return _logs_to_text(_sim_logs), "(simulation not running)"
        _kill_proc(_sim_proc)
        _sim_logs.append("[stop] simulation process terminated")
        _sim_proc = None
    return _logs_to_text(_sim_logs), "⏹ Simulation stopped"


def poll_sim_log():
    return _logs_to_text(_sim_logs)


# ---- ROS2 侧 -----------------------------------------------------------------
def save_ros2_config(ros2_features_cb, gt_cb, env_id):
    cfg = {
        "perception": {"elevation_mapping": "elevation_mapping" in ros2_features_cb},
        "identification": {"friction_identifier": "friction_identifier" in ros2_features_cb},
        "viz": {"rviz": "rviz" in ros2_features_cb},
        "gt_layers": {
            "friction_gt": "friction_gt" in gt_cb,
            "soil_params": "soil_params" in gt_cb,
            "terrain_class": "terrain_class" in gt_cb,
            "rock_gt": "rock_gt" in gt_cb,
        },
    }
    _dump_yaml_preserve(_ROS2_CONFIG_PATH, cfg)
    return f"✅ Saved {_ROS2_CONFIG_PATH}\n   env_id={int(env_id)}"


def start_ros2(robot, env_id, ros2_features_cb, gt_cb):
    """先保存 ROS2 配置到 YAML，再启动。返回 (log_text, status_text)。"""
    global _ros2_proc
    with _proc_lock:
        if _ros2_proc is not None and _ros2_proc.poll() is None:
            return _logs_to_text(_ros2_logs), "⚠️ ROS 2 already running — stop it first"
    # 先保存 UI 状态到 YAML
    save_ros2_config(ros2_features_cb, gt_cb, env_id)
    launch_file = f"{robot.lower()}_elevation_mapping.launch.py"
    full_cmd = (
        f'source {_CONDA_SH} && conda activate ros2_env && '
        f'source /opt/ros/humble/setup.bash && '
        f'source {_ROS2_WS}/install/setup.bash && '
        f'ros2 launch physplanet_utils {launch_file} '
        f'env_id:={int(env_id)} ros2_config_file:={_ROS2_CONFIG_PATH}'
    )
    _ros2_logs.clear()
    _ros2_logs.append(f"$ {full_cmd}")
    try:
        _ros2_proc = subprocess.Popen(
            ["stdbuf", "-oL", "-eL", "bash", "-c", full_cmd],
            preexec_fn=os.setsid,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            env=_clean_env(),
        )
    except Exception as exc:
        _ros2_logs.append(f"❌ Failed to start: {exc}")
        return _logs_to_text(_ros2_logs), f"❌ Failed to start: {exc}"

    stop_event = threading.Event()
    t = threading.Thread(target=_stream_stdout, args=(_ros2_proc, _ros2_logs, stop_event), daemon=True)
    t.start()
    _wait_first_output(_ros2_logs)
    return _logs_to_text(_ros2_logs), "🚀 Starting ROS 2…"


def stop_ros2():
    """返回 (log_text, status_text) — 绑定到 [ros2_log, ros2_status]。"""
    global _ros2_proc
    with _proc_lock:
        if _ros2_proc is None or _ros2_proc.poll() is not None:
            return _logs_to_text(_ros2_logs), "(ROS 2 not running)"
        _kill_proc(_ros2_proc)
        _ros2_logs.append("[stop] ROS 2 process terminated")
        _ros2_proc = None
    return _logs_to_text(_ros2_logs), "⏹ ROS 2 stopped"


def poll_ros2_log():
    return _logs_to_text(_ros2_logs)


# ---- 加载初始 UI 值 ----------------------------------------------------------
_FEAT_DEFAULTS = {"height_scan": True, "height_scan_raycast": False,
                  "gt_publish": True, "gt_image": False, "wheel_sensor": True}
def _initial_sim_values():
    cfg, _ = _default_sim_config("Hunter")
    sim = cfg.get("sim", {})
    feats = cfg.get("features", {})
    return (
        "Hunter",                                                               # [0] robot
        "Soft soil (Bekker\u2013Wong)",                                           # [1] physics
        cfg.get("terrain", {}).get("name", "terrain_2"),                         # [2] terrain
        sim.get("num_envs", 1),                                                  # [3] num_envs
        sim.get("env_spacing", 30.0),                                            # [4] env_spacing
        sim.get("headless", False),                                              # [5] headless
        sim.get("enable_cameras", True),                                         # [6] cameras
        sim.get("pub_freq", 10),                                                 # [7] pub_freq
        [k for k in _FEAT_DEFAULTS if feats.get(k, _FEAT_DEFAULTS[k])],          # [8] features
        cfg.get("debug", False),                                                 # [9] debug
    )


def _initial_ros2_values():
    cfg, _ = _default_ros2_config()
    perception = cfg.get("perception", {})
    ident = cfg.get("identification", {})
    viz = cfg.get("viz", {})
    gt = cfg.get("gt_layers", {})
    ros2_features = []
    if perception.get("elevation_mapping", True):
        ros2_features.append("elevation_mapping")
    if ident.get("friction_identifier", False):
        ros2_features.append("friction_identifier")
    if viz.get("rviz", True):
        ros2_features.append("rviz")
    gt_features = [k for k in ("friction_gt", "soil_params", "terrain_class", "rock_gt") if gt.get(k, True)]
    return ros2_features, gt_features, 0


def _on_robot_change(robot):
    """切换车型时刷新默认 config 值。返回顺序匹配 [sim_terrain, sim_num_envs, sim_env_spacing, sim_headless, sim_cameras, sim_pub_freq, sim_features, sim_debug]。"""
    cfg, _ = _default_sim_config(robot)
    sim = cfg.get("sim", {})
    feats = cfg.get("features", {})
    terrain_name = cfg.get("terrain", {}).get("name", "terrain_2")
    return (
        gr.update(value=terrain_name),
        sim.get("num_envs", 1),
        sim.get("env_spacing", 30.0),
        sim.get("headless", False),
        sim.get("enable_cameras", True),
        sim.get("pub_freq", 10),
        [k for k in _FEAT_DEFAULTS if feats.get(k, _FEAT_DEFAULTS[k])],
        cfg.get("debug", False),
    )


# ---- UI 构建 -----------------------------------------------------------------
with gr.Blocks(title="PhysPlanet Sim Launcher") as demo:
    gr.Markdown("# PhysPlanet Simulation Configuration & Launcher")

    sv = _initial_sim_values()
    rv = _initial_ros2_values()

    with gr.Row():
        # ===== 左栏：仿真侧 =====
        with gr.Column():
            gr.Markdown("## Simulation")
            sim_robot = gr.Radio(choices=["Hunter", "Zhurong"], value=sv[0], label="Robot")
            sim_physics = gr.Radio(
                choices=["Hard-ground (PhysX)", "Soft soil (Bekker\u2013Wong)"], value=sv[1], label="Physics model")
            sim_terrain = gr.Dropdown(choices=_scan_terrains(), value=sv[2], label="Terrain")
            with gr.Row():
                sim_num_envs = gr.Number(value=sv[3], label="num_envs", precision=0)
                sim_env_spacing = gr.Number(value=sv[4], label="env_spacing (m)", precision=1)
                sim_pub_freq = gr.Number(value=sv[7], label="pub_freq (Hz)", precision=0)
            with gr.Row():
                sim_headless = gr.Checkbox(value=sv[5], label="headless")
                sim_cameras = gr.Checkbox(value=sv[6], label="enable_cameras")
                sim_debug = gr.Checkbox(value=sv[9], label="debug")
            sim_features = gr.CheckboxGroup(
                choices=["height_scan", "height_scan_raycast", "gt_publish", "gt_image", "wheel_sensor"],
                value=sv[8],
                label="Features")
            with gr.Row():
                sim_save_btn = gr.Button("Save simulation config")
                sim_start_btn = gr.Button("Start simulation", variant="primary")
                sim_stop_btn = gr.Button("Stop simulation", variant="stop")
            sim_status = gr.Textbox(label="Status", lines=1)
            sim_log = gr.Textbox(label="Simulation log", lines=12, max_lines=30, interactive=False)

        # ===== 右栏：ROS2 侧 =====
        with gr.Column():
            gr.Markdown("## ROS 2")
            gr.Markdown("⚠️ Start the simulation first (it writes `active_terrain.json`), then launch ROS 2")
            ros2_env_id = gr.Number(value=rv[2], label="env_id", precision=0)
            ros2_features = gr.CheckboxGroup(
                choices=["elevation_mapping", "friction_identifier", "rviz"],
                value=rv[0], label="Features")
            ros2_gt = gr.CheckboxGroup(
                choices=["friction_gt", "soil_params", "terrain_class", "rock_gt"],
                value=rv[1], label="Ground-truth layers")
            with gr.Row():
                ros2_save_btn = gr.Button("Save ROS 2 config")
                ros2_start_btn = gr.Button("Start ROS 2", variant="primary")
                ros2_stop_btn = gr.Button("Stop ROS 2", variant="stop")
            ros2_status = gr.Textbox(label="Status", lines=1)
            ros2_log = gr.Textbox(label="ROS 2 log", lines=12, max_lines=30, interactive=False)

    # ===== 事件绑定 =====
    sim_robot.change(
        _on_robot_change, sim_robot,
        [sim_terrain, sim_num_envs, sim_env_spacing, sim_headless, sim_cameras, sim_pub_freq, sim_features, sim_debug])

    sim_save_btn.click(
        save_sim_config,
        [sim_robot, sim_physics, sim_terrain, sim_num_envs, sim_env_spacing, sim_headless, sim_cameras,
         sim_pub_freq, sim_features, sim_debug],
        sim_status)

    sim_start_btn.click(
        start_sim,
        [sim_robot, sim_physics, sim_terrain, sim_num_envs, sim_env_spacing, sim_headless,
         sim_cameras, sim_pub_freq, sim_features, sim_debug],
        [sim_log, sim_status])

    sim_stop_btn.click(stop_sim, outputs=[sim_log, sim_status])

    ros2_save_btn.click(
        save_ros2_config,
        [ros2_features, ros2_gt, ros2_env_id],
        ros2_status)

    ros2_start_btn.click(
        start_ros2,
        [sim_robot, ros2_env_id, ros2_features, ros2_gt],
        [ros2_log, ros2_status])

    ros2_stop_btn.click(stop_ros2, outputs=[ros2_log, ros2_status])

    # 日志定时刷新（Gradio 4.x 用 gr.Timer）
    log_timer = gr.Timer(value=1)
    log_timer.tick(poll_sim_log, outputs=sim_log)
    log_timer.tick(poll_ros2_log, outputs=ros2_log)


if __name__ == "__main__":
    import atexit
    atexit.register(lambda: (_kill_proc(_sim_proc), _kill_proc(_ros2_proc)))
    demo.queue().launch(server_name="localhost", server_port=7861)
