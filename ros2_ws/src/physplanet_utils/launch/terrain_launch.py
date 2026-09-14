"""Shared launch helpers for one selected Isaac environment."""

import copy
import json
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


SOIL_KEYS = ("Kc", "Kphi", "n0", "n1", "c", "phi", "K")


def _load_ros2_config(context):
    """读取 ros2_config.yaml，返回（cfg_dict, resolved_path）。空字符串=包内默认。"""
    pkg_share = get_package_share_directory("physplanet_utils")
    default_path = os.path.join(pkg_share, "config", "ros2_config.yaml")
    raw = context.launch_configurations.get("ros2_config_file", "") or default_path
    config_path = os.path.expanduser(raw)
    if not os.path.isfile(config_path):
        print(f"[WARN] ros2_config not found: {config_path}, using all defaults")
        return {}, config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    print(f"[INFO] ros2_config: {config_path}")
    return cfg, config_path


def _cfg_get(cfg, *path, default=None):
    """安全嵌套取值：_cfg_get(cfg, 'perception', 'elevation_mapping')。"""
    d = cfg
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d


def make_actions(context, robot):
    env_id = int(context.launch_configurations["env_id"])
    context_file = os.path.expanduser(context.launch_configurations["context_file"])
    with open(context_file) as f:
        terrain = json.load(f)
    if terrain["robot"] != robot:
        raise RuntimeError(f"Context robot is {terrain['robot']}, not {robot}")
    if not 0 <= env_id < terrain["num_envs"]:
        raise RuntimeError(f"env_id={env_id} outside [0, {terrain['num_envs'] - 1}]")

    ros2_cfg, _ = _load_ros2_config(context)
    em_enabled = _cfg_get(ros2_cfg, "perception", "elevation_mapping", default=True)
    friction_id_enabled = _cfg_get(ros2_cfg, "identification", "friction_identifier", default=False)
    rviz_enabled = _cfg_get(ros2_cfg, "viz", "rviz", default=True)
    gt = _cfg_get(ros2_cfg, "gt_layers", default={}) or {}

    ns, frame = f"/env_{env_id}", f"env_{env_id}"
    cache = os.path.join(os.path.expanduser("~"), ".cache", "physplanet")
    os.makedirs(cache, exist_ok=True)

    def node(executable, name, **kwargs):
        return Node(package="physplanet_utils", executable=executable, name=name,
                    parameters=[{"use_sim_time": True, "env_id": env_id}, *kwargs.pop("parameters", [])], output="screen", **kwargs)

    # terrain_marker 始终启动
    actions = [
        Node(package="physplanet_utils", executable="terrain_marker_publisher.py", name="terrain_marker_publisher",
             arguments=["--terrain", terrain["terrain_name"]], parameters=[{"use_sim_time": True}], output="screen"),
    ]

    # elevation_mapping 感知主管线（em 关则跳过配置 + 所有依赖节点）
    setup_path = None
    if em_enabled:
        pkg = get_package_share_directory("elevation_mapping_cupy")
        setup = os.path.join(pkg, "config", "setups", robot)
        with open(os.path.join(setup, f"{robot}_simple.yaml")) as f:
            node_cfg = yaml.safe_load(f)
        params = node_cfg["elevation_mapping_node"]["ros__parameters"]
        params["base_frame"] = f"{frame}/base_link"
        for subscriber in params["subscribers"].values():
            subscriber["topic_name"] = subscriber["topic_name"].replace("/env_0", ns)

        terrain_dir = os.path.join(os.path.expanduser("~/isaaclab_ws/physplanet"),
                                   "source/physplanet_assets/terrain_generation/terrain", terrain["terrain_name"])

        # GT 层开关：默认全开，按 ros2_config.gt_layers 覆盖
        gt_friction = gt.get("friction_gt", True)
        gt_soil = gt.get("soil_params", True)
        gt_class = gt.get("terrain_class", True)
        gt_rock = gt.get("rock_gt", True)

        # 检查必需地图
        required_maps = []
        if gt_class:
            required_maps.append("terrain_class_map.npy")
        if gt_rock:
            required_maps.append("rock_map.npy")
        if terrain["mode"] == "hard":
            if gt_friction:
                required_maps.append("friction_map.npy")
        else:
            if gt_soil:
                required_maps.append("soil_params_map.npy")
        missing_maps = [name for name in required_maps if not os.path.isfile(os.path.join(terrain_dir, name))]
        if missing_maps:
            raise FileNotFoundError(
                f"Terrain '{terrain['terrain_name']}' is missing {', '.join(missing_maps)}. Regenerate it before launching ROS2.")

        common = {"terrain_size": terrain["terrain_size"], "map_length": params["map_length"],
                  "env_origin_xy": (0.0, 0.0)}
        plugins = {
            "min_filter": {"enable": True, "fill_nan": False, "is_height_layer": True,
                           "layer_name": "min_filter", "extra_params": {"dilation_size": 1, "iteration_n": 30}},
            "smooth_filter": {"enable": True, "fill_nan": False, "is_height_layer": True,
                              "layer_name": "smooth", "extra_params": {"input_layer_name": "min_filter"}},
            "inpainting": {"enable": True, "fill_nan": False, "is_height_layer": True,
                            "layer_name": "inpaint", "extra_params": {"method": "telea"}},
        }
        layers = []
        maps = []
        if terrain["mode"] == "hard":
            if gt_friction:
                layers.insert(0, "friction_GT")
                maps.append(("friction_GT", "friction_map.npy", None, "linear"))
        else:
            if gt_soil:
                layers = list(SOIL_KEYS) + layers
                maps = [(key, "soil_params_map.npy", index, "linear") for index, key in enumerate(SOIL_KEYS)] + maps
        if gt_class:
            layers.append("terrain_class_GT")
            maps.append(("terrain_class_GT", "terrain_class_map.npy", None, "nearest"))
        if gt_rock:
            layers.append("rock_GT")
            maps.append(("rock_GT", "rock_map.npy", None, "nearest"))
        for layer, filename, channel, interpolation in maps:
            plugins[f"terrain_gt_{layer}"] = {
                "type": "frictionGT_layer", "enable": True, "fill_nan": False, "is_height_layer": False,
                "layer_name": layer, "extra_params": {**common, "map_file": os.path.join(terrain_dir, filename),
                                                          "channel": channel, "interpolation": interpolation},
            }
        for publisher in params["publishers"].values():
            publisher["layers"] = [x for x in publisher["layers"] if x != "friction_GT"] + layers

        suffix = f"{robot}_env_{env_id}"
        setup_path = os.path.join(cache, f"mapping_{suffix}.yaml")
        plugin_path = os.path.join(cache, f"plugins_{suffix}.yaml")
        params["plugin_config_file"] = plugin_path
        with open(setup_path, "w") as f:
            yaml.safe_dump(node_cfg, f, sort_keys=False)
        with open(plugin_path, "w") as f:
            yaml.safe_dump(plugins, f, sort_keys=False)

        # em 主节点
        actions.append(Node(package="elevation_mapping_cupy", executable="elevation_mapping_node", name="elevation_mapping_node",
                            parameters=[os.path.join(pkg, "config", "core", "core_param.yaml"), setup_path, {"use_sim_time": True}], output="screen"))

        # 感知前置节点（camera_info + depth_to_pointcloud）+ local_height_scan
        if robot == "hunter":
            actions.append(node("hunter_camera_info_publisher.py", "hunter_camera_info"))
            actions.append(node("hunter_depth_to_pointcloud.py", "hunter_depth_to_pointcloud"))
            actions.append(node("hunter_local_height_scan.py", "hunter_local_height_scan", parameters=[{
                "scan_topic": f"{ns}/local_height_scan", "image_topic": f"{ns}/local_height_scan_image",
                "base_frame": f"{frame}/base_link"}]))
        else:
            actions.append(node("zhurong_camera_info_publisher.py", "zhurong_camera_info"))
            actions.append(node("zhurong_depth_to_pointcloud.py", "zhurong_depth_to_pointcloud"))
            actions.append(node("zhurong_obs_camera_info_publisher.py", "zhurong_obs_camera_info"))
            actions.append(node("zhurong_obs_depth_to_pointcloud.py", "zhurong_obs_depth_to_pointcloud"))
            actions.append(node("zhurong_local_height_scan.py", "zhurong_local_height_scan", parameters=[{
                "scan_topic": f"{ns}/local_height_scan", "image_topic": f"{ns}/local_height_scan_image",
                "base_frame": f"{frame}/base_link"}]))

    # friction_identifier（默认关）
    if friction_id_enabled:
        actions.append(node("friction_identifier.py", "friction_identifier", arguments=["--robot", robot]))

    # RViz
    launch_rviz = context.launch_configurations.get("launch_rviz", "true").lower() in ("1", "true", "yes")
    if launch_rviz and rviz_enabled:
        rviz_config = os.path.expanduser(context.launch_configurations["rviz_config"])
        if not os.path.isfile(rviz_config):
            raise FileNotFoundError(f"RViz config not found: {rviz_config}")
        resolved_rviz = _build_rviz_config(rviz_config, cache, robot, env_id, terrain["num_envs"])
        actions.append(Node(package="rviz2", executable="rviz2", name=f"rviz_{robot}_{env_id}",
                            arguments=["-d", resolved_rviz], parameters=[{"use_sim_time": True}], output="screen"))
    return actions


def _build_rviz_config(template_path, cache, robot, env_id, num_envs):
    """读模板 rviz，动态生成 num_envs 个 RobotModel display，写入 cache。

    支持两种模板结构：
    - Hunter：RobotModel 在顶层 Displays 列表
    - Zhurong：RobotModel 在 rviz_common/Group 的 sub-Displays 里
    """
    with open(template_path) as f:
        rviz_cfg = yaml.safe_load(f)

    vm = rviz_cfg.get("Visualization Manager", {})
    displays = vm.get("Displays", [])

    def _make_robot_display(template, i):
        r = copy.deepcopy(template)
        r["Name"] = f"RobotModel{i}"
        r["Description Topic"]["Value"] = f"/robot_description_env_{i}"
        r["Enabled"] = True
        r["Links"] = {"All Links Enabled": True, "Expand Joint Details": False,
                      "Expand Link Details": False, "Expand Tree": False,
                      "Link Tree Style": "Links in Alphabetic Order"}
        return r

    # 生成 num_envs 个 RobotModel
    def _gen_robots(template):
        return [_make_robot_display(template, i) for i in range(num_envs)]

    # 找模板 + 放置策略：Group 内优先，否则顶层
    group = None
    template = None
    for d in displays:
        if "Group" in d.get("Class", "") and "Displays" in d:
            sub_robot = [s for s in d["Displays"] if s.get("Class") == "rviz_default_plugins/RobotModel"]
            if sub_robot:
                group = d
                template = sub_robot[0]
                break
    if template is None:
        top_robot = [d for d in displays if d.get("Class") == "rviz_default_plugins/RobotModel"]
        if top_robot:
            template = top_robot[0]

    if template:
        new_robots = _gen_robots(template)
        if group is not None:
            # Zhurong 模式：保留 Group，替换内部 sub-displays
            non_robot = [s for s in group["Displays"] if s.get("Class") != "rviz_default_plugins/RobotModel"]
            group["Displays"] = new_robots + non_robot
        else:
            # Hunter 模式：把 RobotModel 包进新 Group
            non_robot = [d for d in displays if d.get("Class") != "rviz_default_plugins/RobotModel"]
            new_group = {"Class": "rviz_common/Group", "Enabled": True,
                         "Name": "RobotModels", "Displays": new_robots}
            displays[:] = [new_group] + non_robot

    # Target Frame 跟随选中的 env_id
    views = vm.get("Views", {})
    curr = views.get("Current")
    if isinstance(curr, dict) and "Target Frame" in curr:
        curr["Target Frame"] = f"env_{env_id}/base_link"

    resolved = os.path.join(cache, f"{robot}_env_{env_id}.rviz")
    with open(resolved, "w") as f:
        yaml.safe_dump(rviz_cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return resolved
