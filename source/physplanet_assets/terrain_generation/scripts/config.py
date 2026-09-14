"""地形生成配置:YAML 加载 + 默认值 + 纹理层归一化。"""

import os

import yaml


# 默认 5 层纹理（从低到高），同时用于 _normalize_texture_layers 和 load_config
_DEFAULT_TEXTURE_LAYERS = [
    {"name": "layer_1", "path": "textures/mars/1.jpg", "static_friction": 0.05, "dynamic_friction": 0.03, "restitution": 0.0,
     "soil": {"Kc": 13600.0, "Kphi": 2259100.0, "n0": 0.95, "n1": 0.5, "c": 462.3, "phi": 0.25, "K": 0.014}},
    {"name": "layer_2", "path": "textures/mars/2.jpg", "static_friction": 0.3, "dynamic_friction": 0.2, "restitution": 0.0,
     "soil": {"Kc": 13600.0, "Kphi": 2259100.0, "n0": 0.87, "n1": 0.45, "c": 462.3, "phi": 0.45, "K": 0.014}},
    {"name": "layer_3", "path": "textures/mars/3.jpg", "static_friction": 2.0, "dynamic_friction": 1.5, "restitution": 0.0,
     "soil": {"Kc": 13600.0, "Kphi": 2259100.0, "n0": 0.80, "n1": 0.4, "c": 462.3, "phi": 0.61, "K": 0.014}},
    {"name": "layer_4", "path": "textures/mars/4.jpg", "static_friction": 8.0, "dynamic_friction": 6.0, "restitution": 0.0,
     "soil": {"Kc": 13600.0, "Kphi": 2259100.0, "n0": 0.72, "n1": 0.35, "c": 462.3, "phi": 0.80, "K": 0.014}},
    {"name": "layer_5", "path": "textures/mars/5.jpg", "static_friction": 8.0, "dynamic_friction": 6.0, "restitution": 0.0,
     "soil": {"Kc": 13600.0, "Kphi": 2259100.0, "n0": 0.65, "n1": 0.3, "c": 462.3, "phi": 1.00, "K": 0.014}},
]


# soil_params_map 7 通道顺序 + 默认土参数（对齐 utils/terramechanics.py 的 SOIL_KEYS / SOIL，
# 离线脚本不 import 运行时模块；层无 soil 配置时用 _SOIL_DEFAULT 回填）
SOIL_KEYS = ("Kc", "Kphi", "n0", "n1", "c", "phi", "K")
_SOIL_DEFAULT = {"Kc": 13600.0, "Kphi": 2259100.0, "n0": 0.92, "n1": 0.5, "c": 462.3, "phi": 0.61, "K": 0.014}


def _deep_fill_defaults(cfg, defaults):
    """Recursively fill missing values from defaults into cfg."""
    if not isinstance(defaults, dict):
        return cfg if cfg is not None else defaults
    if not isinstance(cfg, dict):
        cfg = {}
    merged = dict(cfg)
    for key, value in defaults.items():
        merged[key] = _deep_fill_defaults(merged.get(key), value)
    return merged


def _normalize_texture_layers(raw_layers):
    """Normalize texture layer config to a dict list with path + physics fields."""
    if not raw_layers:
        return [layer.copy() for layer in _DEFAULT_TEXTURE_LAYERS]

    normalized = []
    for i, layer in enumerate(raw_layers):
        base = _DEFAULT_TEXTURE_LAYERS[min(i, len(_DEFAULT_TEXTURE_LAYERS) - 1)].copy()
        base["name"] = f"layer_{i}"
        if isinstance(layer, str):
            base["path"] = layer
        elif isinstance(layer, dict):
            base.update(layer)
            base.setdefault("name", f"layer_{i}")
        else:
            raise TypeError(f"Unsupported texture_layers[{i}] type: {type(layer)!r}")
        normalized.append(base)
    return normalized


def load_config(config_path, cli_args):
    """加载 YAML 配置，CLI 参数覆盖默认值。

    默认配置路径为 terrain_generation/config/terrain_config.yaml（本模块在 scripts/，默认 yaml 在上一级 config/）。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))                      # terrain_generation/scripts/
    default_config = os.path.join(os.path.dirname(script_dir), "config", "terrain_config.yaml")  # terrain_generation/config/terrain_config.yaml

    config_path = config_path or default_config
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        print(f"[INFO] Loaded config: {config_path}")
    else:
        print(f"[WARN] Config not found: {config_path}, using hardcoded defaults")
        cfg = {}

    # 默认值
    defaults = {
        "hm": 1,
        "terrain_size": 80.0,
        "terrain_height": 0.1,
        "terrain_downsample": 2,
        "gt_resolution": 0.1,
        "rock_downsample": 32,
        "rock_k": 0.07,
        "rock_seed": 42,
        "rock_sizes": [0.4, 0.8, 1.6, 3.2, 6.4],
        "sky": "mars",
        "texture_layers": [layer.copy() for layer in _DEFAULT_TEXTURE_LAYERS],
        "texture_size": 640,
        "texture_quilting": False,
    }
    cfg = _deep_fill_defaults(cfg, defaults)
    cfg["texture_layers"] = _normalize_texture_layers(cfg.get("texture_layers"))

    # CLI 覆盖（只覆盖用户显式传了的参数）
    cli_overrides = {
        "hm": cli_args.hm,
        "terrain_size": cli_args.terrain_size,
        "terrain_height": cli_args.terrain_height,
        "terrain_downsample": cli_args.terrain_downsample,
        "gt_resolution": getattr(cli_args, "gt_resolution", None),
        "rock_downsample": cli_args.rock_downsample,
        "rock_k": cli_args.rock_k,
        "rock_seed": cli_args.seed,
    }
    # argparse 对未传的参数使用 default 值，无法区分"用户传了"和"默认"
    # 用 None 作为 sentinel：用户没传的参数是 None
    for k, v in cli_overrides.items():
        if v is not None:
            cfg[k] = v

    return cfg
