# -*- coding: UTF-8 -*-
"""离线脚本：从 PNG heightmap 生成 IsaacLab 可用的火星地形 USD。

用法：
    conda activate env_isaaclab
    cd /path/to/physplanet
    python source/physplanet_assets/terrain_generation/generate_terrain.py --hm 1 --output terrain2

    # 使用自定义配置文件
    python source/physplanet_assets/terrain_generation/generate_terrain.py --config my_config.yaml --output terrain3

    # CLI 参数覆盖 YAML 默认值
    python source/physplanet_assets/terrain_generation/generate_terrain.py --terrain_height 0.05 --rock_k 0.03

输出到：source/physplanet_assets/terrain_generation/terrain/<output>/
  - terrain_only.usd / terrain_only_nocollide.usd
  - rocks_merged.usd
  - terrain_merged.usd

实际逻辑拆到 terrain_generation/scripts/ 包（config/terrain/rocks/usd_export/build）。
本文件是 CLI 入口 shim。
"""

import argparse
import os
import sys

# 把 source/physplanet_assets/ 加入 path 以 import terrain_generation 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terrain_generation.scripts.config import load_config
from terrain_generation.scripts.build import generate_terrain


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mars terrain USD from PNG heightmap")
    parser.add_argument("--config", type=str, default=None, help="YAML config path (default: terrain_config.yaml)")
    parser.add_argument("--output", type=str, default="terrain2", help="Output terrain directory name")
    # CLI 覆盖参数，None 表示未传（使用 YAML 默认值）
    parser.add_argument("--hm", type=int, default=None)
    parser.add_argument("--terrain_size", type=float, default=None)
    parser.add_argument("--terrain_height", type=float, default=None, help="Max terrain height in meters")
    parser.add_argument("--terrain_downsample", type=int, default=None, help="Heightmap downsample factor")
    parser.add_argument("--gt_resolution", type=float, default=None, help="GT map resolution in meters")
    parser.add_argument("--rock_downsample", type=float, default=None, help="Rock triangle reduction factor")
    parser.add_argument("--rock_k", type=float, default=None, help="Rock density parameter")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args)
    # --output 未传时从 yaml 的 terrain_name 字段取
    output_name = args.output if args.output != parser.get_default("output") else cfg.get("terrain_name", args.output)
    generate_terrain(cfg, output_name)
