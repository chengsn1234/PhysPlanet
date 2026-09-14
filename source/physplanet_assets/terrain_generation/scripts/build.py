"""地形构建主流程:heightmap → terrain + rocks USD → RViz 导出。"""

import json
import os

import numpy as np
import trimesh

from .terrain import (heightmap_to_gt, heightmap_to_mesh, bake_blended_texture, write_layer_debug_image,
                      build_region_fields, build_region_soil_profiles,
                      write_friction_map, write_soil_params_map, write_terrain_class_map,
                      write_region_soil_params_map,
                      write_rock_map, write_height_merged_map, write_terrain_rocks_preview,
                      write_texture_normal_map, write_perlin_heightmap)
from .usd_export import write_layered_terrain_usd, write_mesh_usd, write_rocks_usd, _export_dae_with_texture
from .rocks import calculate_rock_distribution, load_rocks_with_materials, simplify_rock_groups


def _build_terrain_mesh(cfg, output_dir, materials_dir):
    # heightmap → mesh → 贴图烘焙 → terrain_only.usd
    ## 解析配置
    hm_number = int(cfg["hm"])
    terrain_size = float(cfg["terrain_size"])
    terrain_height = float(cfg["terrain_height"])
    height_scale = terrain_height / (2**16 - 1)
    terrain_downsample = int(cfg.get("terrain_downsample", 1))
    gt_resolution = float(cfg.get("gt_resolution", 0.1))
    texture_layers = cfg.get("texture_layers", [])
    texture_resolution = cfg.get("texture_resolution")
    tex_size = (max(1, round(terrain_size / float(texture_resolution)))
                if texture_resolution is not None else int(cfg.get("texture_size", 640)))
    texture_quilting = bool(cfg.get("texture_quilting", False))
    tex_layers_abs = [os.path.join(materials_dir, layer["path"]) for layer in texture_layers]

    heightmap_source = cfg.get("heightmap_source", "image")
    heightmap_path = cfg.get("heightmap_path")
    if heightmap_source == "perlin":
        hm_path = os.path.join(output_dir, "heightmap.png")
        write_perlin_heightmap(
            hm_path,
            resolution=cfg.get("perlin_resolution", 512),
            seed=cfg.get("perlin_seed", 42),
            scale=cfg.get("perlin_scale", 6.0),
            octaves=cfg.get("perlin_octaves", 4),
            persistence=cfg.get("perlin_persistence", 0.5),
            lacunarity=cfg.get("perlin_lacunarity", 2.0),
        )
    elif heightmap_path:
        hm_path = heightmap_path
    else:
        heightmap_dir = cfg.get("heightmap_dir") or os.path.join(materials_dir, "heightmaps")
        hm_path = os.path.join(heightmap_dir, f"HM{hm_number}.png")
        if not os.path.exists(hm_path):
            available = sorted([f for f in os.listdir(heightmap_dir) if f.endswith('.png')])
            if not available:
                raise FileNotFoundError(f"No PNG files found in {heightmap_dir}")
            hm_path = os.path.join(heightmap_dir, available[0])
            print(f"[WARN] HM{hm_number}.png not found, using {available[0]}")

    print(f"[INFO] Processing {hm_path} -> {output_dir}")
    print(f"  Terrain size: {terrain_size}m, Height: {terrain_height}m (scale: {height_scale:.6f})")

    ## 生成地形 mesh（共享顶点 + UV）
    vertices, triangles, heights, uvs = heightmap_to_mesh(
        hm_path, terrain_size, height_scale, downsample=terrain_downsample)
    gt_heights = heightmap_to_gt(hm_path, terrain_size, height_scale, gt_resolution)
    print(f"  GT resolution: {gt_resolution}m ({gt_heights.shape[1]}x{gt_heights.shape[0]})")

    ## 层控制场：区域实验中，视觉类别和物理属性来自同一 region_id。
    region_cfg = cfg.get("region_field") or {}
    region_enabled = bool(region_cfg.get("enabled", False))
    if region_enabled:
        mesh_region_ids, mesh_field, _ = build_region_fields(heights.shape, region_cfg)
        gt_region_ids, gt_field, region_mapping = build_region_fields(gt_heights.shape, region_cfg)
        region_profiles, region_profile_metadata = build_region_soil_profiles(region_cfg)
    else:
        mesh_region_ids = gt_region_ids = region_mapping = None
        region_profiles = region_profile_metadata = None
        mesh_field, gt_field = heights, gt_heights

    ## 烘焙混合贴图
    blended_tex = None
    normal_tex = None
    if tex_layers_abs:
        blended_tex = os.path.join(output_dir, "terrain_blended.jpg")
        baked_size = bake_blended_texture(
            tex_layers_abs, mesh_field, blended_tex,
            size=tex_size,
            quilting=texture_quilting,
        )
        write_layer_debug_image(os.path.join(output_dir, "terrain_layers_debug.png"), mesh_field, texture_layers, baked_size)
        normal_tex = os.path.join(output_dir, "terrain_normal.png")
        write_texture_normal_map(normal_tex, blended_tex)

    ## 写地形 USD（按高度分层，每层独立物理材质）
    terrain_usd = os.path.join(output_dir, "terrain_only.usd")
    write_layered_terrain_usd(
        terrain_usd, vertices, triangles, texture_layers,
        root_name="terrain", texture_path=blended_tex, normal_map_path=normal_tex, uvs=uvs,
        vertex_field=mesh_field.ravel(),
    )
    # 无碰撞版本：轮地力学用（地面碰撞由 Bekker-Wong 公式力替代，rocks 碰撞另算）
    write_layered_terrain_usd(
        os.path.join(output_dir, "terrain_only_nocollide.usd"),
        vertices, triangles, texture_layers,
        root_name="terrain", texture_path=blended_tex, normal_map_path=normal_tex, uvs=uvs, add_collision=False,
        vertex_field=mesh_field.ravel(),
    )

    ## GT 独立高分辨率；heights.npy 保持与物理 mesh 一致。
    write_friction_map(os.path.join(output_dir, "friction_map.npy"), gt_field, texture_layers)
    if region_enabled:
        write_region_soil_params_map(
            os.path.join(output_dir, "soil_params_map.npy"), gt_region_ids, region_profiles)
    else:
        write_soil_params_map(os.path.join(output_dir, "soil_params_map.npy"), gt_field, texture_layers)
    write_terrain_class_map(os.path.join(output_dir, "terrain_class_map.npy"), gt_field, texture_layers)
    if region_enabled:
        np.save(os.path.join(output_dir, "region_id_map.npy"), gt_region_ids)
        region_metadata = dict(region_cfg)
        region_metadata["region_to_semantic"] = region_mapping.astype(int).tolist()
        region_metadata["profiles"] = region_profile_metadata
        with open(os.path.join(output_dir, "region_semantic_map.json"), "w") as file:
            json.dump(region_metadata, file, indent=2)
        print(f"[INFO] Saved region-to-semantic mapping: {region_metadata['semantic_count']} semantic classes")
    np.save(os.path.join(output_dir, "height_gt_map.npy"), gt_heights)
    np.save(os.path.join(output_dir, "heights.npy"), heights)
    with open(os.path.join(output_dir, "terrain_size.txt"), "w") as _f:
        _f.write(repr(terrain_size))   # 仿真端读，避免 zhurong_config 全局 size 与实际 terrain 不匹配（如 slip_test_soil 30m）

    terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    return terrain_mesh, heights, gt_heights, uvs, blended_tex, terrain_size


def _build_rocks(cfg, heights, gt_shape, terrain_size, materials_dir, output_dir):
    # 岩石分布 → 加载 → 简化 → rocks_merged.usd
    rock_k = float(cfg["rock_k"])
    rock_seed = int(cfg["rock_seed"])
    rock_sizes = cfg.get("rock_sizes", [0.4, 0.8, 1.6, 3.2, 6.4])
    rock_downsample = float(cfg.get("rock_downsample", 1))
    rocks_base = os.path.join(materials_dir, "rocks", "mars_rocks")
    rock_texture = cfg.get("rock_texture")
    if rock_texture and not os.path.isabs(rock_texture):
        rock_texture = os.path.join(materials_dir, rock_texture)
    if rock_texture and not os.path.isfile(rock_texture):
        raise FileNotFoundError(f"Rock texture not found: {rock_texture}")

    print(f"[INFO] Generating rocks (k={rock_k}, seed={rock_seed})...")
    rock_list = calculate_rock_distribution(heights, terrain_size, k=rock_k, rock_sizes=rock_sizes, seed=rock_seed)
    print(f"  Generated {len(rock_list)} rocks")
    rock_groups = load_rocks_with_materials(
        rock_list, rocks_base, heights, terrain_size, texture_override=rock_texture) if rock_list else []
    if not rock_groups:
        write_rocks_usd(os.path.join(output_dir, "rocks_merged.usd"), [], add_collision=True, root_name="obstacles")
        write_rock_map(os.path.join(output_dir, "rock_map.npy"), gt_shape, terrain_size)
        return None, None

    ## 简化岩石 mesh
    rocks_only = trimesh.util.concatenate([g[0] for g in rock_groups])
    rock_downsample = max(rock_downsample, 1.0)
    rock_target = max(16, round(len(rocks_only.faces) / rock_downsample))
    visual_groups = simplify_rock_groups(rock_groups, rock_target)
    visual_rocks = trimesh.util.concatenate([g[0] for g in visual_groups])
    print(f"[INFO] Rocks mesh: {len(rocks_only.faces)} -> {len(visual_rocks.faces)} triangles")

    ## 写石头 USD（多组，各带贴图）
    write_rocks_usd(os.path.join(output_dir, "rocks_merged.usd"), visual_groups, add_collision=True, root_name="obstacles")
    write_rock_map(os.path.join(output_dir, "rock_map.npy"), gt_shape, terrain_size, visual_rocks)

    return visual_groups, visual_rocks


def _build_merged_usd(terrain_mesh, collision_rocks, output_dir, gt_heights, terrain_size):
    # 合并地形+岩石 → terrain_merged.usd（用于 raycast）+ height_merged_map.npy（供 height_scan 查表）
    all_meshes = [terrain_mesh]
    if collision_rocks is not None:
        all_meshes.append(collision_rocks)
    full_merged = trimesh.util.concatenate(all_meshes)
    write_mesh_usd(
        os.path.join(output_dir, "terrain_merged.usd"),
        np.asarray(full_merged.vertices, dtype=np.float32),
        np.asarray(full_merged.faces, dtype=np.uint32),
        add_collision=False, root_name="hidden_terrain",
    )
    # 地形 GT 已经是高分辨率底图，只需叠加岩石。
    write_height_merged_map(
        os.path.join(output_dir, "height_merged_map.npy"),
        gt_heights, collision_rocks, terrain_size)
    return full_merged


def _export_rviz_assets(terrain_mesh, visual_rocks, full_merged, uvs, blended_tex, output_dir):
    # 导出 STL + DAE 供 RViz 可视化
    def _save(mesh, name):
        path = os.path.join(output_dir, name)
        mesh.export(path)
        print(f"[INFO] Saved {path}")

    _save(terrain_mesh, "terrain.stl")
    if visual_rocks is not None:
        _save(visual_rocks, "rocks.stl")
    _save(full_merged, "terrain_with_rocks.stl")

    ## DAE（带贴图版本）
    if blended_tex and os.path.exists(blended_tex):
        _export_dae_with_texture(
            os.path.join(output_dir, "terrain.dae"),
            np.asarray(terrain_mesh.vertices, dtype=np.float32),
            np.asarray(terrain_mesh.faces, dtype=np.uint32),
            uvs, blended_tex,
        )
    full_merged.export(os.path.join(output_dir, "terrain_with_rocks.dae"))
    print(f"[INFO] Saved {output_dir}/terrain_with_rocks.dae")


def generate_terrain(cfg, output_name):
    # 主流程：heightmap → terrain + rocks USD → RViz 导出
    # 材料库(models/)和产物(terrain/)分离:materials_dir 取素材,output_root 放产物
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # terrain_generation/
    materials_dir = os.path.join(pkg_dir, "models")
    output_root = os.path.join(pkg_dir, "terrain")
    output_dir = os.path.join(output_root, output_name)
    os.makedirs(output_dir, exist_ok=True)
    profile = {"sky": cfg.get("sky", "mars")}
    if cfg.get("navigation"):
        profile["navigation"] = cfg["navigation"]
    with open(os.path.join(output_dir, "terrain_profile.json"), "w") as f:
        json.dump(profile, f)

    ## 1. 地形 mesh + 贴图
    terrain_mesh, heights, gt_heights, uvs, blended_tex, terrain_size = _build_terrain_mesh(cfg, output_dir, materials_dir)

    ## 2. 岩石
    _, visual_rocks = _build_rocks(cfg, heights, gt_heights.shape, terrain_size, materials_dir, output_dir)
    texture_resolution = cfg.get("texture_resolution")
    preview_size = (max(1, round(terrain_size / float(texture_resolution)))
                    if texture_resolution is not None else int(cfg.get("texture_size", 640)))
    write_terrain_rocks_preview(
        os.path.join(output_dir, "terrain_with_rocks_preview.png"), blended_tex, heights,
        os.path.join(output_dir, "rock_map.npy"), preview_size,
    )

    ## 3. 合并 USD + 合并高程图
    full_merged = _build_merged_usd(terrain_mesh, visual_rocks, output_dir, gt_heights, terrain_size)

    ## 4. RViz 导出
    _export_rviz_assets(terrain_mesh, visual_rocks, full_merged, uvs, blended_tex, output_dir)

    print(f"[INFO] Done! Output: {output_dir}/")
