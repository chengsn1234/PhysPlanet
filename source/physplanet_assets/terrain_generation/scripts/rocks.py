"""岩石分布 + 加载 + 简化(移植自 mars_sim RockDistCalc)。"""

import os
import random

import numpy as np
import trimesh

from .terrain import get_ground_z


def rock_distribution_function(k, D):
    """指数分布 F = k * e^(-qD)，q = 1.79 + 0.152/k"""
    q = 1.79 + 0.152 / k
    return k * np.exp(-q * D)


def calculate_rock_distribution(heightmap, terrain_size, k=0.07, rock_sizes=None, seed=1):
    """计算岩石二维分布（x, y, D），D 为直径（米）"""
    if k <= 0:
        return []   # k=0 表示无岩石(平地测试地形 terrain_config/slip_test/terrain_flat)
    random.seed(seed)
    np.random.seed(seed)
    if rock_sizes is None:
        rock_sizes = [0.4, 0.8, 1.6, 3.2, 6.4]

    H, W = heightmap.shape
    area = terrain_size * terrain_size
    rock_list = []

    for D in rock_sizes:
        F = rock_distribution_function(k, D)
        N_per_m2 = F / (np.pi * (D / 2) ** 2)
        N = int(N_per_m2 * area)

        for _ in range(N):
            x = random.uniform(2, terrain_size - 2)
            y = random.uniform(2, terrain_size - 2)
            if np.sqrt((x - terrain_size/2)**2 + (y - terrain_size/2)**2) < 4.0:
                continue
            rock_list.append({"x": x, "y": y, "D": D})

    for i in range(len(rock_list)):
        for j in range(i + 1, len(rock_list)):
            dx = rock_list[i]["x"] - rock_list[j]["x"]
            dy = rock_list[i]["y"] - rock_list[j]["y"]
            min_dist = (rock_list[i]["D"] + rock_list[j]["D"]) * 0.8
            if dx*dx + dy*dy < min_dist * min_dist:
                rock_list[j]["D"] *= 0.5

    return rock_list


def load_rocks_with_materials(rock_list, rocks_base, heights, terrain_size, texture_override=None):
    """加载石头 mesh，按种类分组合并，保留 UV。
    返回：[(trimesh_merged, texture_abs_path_or_None, rock_idx), ...]
    """
    rock_dirs = sorted([d for d in os.listdir(rocks_base)
                       if os.path.isdir(os.path.join(rocks_base, d))])

    # 只保留有"真实贴图"的石头（{idx}.jpg 或 {idx:02d}.jpg，而非 {idx}_color.jpg 语义标签）
    valid_indices = []
    for i in range(1, len(rock_dirs) + 1):
        tex_dir = os.path.join(rocks_base, f"mars_rock_{i}", "Textures")
        if not os.path.isdir(tex_dir):
            continue
        if (os.path.exists(os.path.join(tex_dir, f"{i:02d}.jpg")) or
            os.path.exists(os.path.join(tex_dir, f"{i}.jpg"))):
            valid_indices.append(i)
    if not valid_indices:
        valid_indices = list(range(1, len(rock_dirs) + 1))

    groups = {}
    for rock_info in rock_list:
        rock_idx = random.choice(valid_indices)
        rock_dir = os.path.join(rocks_base, f"mars_rock_{rock_idx}")
        obj_file = os.path.join(rock_dir, f"rock{rock_idx}.obj")
        if not os.path.exists(obj_file):
            continue
        try:
            mesh = trimesh.load(obj_file, process=False)
        except Exception:
            continue

        target_d = rock_info["D"]
        current_d = mesh.extents.max()
        if current_d > 0:
            mesh.apply_scale(target_d / current_d)

        gz = get_ground_z(rock_info["x"], rock_info["y"], heights, terrain_size)
        mesh.apply_translation([rock_info["x"], rock_info["y"], gz])
        groups.setdefault(rock_idx, []).append(mesh)

    result = []
    for rock_idx, meshes in groups.items():
        if not meshes:
            continue
        merged = trimesh.util.concatenate(meshes)
        tex_candidates = [
            os.path.join(rocks_base, f"mars_rock_{rock_idx}", "Textures", f"{rock_idx:02d}.jpg"),
            os.path.join(rocks_base, f"mars_rock_{rock_idx}", "Textures", f"{rock_idx}.jpg"),
        ]
        tex_path = texture_override or next((p for p in tex_candidates if os.path.exists(p)), None)
        result.append((merged, tex_path, rock_idx))
    return result


def simplify_mesh_open3d(mesh, target_triangles):
    """Simplify a trimesh mesh with Open3D if available."""
    if mesh is None or len(mesh.faces) == 0:
        return mesh
    target_triangles = int(target_triangles)
    if target_triangles <= 0 or len(mesh.faces) <= target_triangles:
        return mesh

    try:
        import open3d as o3d
    except ImportError:
        print("[WARN] Open3D not installed, keeping original mesh")
        return mesh

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_degenerate_triangles()
    simplified = o3d_mesh.simplify_quadric_decimation(target_triangles)
    simplified.remove_duplicated_vertices()
    simplified.remove_duplicated_triangles()
    simplified.remove_degenerate_triangles()

    return trimesh.Trimesh(
        vertices=np.asarray(simplified.vertices, dtype=np.float32),
        faces=np.asarray(simplified.triangles, dtype=np.uint32),
        process=False,
    )


def simplify_rock_groups(rock_groups, target_total_triangles):
    """Simplify rock groups while keeping texture grouping."""
    if not rock_groups:
        return []

    total_faces = sum(len(mesh.faces) for mesh, _, _ in rock_groups)
    target_total_triangles = int(target_total_triangles)
    if target_total_triangles <= 0 or total_faces <= target_total_triangles:
        return rock_groups

    simplified_groups = []
    used_target = 0
    for idx, (mesh, tex_path, rock_idx) in enumerate(rock_groups):
        if idx == len(rock_groups) - 1:
            target = max(16, target_total_triangles - used_target)
        else:
            target = max(16, round(target_total_triangles * len(mesh.faces) / total_faces))
            used_target += target
        simplified_groups.append((simplify_mesh_open3d(mesh, target), tex_path, rock_idx))
    return simplified_groups
