# GT 查表模块
####################
## 功能：加载 GT 地图、生成世界坐标网格、批量查表发布 patch
## 依赖：terramechanics.bilinear_sample（纯 tensor 双线性插值）
## 兼容：Hunter / Zhurong（车型差异由调用方传 HEIGHT_OFFSET）
####################

import os

import numpy as np
import torch

from terramechanics import bilinear_sample

SOIL_KEYS = ("Kc", "Kphi", "n0", "n1", "c", "phi", "K")

# terrain 目录里 GT 地图文件名 → dict key
_GT_FILES = {
    "height_gt": "height_gt_map.npy",   # 精高程 GT
    "friction": "friction_map.npy",
    "soil": "soil_params_map.npy",       # (H,W,7)
    "terrain_class": "terrain_class_map.npy",
    "rock": "rock_map.npy",
}


####################
## 地图加载
####################

def load_gt_maps(terrain_dir, device):
    """加载所有 GT 地图到 GPU tensor。存在的加载，不存在的跳过（None）。"""
    maps = {}
    for key, filename in _GT_FILES.items():
        path = os.path.join(terrain_dir, filename)
        if os.path.exists(path):
            arr = np.load(path)
            maps[key] = torch.as_tensor(arr, dtype=torch.float32, device=device)
            print(f"[GT] Loaded {filename}: shape={maps[key].shape}")
        else:
            maps[key] = None
    return maps


####################
## 世界坐标网格
####################

def make_world_grid(robot_pos_w, root_quat_w, scan_size, scan_res, device):
    """生成 yaw 对齐的世界坐标网格点。

    robot_pos_w: (E, 3), root_quat_w: (E, 4) wxyz
    scan_size: [sx, sy] 米, scan_res: 米/cell
    返回: world_xy (E, grid_h*grid_w, 2)
    """
    sx, sy = float(scan_size[0]), float(scan_size[1])
    res = float(scan_res)

    # 本地网格 (grid_h*grid_w, 2)，中心 [0,0]
    xs = torch.arange(-sx / 2, sx / 2 + res / 2, res, device=device)
    ys = torch.arange(-sy / 2, sy / 2 + res / 2, res, device=device)
    grid_h, grid_w = len(ys), len(xs)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    local_xy = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (N, 2)

    # yaw 从四元数提取（同 root_axes_w 的 quat_apply）
    forward_b = torch.zeros(root_quat_w.shape[0], 3, device=device)
    forward_b[:, 0] = 1.0
    q_vec = root_quat_w[..., 1:4]
    q_w = root_quat_w[..., 0:1]
    t = 2.0 * torch.cross(q_vec, forward_b, dim=-1)
    forward_w = forward_b + q_w * t + torch.cross(q_vec, t, dim=-1)
    yaw = torch.atan2(forward_w[:, 1], forward_w[:, 0])  # (E,)

    # 旋转本地网格: cos/sin (E,1,1) × local (1,N,2) → (E,N,2)
    cos_y = torch.cos(yaw).unsqueeze(1).unsqueeze(2)  # (E,1,1)
    sin_y = torch.sin(yaw).unsqueeze(1).unsqueeze(2)
    local = local_xy.unsqueeze(0)  # (1,N,2)
    world_x = cos_y * local[..., 0:1] - sin_y * local[..., 1:2]  # (E,N,1)
    world_y = sin_y * local[..., 0:1] + cos_y * local[..., 1:2]
    world_offset = torch.cat([world_x, world_y], dim=-1)  # (E,N,2)

    # 加 robot_xy
    world_xy = world_offset + robot_pos_w[:, None, :2]  # (E, N, 2)
    return world_xy


####################
## 批量查表
####################

def compute_patches(world_xy, maps_dict, terrain_size):
    """一次网格点查所有存在的 map。

    world_xy: (E, N, 2)
    maps_dict: load_gt_maps 返回的 dict（或含 "height" 的自定义 dict）
    返回 dict：只含 maps_dict 中非 None 的 key。
      标量 map → (E, N)；soil → (E, N, 7)。
    """
    E, N, _ = world_xy.shape
    flat_xy = world_xy.reshape(E * N, 2)
    patches = {}
    for key, gt_map in maps_dict.items():
        if gt_map is None:
            continue
        result = bilinear_sample(gt_map, flat_xy, terrain_size)
        if result.dim() == 1:
            patches[key] = result.reshape(E, N)
        else:
            channels = result.shape[-1]
            patches[key] = result.reshape(E, N, channels)
    return patches
