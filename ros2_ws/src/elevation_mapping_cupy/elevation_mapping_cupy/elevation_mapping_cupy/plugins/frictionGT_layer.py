"""Terrain raster GT plugin for friction, soil, class and rock layers."""

import os

import cv2
import cupy as cp
import numpy as np

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


_MAP_CACHE = {}


class FrictionGTLayer(PluginBase):
    def __init__(self, cell_n=100, map_file="", channel=None, terrain_size=100.0,
                 map_length=20.0, env_origin_xy=(0.0, 0.0), interpolation="linear", **_):
        self.cell_n = cell_n
        self.terrain_size = float(terrain_size)
        self.map_length = float(map_length)
        self.env_origin_xy = tuple(float(v) for v in env_origin_xy)
        self.interpolation = cv2.INTER_NEAREST if interpolation == "nearest" else cv2.INTER_LINEAR
        self.default_layer = cp.full((cell_n, cell_n), cp.nan, dtype=cp.float32)
        self.global_map = self._load(map_file, channel)

    @staticmethod
    def _load(path, channel):
        if not path:
            return None
        path = os.path.abspath(path)
        key = (path, channel)
        if key not in _MAP_CACHE:
            if not os.path.exists(path):
                print(f"[TerrainGT] WARN: map not found: {path}")
                _MAP_CACHE[key] = None
            else:
                array = np.load(path)
                if channel is not None:
                    array = array[..., int(channel)]
                if array.ndim != 2:
                    raise ValueError(f"Terrain GT must be 2D after channel selection: {path}, got {array.shape}")
                _MAP_CACHE[key] = array.astype(np.float32)
                print(f"[TerrainGT] Loaded {path}, channel={channel}, shape={array.shape}")
        return _MAP_CACHE[key]

    def __call__(self, elevation_map, layer_names, plugin_layers, plugin_layer_names,
                 semantic_map, semantic_layer_names, rotation, elements_to_shift, *args):
        if self.global_map is None:
            return self.default_layer
        center = elements_to_shift.get("center")
        if center is None:
            return self.default_layer
        cx = float(cp.asnumpy(center[0])) - self.env_origin_xy[0]
        cy = float(cp.asnumpy(center[1])) - self.env_origin_xy[1]
        h, w = self.global_map.shape
        mpp_x, mpp_y = self.terrain_size / max(w - 1, 1), self.terrain_size / max(h - 1, 1)
        half_w = max(1, round(self.map_length * 0.5 / mpp_x))
        half_h = max(1, round(self.map_length * 0.5 / mpp_y))
        col, row = round(cx / mpp_x), round(cy / mpp_y)
        if row - half_h < 0 or row + half_h >= h or col - half_w < 0 or col + half_w >= w:
            return self.default_layer
        crop = self.global_map[row - half_h:row + half_h + 1, col - half_w:col + half_w + 1]
        return cp.asarray(cv2.resize(crop, (self.cell_n, self.cell_n), interpolation=self.interpolation), dtype=cp.float32)
