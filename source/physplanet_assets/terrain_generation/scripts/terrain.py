"""地形几何 + 派生图:heightmap→mesh + 高程采样 + 高度→层映射 + 贴图烘焙 + friction/soil/debug 图。"""

import os

import cv2
import numpy as np

from .config import SOIL_KEYS, _SOIL_DEFAULT


# ── Perlin heightmap ──


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def _perlin2d(shape, periods, rng):
    h, w = shape
    py, px = periods
    angles = rng.random((py + 1, px + 1)) * 2.0 * np.pi
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    y = np.linspace(0, py, h, endpoint=False)
    x = np.linspace(0, px, w, endpoint=False)
    yi, xi = y.astype(int), x.astype(int)
    yf, xf = y - yi, x - xi

    g00 = gradients[yi[:, None], xi[None, :]]
    g10 = gradients[yi[:, None], xi[None, :] + 1]
    g01 = gradients[yi[:, None] + 1, xi[None, :]]
    g11 = gradients[yi[:, None] + 1, xi[None, :] + 1]
    dx, dy = xf[None, :], yf[:, None]
    n00 = g00[..., 0] * dx + g00[..., 1] * dy
    n10 = g10[..., 0] * (dx - 1) + g10[..., 1] * dy
    n01 = g01[..., 0] * dx + g01[..., 1] * (dy - 1)
    n11 = g11[..., 0] * (dx - 1) + g11[..., 1] * (dy - 1)

    u, v = _fade(dx), _fade(dy)
    nx0 = n00 * (1 - u) + n10 * u
    nx1 = n01 * (1 - u) + n11 * u
    return nx0 * (1 - v) + nx1 * v


def perlin_heightmap(resolution=512, seed=42, scale=6.0, octaves=4, persistence=0.5, lacunarity=2.0):
    """Generate a normalized uint16 Perlin heightmap."""
    resolution, seed, octaves = int(resolution), int(seed), int(octaves)
    scale, persistence, lacunarity = float(scale), float(persistence), float(lacunarity)
    if not 1 <= resolution <= 4096:
        raise ValueError("perlin_resolution must be in [1, 4096]")
    if not 1 <= octaves <= 12:
        raise ValueError("perlin_octaves must be in [1, 12]")
    if scale <= 0 or persistence <= 0 or lacunarity <= 0:
        raise ValueError("Perlin scale, persistence and lacunarity must be positive")

    rng = np.random.default_rng(seed)
    noise = np.zeros((resolution, resolution), dtype=np.float32)
    amplitude = frequency = 1.0
    total_amplitude = 0.0
    for _ in range(octaves):
        periods = min(resolution, max(1, round(scale * frequency)))
        noise += amplitude * _perlin2d((resolution, resolution), (periods, periods), rng)
        total_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    noise /= max(total_amplitude, 1.0e-6)
    noise -= float(noise.min())
    noise /= max(float(noise.max()), 1.0e-6)
    return np.rint(noise * 65535).astype(np.uint16)


def write_perlin_heightmap(path, **params):
    image = perlin_heightmap(**params)
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"Failed to write Perlin heightmap: {path}")
    return path


# ── heightmap → mesh + 高程采样 ──


def heightmap_to_mesh(heightmap_path, terrain_size, height_scale, downsample=1):
    """读取 16-bit PNG heightmap，生成共享顶点的三角形 mesh（含 UV 坐标）

    downsample: 下采样步长，>1 降低 mesh 密度（例如 2 = 顶点数降 4 倍）
    """
    img = cv2.imread(heightmap_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read heightmap: {heightmap_path}")

    if downsample > 1:
        img = img[::downsample, ::downsample]

    H, W = img.shape
    heights = img.astype(np.float32) * height_scale
    horizontal_scale = terrain_size / max(H - 1, W - 1)

    # 顶点网格（向量化）
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    xs = jj * horizontal_scale
    ys = ii * horizontal_scale
    vertices = np.stack([xs.ravel(), ys.ravel(), heights.ravel()], axis=1).astype(np.float32)

    # UV 坐标
    us = jj / max(W - 1, 1)
    vs = ii / max(H - 1, 1)
    uvs = np.stack([us.ravel(), vs.ravel()], axis=1).astype(np.float32)

    # 三角形索引（每个 quad 两个三角形，顶点按 row-major 编号 i*W+j）
    idx_grid = (ii * W + jj).astype(np.uint32)
    i00 = idx_grid[:-1, :-1].ravel()
    i10 = idx_grid[:-1, 1:].ravel()
    i11 = idx_grid[1:, 1:].ravel()
    i01 = idx_grid[1:, :-1].ravel()
    tri1 = np.stack([i00, i10, i11], axis=1)
    tri2 = np.stack([i00, i11, i01], axis=1)
    triangles = np.concatenate([tri1, tri2], axis=0).astype(np.uint32)

    return vertices, triangles, heights, uvs


def heightmap_to_gt(heightmap_path, terrain_size, height_scale, gt_resolution):
    """从原始 heightmap 导出独立高分辨率查询高程。"""
    if gt_resolution <= 0.0:
        raise ValueError(f"gt_resolution must be positive, got {gt_resolution}")
    img = cv2.imread(heightmap_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read heightmap: {heightmap_path}")
    size = max(2, round(terrain_size / gt_resolution) + 1)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) * height_scale


def build_region_id_map(shape, rows=4, cols=5, seed=0):
    """Build a spatially coherent block map with a seeded region permutation."""
    height, width = map(int, shape[:2])
    rows, cols = int(rows), int(cols)
    if rows <= 0 or cols <= 0 or height < rows or width < cols:
        raise ValueError(f"Invalid region grid {rows}x{cols} for map shape {shape}")
    row_ids = np.minimum(np.arange(height) * rows // height, rows - 1)
    col_ids = np.minimum(np.arange(width) * cols // width, cols - 1)
    base_ids = row_ids[:, None] * cols + col_ids[None, :]
    permutation = np.random.default_rng(int(seed)).permutation(rows * cols)
    return permutation[base_ids].astype(np.int16)


def build_region_fields(shape, region_cfg):
    """Return region IDs and four-class layer IDs from one aligned mapping."""
    rows = int(region_cfg.get("rows", 4))
    cols = int(region_cfg.get("cols", 5))
    region_count = rows * cols
    mapping = region_cfg.get("region_to_semantic")
    if mapping is None:
        semantic_count = int(region_cfg.get("semantic_count", 1))
        if region_count % semantic_count != 0:
            raise ValueError("region_count must be divisible by semantic_count")
        per_class = region_count // semantic_count
        mapping = [i // per_class for i in range(region_count)]
    mapping = np.asarray(mapping, dtype=np.int16)
    semantic_count = int(region_cfg.get("semantic_count", int(mapping.max()) + 1))
    if mapping.size != region_count:
        raise ValueError(f"region_to_semantic must contain {region_count} entries, got {mapping.size}")
    if int(mapping.min()) < 0 or int(mapping.max()) >= semantic_count:
        raise ValueError("region_to_semantic contains an invalid semantic ID")
    region_ids = build_region_id_map(
        shape, rows=rows, cols=cols, seed=int(region_cfg.get("seed", 0)))
    return region_ids, mapping[region_ids], mapping


def _stratified_uniform(low, high, count, rng):
    """Draw one value from every equal-width subinterval and shuffle it."""
    edges = np.linspace(float(low), float(high), int(count) + 1, dtype=np.float64)
    values = rng.uniform(edges[:-1], edges[1:])
    rng.shuffle(values)
    return values.astype(np.float32)


def build_region_soil_profiles(region_cfg):
    """Create one Bekker--Wong profile per region from class-specific ranges."""
    rows = int(region_cfg.get("rows", 4))
    cols = int(region_cfg.get("cols", 5))
    region_count = rows * cols
    semantic_count = int(region_cfg["semantic_count"])
    mapping = np.asarray(region_cfg["region_to_semantic"], dtype=np.int16)
    if mapping.size != region_count:
        raise ValueError("region_to_semantic is required for region-level profiles")
    rules = region_cfg.get("profile_rules") or []
    if len(rules) != semantic_count:
        raise ValueError(f"profile_rules must contain {semantic_count} entries")
    fixed = {key: float(value) for key, value in (region_cfg.get("fixed_soil") or {}).items()}
    required_fixed = [key for key in SOIL_KEYS if key not in {"n0", "phi"}]
    missing = [key for key in required_fixed if key not in fixed]
    if missing:
        raise ValueError(f"fixed_soil is missing {missing}")

    rng = np.random.default_rng(int(region_cfg.get("profile_seed", 0)))
    profiles = [None] * region_count
    metadata = []
    for semantic_id, rule in enumerate(rules):
        region_ids = np.flatnonzero(mapping == semantic_id)
        if region_ids.size == 0:
            raise ValueError(f"semantic class {semantic_id} has no regions")
        if "n0_range" in rule:
            n0_values = _stratified_uniform(*rule["n0_range"], region_ids.size, rng)
        else:
            n0_values = np.full(region_ids.size, float(rule["n0"]), dtype=np.float32)
        if "phi_deg_range" in rule:
            phi_values = _stratified_uniform(*rule["phi_deg_range"], region_ids.size, rng)
        else:
            phi_values = np.full(region_ids.size, float(rule["phi_deg"]), dtype=np.float32)

        for region_id, n0, phi_deg in zip(region_ids, n0_values, phi_values):
            profile = dict(fixed)
            profile["n0"] = float(n0)
            profile["phi"] = float(np.deg2rad(phi_deg))
            profiles[int(region_id)] = profile
            metadata.append({
                "region_id": int(region_id),
                "semantic_id": semantic_id,
                "semantic_name": str(rule.get("name", semantic_id)),
                "n0": float(n0),
                "phi_deg": float(phi_deg),
                "soil": {key: float(profile[key]) for key in SOIL_KEYS},
            })
    if any(profile is None for profile in profiles):
        raise ValueError("At least one region has no soil profile")
    metadata.sort(key=lambda item: item["region_id"])
    return profiles, metadata


def get_ground_z(x, y, heights, terrain_size):
    """从 heightmap 采样地面高度"""
    H, W = heights.shape
    ji = int(x / terrain_size * (W - 1))
    ii = int(y / terrain_size * (H - 1))
    ji = min(max(ji, 0), W - 1)
    ii = min(max(ii, 0), H - 1)
    return heights[ii, ji]


# ── 高度→层映射(层逻辑枢纽,usd_export 也用)──


def _texture_layer_upper_bounds(layer_cfgs):
    """Compute layer bounds consistent with the texture blend ordering."""
    explicit = [layer.get("max_height_ratio") for layer in layer_cfgs]
    if all(value is not None for value in explicit):
        return [float(value) for value in explicit]

    num_layers = len(layer_cfgs)
    if num_layers <= 1:
        return [1.0]

    centers = np.linspace(0.0, 1.0, num_layers)
    bounds = []
    for i in range(num_layers):
        if i == num_layers - 1:
            bounds.append(1.0)
        else:
            bounds.append(float(0.5 * (centers[i] + centers[i + 1])))
    return bounds


def _height_to_layer_indices(heights, layer_cfgs):
    """Map a height array to texture/physics layer indices using shared bounds."""
    if not layer_cfgs:
        return np.zeros_like(heights, dtype=np.uint8)

    h_min = float(np.min(heights)) if heights.size else 0.0
    h_max = float(np.max(heights)) if heights.size else 1.0
    denom = max(h_max - h_min, 1.0e-6)
    h_norm = (heights - h_min) / denom
    upper_bounds = _texture_layer_upper_bounds(layer_cfgs)

    layer_ids = np.zeros_like(h_norm, dtype=np.uint8)
    lower = -1.0e-6
    for i, upper in enumerate(upper_bounds):
        if i == len(upper_bounds) - 1:
            mask = h_norm >= lower
        else:
            mask = (h_norm >= lower) & (h_norm <= upper)
        layer_ids[mask] = i
        lower = upper
    return layer_ids


def _split_triangles_by_height_layers(vertices, triangles, layer_cfgs, vertex_field=None):
    """Split triangles by height or by a supplied per-vertex layer field."""
    if not layer_cfgs:
        return [("terrain", triangles)]

    if vertex_field is None:
        tri_heights = vertices[triangles][:, :, 2].mean(axis=1)
        vertex_min = float(vertices[:, 2].min()) if len(vertices) > 0 else 0.0
        vertex_max = float(vertices[:, 2].max()) if len(vertices) > 0 else 1.0
        # Classify each triangle by its mean height because a triangle can
        # only carry one material in the exported USD.
        tri_domain = np.concatenate([
            np.asarray([vertex_min, vertex_max], dtype=np.float32),
            tri_heights.astype(np.float32),
        ])
        tri_layer_ids = _height_to_layer_indices(tri_domain, layer_cfgs)[2:]
    else:
        values = np.asarray(vertex_field, dtype=np.float32)
        if values.shape[0] != vertices.shape[0]:
            raise ValueError("vertex_field must have one value per mesh vertex")
        tri_layer_ids = np.rint(values[triangles].mean(axis=1)).astype(np.int32)
        tri_layer_ids = np.clip(tri_layer_ids, 0, len(layer_cfgs) - 1)

    layers = []
    remaining = np.ones(len(triangles), dtype=bool)
    for i, layer in enumerate(layer_cfgs):
        mask = (tri_layer_ids == i) & remaining
        if np.any(mask):
            name = str(layer.get("name", f"layer_{i}"))
            layers.append((name, triangles[mask], layer))
            remaining[mask] = False

    if np.any(remaining):
        last_layer = layer_cfgs[-1]
        name = str(last_layer.get("name", f"layer_{len(layer_cfgs)-1}"))
        layers.append((f"{name}_overflow", triangles[remaining], last_layer))

    return layers


# ── 多层贴图烘焙 ──


def _smoothstep(edge0, edge1, x):
    """GLSL-style smoothstep"""
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _blend_texture_layers(layers, height_norm):
    """Blend layers on CUDA when available, with a NumPy fallback."""
    count, size = len(layers), height_norm.shape[0]
    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except ImportError:
        use_cuda = False

    if use_cuda:
        device = torch.device("cuda")
        with torch.inference_mode():
            height_t = torch.from_numpy(height_norm).to(device)
            if count == 1:
                blended = torch.from_numpy(layers[0]).to(device)
            else:
                centers = torch.linspace(0.0, 1.0, count, device=device)
                band = 1e-3 / (count - 1)

                def smoothstep(lo, hi):
                    value = ((height_t - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)
                    return value * value * (3.0 - 2.0 * value)

                blended = torch.zeros((size, size, 3), dtype=torch.float32, device=device)
                for i in range(count):
                    if i == 0:
                        hi = (centers[i] + centers[i + 1]) * 0.5
                        weight = 1.0 - smoothstep(hi - band, hi + band)
                    elif i == count - 1:
                        lo = (centers[i - 1] + centers[i]) * 0.5
                        weight = smoothstep(lo - band, lo + band)
                    else:
                        lo = (centers[i - 1] + centers[i]) * 0.5
                        hi = (centers[i] + centers[i + 1]) * 0.5
                        weight = smoothstep(lo - band, lo + band) * (1.0 - smoothstep(hi - band, hi + band))
                    layer_t = torch.from_numpy(layers[i]).to(device)
                    blended.addcmul_(layer_t, weight[..., None])
                    del layer_t, weight
            return blended.clamp(0.0, 255.0).to(torch.uint8).cpu().numpy(), "CUDA"

    if count == 1:
        return layers[0].astype(np.uint8), "CPU"

    centers = np.linspace(0, 1, count)
    band = 1e-3 / (count - 1)
    blended = np.zeros((size, size, 3), dtype=np.float32)
    for i in range(count):
        if i == 0:
            hi = (centers[i] + centers[i + 1]) * 0.5
            weight = 1.0 - _smoothstep(hi - band, hi + band, height_norm)
        elif i == count - 1:
            lo = (centers[i - 1] + centers[i]) * 0.5
            weight = _smoothstep(lo - band, lo + band, height_norm)
        else:
            lo = (centers[i - 1] + centers[i]) * 0.5
            hi = (centers[i] + centers[i + 1]) * 0.5
            weight = _smoothstep(lo - band, lo + band, height_norm) * (1.0 - _smoothstep(hi - band, hi + band, height_norm))
        blended += weight[..., None] * layers[i]
    return np.clip(blended, 0, 255).astype(np.uint8), "CPU"


def _flatten_illumination(image):
    """Remove broad lighting gradients while preserving local texture contrast."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = lab[..., 0]
    sigma = max(image.shape[:2]) / 8.0
    base = cv2.GaussianBlur(lightness, (0, 0), sigma)
    lab[..., 0] = np.clip(lightness * np.median(base) / np.maximum(base, 1.0), 0.0, 255.0)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _min_cut_vertical(error):
    """Return the lowest-cost top-to-bottom seam through an overlap image."""
    cost = error.astype(np.float32).copy()
    parent = np.zeros(cost.shape, dtype=np.int32)
    for row in range(1, cost.shape[0]):
        prev = cost[row - 1]
        padded = np.pad(prev, (1, 1), mode="constant", constant_values=np.inf)
        choices = np.stack([padded[:-2], padded[1:-1], padded[2:]], axis=0)
        offset = np.argmin(choices, axis=0) - 1
        parent[row] = np.arange(cost.shape[1]) + offset
        cost[row] += choices.min(axis=0)

    seam = np.empty(cost.shape[0], dtype=np.int32)
    seam[-1] = int(np.argmin(cost[-1]))
    for row in range(cost.shape[0] - 1, 0, -1):
        seam[row - 1] = parent[row, seam[row]]
    return seam


def _min_cut_horizontal(error):
    """Return the lowest-cost left-to-right seam through an overlap image."""
    return _min_cut_vertical(error.T)


def _random_patch(image, patch_size, rng):
    height, width = image.shape[:2]
    y = int(rng.integers(0, height - patch_size + 1))
    x = int(rng.integers(0, width - patch_size + 1))
    return image[y:y + patch_size, x:x + patch_size].copy()


def synthesize_texture_canvas(image, size, rng):
    """Expand one texture with overlap quilting and minimum-error seams."""
    image = _flatten_illumination(image)
    patch_size = min(256, *image.shape[:2])
    if patch_size < 4:
        raise ValueError(f"Texture is too small for quilting: {image.shape[:2]}")
    overlap = max(1, patch_size // 4)
    candidates = 4
    step = patch_size - overlap
    canvas = np.zeros((size + patch_size, size + patch_size, 3), dtype=np.uint8)

    for y in range(0, size, step):
        for x in range(0, size, step):
            best_patch = None
            best_error = np.inf
            for _ in range(max(int(candidates), 1)):
                patch = _random_patch(image, patch_size, rng)
                error = 0.0
                if y:
                    error += np.mean((canvas[y:y + overlap, x:x + patch_size].astype(np.float32)
                                      - patch[:overlap].astype(np.float32)) ** 2)
                if x:
                    error += np.mean((canvas[y:y + patch_size, x:x + overlap].astype(np.float32)
                                      - patch[:, :overlap].astype(np.float32)) ** 2)
                if error < best_error:
                    best_patch, best_error = patch, error

            if not x and not y:
                canvas[y:y + patch_size, x:x + patch_size] = best_patch
                continue

            mask = np.ones((patch_size, patch_size), dtype=bool)
            if y:
                error = np.mean((canvas[y:y + overlap, x:x + patch_size].astype(np.float32)
                                 - best_patch[:overlap].astype(np.float32)) ** 2, axis=2)
                seam = _min_cut_horizontal(error)
                top_mask = np.zeros((overlap, patch_size), dtype=bool)
                for col, row in enumerate(seam):
                    top_mask[row + 1:, col] = True
                mask[:overlap] &= top_mask
            if x:
                error = np.mean((canvas[y:y + patch_size, x:x + overlap].astype(np.float32)
                                 - best_patch[:, :overlap].astype(np.float32)) ** 2, axis=2)
                seam = _min_cut_vertical(error)
                left_mask = np.zeros((patch_size, overlap), dtype=bool)
                for row, col in enumerate(seam):
                    left_mask[row, col + 1:] = True
                mask[:, :overlap] &= left_mask

            region = canvas[y:y + patch_size, x:x + patch_size]
            region[mask] = best_patch[mask]

    return canvas[:size, :size]


def bake_blended_texture(layer_paths, heights, output_path, size=None, quilting=False):
    """按 heightmap 高度把 N 张贴图分层混合烘焙成一张 JPG。
    layer_paths: 从低到高的贴图路径列表（N 张）
    quilting: true 时用 patch quilting 合成每层细节画布；否则自动采用源图分辨率
    """
    N = len(layer_paths)
    if N < 1:
        raise ValueError("Need at least 1 texture layer")

    images = []
    rng = np.random.default_rng(42)
    for p in layer_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Texture layer not found: {p}")
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read texture layer: {p}")
        images.append(img)

    if size is None:
        # A blended texture must be square.  The smallest source side avoids
        # upscaling any layer while keeping the original texture detail.
        size = min(min(image.shape[:2]) for image in images)
    size = int(size)

    layers = []
    for img in images:
        if quilting:
            img = synthesize_texture_canvas(img, size, rng)
        else:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        layers.append(img.astype(np.float32))

    # 归一化高度到 [0,1]（按实际数据范围）
    h_resized = cv2.resize(heights.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    h_min, h_max = h_resized.min(), h_resized.max()
    if h_max - h_min < 1e-6:
        t = np.zeros_like(h_resized)
    else:
        t = (h_resized - h_min) / (h_max - h_min)

    blended, device = _blend_texture_layers(layers, t)

    # USD texture coordinates use V increasing upward in the terrain UVs below,
    # while OpenCV image rows increase downward. Save a vertically flipped image
    # so row i in the heightmap lands on terrain row i after texture sampling.
    blended = cv2.flip(blended, 0)
    cv2.imwrite(output_path, blended)
    detail_note = ", quilting" if quilting else ""
    print(f"[INFO] Baked blended texture: {output_path} ({N} layers, {size}x{size}{detail_note}, {device})")
    return size


def write_texture_normal_map(output_path, texture_path):
    """Derive a subtle tangent-space normal map from the blended texture."""
    image = cv2.imread(texture_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read texture: {texture_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    normal = np.dstack([-dx * 4.0, -dy * 4.0, np.ones_like(gray)])
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    normal = ((normal + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    cv2.imwrite(output_path, normal[..., ::-1])
    print(f"[INFO] Saved texture normal map: {output_path}")


# ── GT 地图导出（独立高分辨率查询栅格）──


def write_layer_debug_image(output_path, heights, layer_cfgs, size=None):
    """Write a false-color image showing the exact height-to-layer map."""
    if not layer_cfgs:
        return
    layer_ids = _height_to_layer_indices(heights.astype(np.float32), layer_cfgs)
    if size is not None:
        layer_ids = cv2.resize(layer_ids, (size, size), interpolation=cv2.INTER_NEAREST)
    palette = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
    ], dtype=np.uint8)
    img = cv2.flip(palette[layer_ids % len(palette)], 0)
    cv2.imwrite(output_path, img)
    print(f"[INFO] Saved layer debug image: {output_path}")


def write_terrain_rocks_preview(output_path, terrain_texture_path, heights, rock_map_path, size=640):
    """写入地形贴图叠加岩石占用轮廓的俯视预览。"""
    image = cv2.imread(terrain_texture_path) if terrain_texture_path else None
    if image is None:
        height_img = cv2.resize(heights, (size, size), interpolation=cv2.INTER_LINEAR)
        height_img = cv2.normalize(height_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image = cv2.applyColorMap(cv2.flip(height_img, 0), cv2.COLORMAP_BONE)
    else:
        size = image.shape[1]
    rock_map = np.load(rock_map_path)
    rock_map = cv2.resize(rock_map, (size, size), interpolation=cv2.INTER_NEAREST)
    mask = cv2.flip(rock_map, 0).astype(bool)
    overlay = image.copy()
    overlay[mask] = (45, 85, 160)
    image[mask] = cv2.addWeighted(image, 0.45, overlay, 0.55, 0)[mask]
    cv2.imwrite(output_path, image)
    print(f"[INFO] Saved terrain rocks preview: {output_path}")


def write_friction_map(output_path, heights, layer_cfgs):
    """导出 GT 摩擦地图，每像素 = static_friction。"""
    if not layer_cfgs:
        return
    layer_ids = _height_to_layer_indices(heights.astype(np.float32), layer_cfgs)
    friction_map = np.zeros_like(heights, dtype=np.float32)
    for i, layer in enumerate(layer_cfgs):
        friction_map[layer_ids == i] = float(layer.get("static_friction", 0.5))
    np.save(output_path, friction_map)
    print(f"[INFO] Saved friction map: {output_path} (shape={heights.shape}, "
          f"range=[{friction_map.min():.2f}, {friction_map.max():.2f}])")


def write_terrain_class_map(output_path, heights, layer_cfgs):
    """写入层类别 GT；0 留给地图外，地形层从 1 开始。"""
    layer_ids = _height_to_layer_indices(heights.astype(np.float32), layer_cfgs)
    np.save(output_path, layer_ids.astype(np.uint8) + 1)
    print(f"[INFO] Saved terrain class map: {output_path} (shape={layer_ids.shape})")


def write_rock_map(output_path, heights_shape, terrain_size, rock_mesh=None):
    """把最终岩石 mesh 的俯视投影光栅化为二值占用 GT。"""
    height, width = heights_shape
    rock_map = np.zeros((height, width), dtype=np.uint8)
    if rock_mesh is not None and len(rock_mesh.faces) > 0:
        sx = (width - 1) / max(terrain_size, 1.0e-6)
        sy = (height - 1) / max(terrain_size, 1.0e-6)
        vertices = np.asarray(rock_mesh.vertices)
        for face in np.asarray(rock_mesh.faces):
            points = vertices[face, :2] * (sx, sy)
            cv2.fillConvexPoly(rock_map, np.rint(points).astype(np.int32), 1)
    np.save(output_path, rock_map)
    print(f"[INFO] Saved rock map: {output_path}")


def write_height_merged_map(output_path, ground_heights, rock_mesh, terrain_size):
    """将岩石最高高度叠加到高分辨率地形 GT，供 height_scan 查表。"""
    merged = ground_heights.astype(np.float32, copy=True)
    height, width = merged.shape
    if rock_mesh is None or len(rock_mesh.faces) == 0:
        np.save(output_path, merged)
        print(f"[INFO] Saved height merged map: {output_path} (terrain only)")
        return

    sx = (width - 1) / max(terrain_size, 1.0e-6)
    sy = (height - 1) / max(terrain_size, 1.0e-6)
    verts = np.asarray(rock_mesh.vertices)
    faces = np.asarray(rock_mesh.faces)
    face_max_z = verts[faces][:, :, 2].max(axis=1)          # 每面最高 Z
    for face, z in zip(faces, face_max_z):
        pts = np.rint(verts[face, :2] * (sx, sy)).astype(np.int32)
        x0, y0 = np.maximum(pts.min(axis=0), 0)
        x1 = min(int(pts[:, 0].max()), width - 1)
        y1 = min(int(pts[:, 1].max()), height - 1)
        if x0 > x1 or y0 > y1:
            continue
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts - (x0, y0), 1)
        region = merged[y0:y1 + 1, x0:x1 + 1]
        region[mask.astype(bool)] = np.maximum(region[mask.astype(bool)], z)
    np.save(output_path, merged)
    print(f"[INFO] Saved height merged map: {output_path} (shape={merged.shape}, "
          f"range=[{merged.min():.3f}, {merged.max():.3f}])")


def write_soil_params_map(output_path, heights, layer_cfgs):
    """导出分层土参数图（和 heightmap 同尺寸，7 通道 = SOIL_KEYS 顺序）。
    每像素按所在高度层填该层 soil 参数；层无 soil 配置则回填 _SOIL_DEFAULT 并 warning。"""
    if not layer_cfgs:
        return
    if not any(layer.get("soil") is not None for layer in layer_cfgs):
        return
    layer_ids = _height_to_layer_indices(heights.astype(np.float32), layer_cfgs)
    layer_params = np.zeros((len(layer_cfgs), len(SOIL_KEYS)), dtype=np.float32)
    for i, layer in enumerate(layer_cfgs):
        soil = layer.get("soil")
        if soil is None:
            print(f"[WARN] layer '{layer.get('name', i)}' 无 soil 配置，回填默认 {_SOIL_DEFAULT}")
            soil = _SOIL_DEFAULT
        layer_params[i] = [float(soil[k]) for k in SOIL_KEYS]
    soil_map = layer_params[layer_ids]          # (H, W, 7)
    phi_ch = SOIL_KEYS.index("phi")
    np.save(output_path, soil_map)
    print(f"[INFO] Saved soil params map: {output_path} (shape={soil_map.shape}, "
          f"phi range=[{soil_map[..., phi_ch].min():.2f}, {soil_map[..., phi_ch].max():.2f}])")


def write_region_soil_params_map(output_path, region_ids, region_profiles):
    """Write a seven-channel soil map indexed by aligned region IDs."""
    params = np.asarray(
        [[float(profile[key]) for key in SOIL_KEYS] for profile in region_profiles],
        dtype=np.float32,
    )
    if int(region_ids.min()) < 0 or int(region_ids.max()) >= len(params):
        raise ValueError("region_ids contains an ID outside region_profiles")
    soil_map = params[region_ids]
    np.save(output_path, soil_map)
    phi_ch = SOIL_KEYS.index("phi")
    print(f"[INFO] Saved region soil params map: {output_path} (shape={soil_map.shape}, "
          f"phi range=[{soil_map[..., phi_ch].min():.2f}, {soil_map[..., phi_ch].max():.2f}])")
