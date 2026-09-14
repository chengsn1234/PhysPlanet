# -*- coding: UTF-8 -*-
"""Terrain generation WebUI."""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gradio as gr

# Isaac's Pydantic stack may expose boolean JSON schemas that Gradio 4.44 cannot parse.
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

from terrain_generation.scripts.config import SOIL_KEYS, _SOIL_DEFAULT, load_config
from terrain_generation.scripts.build import generate_terrain
from terrain_generation.scripts.terrain import perlin_heightmap, write_perlin_heightmap

_TG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_TG_DIR, "models")
_OUTPUT_DIR = os.path.join(_TG_DIR, "terrain")
_OUTPUT_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_GENERATED_HM_DIR = os.path.join(_MODELS_DIR, "heightmaps", "generated")
_MAX_HIRISE_CROP_PIXELS = 25_000_000
_MAX_HIRISE_MESH_VERTICES = 1_500_000
_PERLIN_DEFAULTS = {
    "resolution": 512,
    "seed": 42,
    "scale": 6.0,
    "octaves": 4,
    "persistence": 0.5,
    "lacunarity": 2.0,
}
_PHYSICS_HEADERS = [
    "name", "path", "static_friction", "dynamic_friction", "restitution",
    "Kc", "Kphi", "n0", "n1", "c", "phi", "K",
]


def _default_cfg():
    return load_config(None, _CliOverrides())


def scan_heightmaps():
    hm_dir = os.path.join(_MODELS_DIR, "heightmaps")
    if not os.path.isdir(hm_dir):
        return []
    names = [f[:-4] for f in os.listdir(hm_dir) if f.startswith("HM") and f.endswith(".png")]
    gen_dir = os.path.join(hm_dir, "generated")
    if os.path.isdir(gen_dir):
        names += [f"generated/{f[:-4]}" for f in os.listdir(gen_dir) if f.endswith(".png")]
    return sorted(names, key=lambda s: int(s[2:]) if s.startswith("HM") and s[2:].isdigit() else 1e9)


class _CliOverrides:
    _KEYS = ("hm", "terrain_size", "terrain_height", "terrain_downsample",
             "gt_resolution", "rock_downsample", "rock_k", "seed")

    def __init__(self, **kw):
        for k in self._KEYS:
            setattr(self, k, kw.get(k))


def _clean_output_name(output_name, field_name="output_name"):
    name = (output_name or "").strip()
    if not name:
        raise ValueError(f"Please fill in {field_name}")
    if name in {".", ".."} or not _OUTPUT_NAME_RE.fullmatch(name):
        raise ValueError(f"{field_name} may only contain letters, digits, underscores, dots and dashes")
    return name


def _optional_positive_int(value, field_name):
    if value is None or value == "":
        return None
    ivalue = int(value)
    if ivalue < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return ivalue


def _optional_nonnegative_int(value, field_name):
    if value is None or value == "":
        return None
    ivalue = int(value)
    if ivalue < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return ivalue


def _optional_nonnegative_float(value, field_name):
    if value is None or value == "":
        return None
    fvalue = float(value)
    if fvalue < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return fvalue


def _optional_positive_float(value, field_name):
    if value is None or value == "":
        return None
    fvalue = float(value)
    if fvalue <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return fvalue


def _float_list(value, field_name):
    if isinstance(value, (list, tuple)):
        items = value
    else:
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"{field_name} must not be empty")
        items = re.split(r"[\s,]+", text)
    values = [float(v) for v in items if str(v).strip()]
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"{field_name} must be a list of positive numbers")
    return values


def _list_text(values):
    return ", ".join(str(v) for v in (values or []))


def _hm_path_from_name(hm_name):
    if not hm_name or hm_name.startswith("HM"):
        return None
    path = os.path.join(_MODELS_DIR, "heightmaps", hm_name + ".png")
    return path if os.path.exists(path) else None


def _heightmap_preview_from_array(img):
    import cv2
    import numpy as np

    if img is None:
        return None
    arr = img.astype(np.float32)
    arr -= float(np.nanmin(arr))
    arr /= max(float(np.nanmax(arr)), 1.0e-6)
    gray = np.clip(arr * 255, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def local_hm_preview(hm_name):
    import cv2

    path = _hm_path_from_name(hm_name)
    if not path and hm_name and hm_name.startswith("HM"):
        path = os.path.join(_MODELS_DIR, "heightmaps", hm_name + ".png")
    if not path or not os.path.exists(path):
        return None, "Please select a heightmap"
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, f"Failed to read heightmap: {hm_name}"
    return _heightmap_preview_from_array(img), f"{hm_name}  {img.shape[1]}×{img.shape[0]}  {img.dtype}"


def perlin_hm_preview(resolution, seed, scale, octaves, persistence, lacunarity):
    try:
        img = perlin_heightmap(resolution, seed, scale, octaves, persistence, lacunarity)
        return _heightmap_preview_from_array(img), f"Perlin {img.shape[1]}×{img.shape[0]}"
    except Exception as exc:
        return None, f"Perlin preview failed: {exc}"


def _download_progress(progress, done_bytes, total_bytes):
    done_mb = done_bytes // 1024 // 1024
    if total_bytes and total_bytes > 0:
        progress(done_bytes / total_bytes, desc=f"Downloading {done_mb} MB")
    else:
        progress(0, desc=f"Downloading {done_mb} MB")


def _layers_to_table(cfg):
    rows = []
    for layer in cfg.get("texture_layers", []):
        soil = layer.get("soil") or _SOIL_DEFAULT
        rows.append([
            layer.get("name", ""),
            layer.get("path", ""),
            layer.get("static_friction", ""),
            layer.get("dynamic_friction", ""),
            layer.get("restitution", 0.0),
            *[soil.get(k, _SOIL_DEFAULT[k]) for k in SOIL_KEYS],
        ])
    return rows


def _required_text(value, field_name):
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Please fill in {field_name}")
    return text


def _required_float(value, field_name):
    if value is None or value == "":
        raise ValueError(f"{field_name} must not be empty")
    return float(value)


def _table_to_layers(table):
    layers = []
    for i, row in enumerate(table or []):
        if isinstance(row, dict):
            row = [row.get(h, "") for h in _PHYSICS_HEADERS]
        if not row or all(v is None or v == "" for v in row):
            continue
        values = list(row) + [""] * (len(_PHYSICS_HEADERS) - len(row))
        soil = {
            key: _required_float(values[5 + j], f"physics row {i + 1}: {key}")
            for j, key in enumerate(SOIL_KEYS)
        }
        layers.append({
            "name": _required_text(values[0], f"physics row {i + 1}: name"),
            "path": _required_text(values[1], f"physics row {i + 1}: path"),
            "static_friction": _required_float(values[2], f"physics row {i + 1}: static_friction"),
            "dynamic_friction": _required_float(values[3], f"physics row {i + 1}: dynamic_friction"),
            "restitution": _required_float(values[4], f"physics row {i + 1}: restitution"),
            "soil": soil,
        })
    if not layers:
        raise ValueError("physics needs at least one row")
    return layers


from terrain_generation.ui import hirise  # noqa: E402

_HIRISE_ROOT = os.path.join(_MODELS_DIR, "HiRISE")
_HIRISE_DATA_DIR = os.path.join(_HIRISE_ROOT, "data")
_HIRISE_PREV_MAX = 700
_HIRISE_DEFAULT_HALF = 10


def _hirise_choices():
    import glob
    local = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(_HIRISE_DATA_DIR, "*.IMG"))}
    items = []
    seen = set()
    try:
        for it in hirise.parse_dtm_index():
            seen.add(it["dtm_id"])
            tag = "[local] " if it["dtm_id"] in local else "[downloadable] "
            items.append(f"{tag}{it['title'][:40]} ({it['dtm_id']})")
    except Exception:
        pass
    local_only = [f"[local] {dtm_id}" for dtm_id in sorted(local - seen)]
    return local_only + items


def _parse_dtm_choice(choice):
    if not choice:
        return None
    if "(" in choice and choice.endswith(")"):
        return choice[choice.rfind("(") + 1:-1]
    if choice.startswith("[local] "):
        return choice[len("[local] "):]
    return None


def _hirise_entry(dtm_id):
    return next((x for x in hirise.parse_dtm_index() if x["dtm_id"] == dtm_id), None)


def _rotate_display(image):
    import numpy as np

    return np.rot90(image, k=3).copy()


def _display_to_cropped_xy(x, y, cropped_h):
    return y, cropped_h - 1 - x


def _trim_preview_for_display(preview):
    bbox = hirise.largest_valid_inner_bbox(preview)
    if not bbox:
        h, w = preview.shape[:2]
        return preview, (0, 0, w, h), 0.0
    x0, y0, x1, y1 = bbox
    h, w = preview.shape[:2]
    trimmed = hirise.crop_to_bbox(preview, bbox)
    removed = 1.0 - ((x1 - x0) * (y1 - y0) / max(w * h, 1))
    return trimmed, bbox, removed


def _hirise_load_result(preview, crop, state, status, cx=None, cy=None):
    coord = f"Center: ({cx}, {cy}) px" if cx is not None and cy is not None else "Center: not selected"
    return preview, crop, state, status, cx, cy, coord


def _display_crop_to_dtm_window(preview_state, cx, cy, half_prev):
    import numpy as np
    if not preview_state:
        return None

    src_prev_w = int(preview_state["source_prev_w"])
    src_prev_h = int(preview_state["source_prev_h"])
    bbox = preview_state.get("preview_bbox") or (0, 0, src_prev_w, src_prev_h)
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = [int(v) for v in bbox]
    W = int(preview_state["dtm_w"])
    H = int(preview_state["dtm_h"])
    display = preview_state.get("preview_image")
    if display is None:
        return None
    display_h, display_w = display.shape[:2]

    cx = int(np.clip(cx if cx is not None else display_w // 2, 0, display_w - 1))
    cy = int(np.clip(cy if cy is not None else display_h // 2, 0, display_h - 1))
    crop_prev_h = max(1, bbox_y1 - bbox_y0)
    nx, ny = _display_to_cropped_xy(cx, cy, crop_prev_h)
    nx = float(np.clip(nx + bbox_x0, 0, src_prev_w - 1))
    ny = float(np.clip(ny + bbox_y0, 0, src_prev_h - 1))
    ox = int(nx / max(src_prev_w - 1, 1) * (W - 1))
    oy = int(ny / max(src_prev_h - 1, 1) * (H - 1))

    half_prev = int(half_prev) if half_prev else _HIRISE_DEFAULT_HALF
    half_orig = int(half_prev * min(W / src_prev_w, H / src_prev_h))
    valid_bbox = preview_state.get("dtm_valid_bbox") or (0, 0, W, H)
    vx0, vy0, vx1, vy1 = [int(v) for v in valid_bbox]
    valid_w = max(1, vx1 - vx0)
    valid_h = max(1, vy1 - vy0)
    max_half = max(1, min(W // 2, H // 2, valid_w // 2, valid_h // 2))
    half_orig = min(max(8, half_orig), max_half)
    ox = int(max(vx0 + half_orig, min(vx1 - half_orig, ox)))
    oy = int(max(vy0 + half_orig, min(vy1 - half_orig, oy)))
    crop_side = 2 * half_orig
    return ox - half_orig, oy - half_orig, crop_side, crop_side


def _hirise_update_crop(preview_state, cx, cy, half_prev):
    if not preview_state or preview_state.get("preview_image") is None:
        return None, None, ""
    image = preview_state["preview_image"]
    h, w = image.shape[:2]
    cx = int(cx) if cx is not None else w // 2
    cy = int(cy) if cy is not None else h // 2
    half_prev = int(half_prev) if half_prev else _HIRISE_DEFAULT_HALF
    overlay = hirise.draw_crop_box(image, cx, cy, half_prev)
    crop = hirise.crop_preview(image, cx, cy, half_prev)
    note = ""
    window = _display_crop_to_dtm_window(preview_state, cx, cy, half_prev)
    if window:
        x0, y0, xsize, ysize = window
        try:
            buf_size = min(420, xsize, ysize)
            elev, _, _, _, _ = hirise.read_dtm_band(
                preview_state["dtm_path"],
                xoff=x0,
                yoff=y0,
                xsize=xsize,
                ysize=ysize,
                buf_xsize=buf_size,
                buf_ysize=buf_size,
            )
            crop = hirise.to_elevation_rgb(elev)
        except Exception as exc:
            note = f"⚠️ High-res DTM crop preview failed; falling back to the screen preview: {exc}"

    black_ratio = hirise.black_pixel_ratio(crop)
    if not note and black_ratio > 0.2:
        note = ("⚠️ About {:.0%} of the current crop is black/no-data area; "
                "move the box further into the valid image region.".format(black_ratio))
    return overlay, crop, note


def hirise_load(choice, progress=gr.Progress()):
    dtm_id = _parse_dtm_choice(choice)
    if not dtm_id:
        return _hirise_load_result(None, None, None, "❌ Please select a DTM")
    path = hirise.find_dtm_path(dtm_id) or os.path.join(_HIRISE_DATA_DIR, f"{dtm_id}.IMG")
    try:
        if not os.path.exists(path):
            entry = _hirise_entry(dtm_id)
            if not entry:
                return _hirise_load_result(None, None, None, f"❌ {dtm_id} not found in the CSV index")
            progress(0, desc=f"Downloading {dtm_id}...")
            hirise.download_dtm(dtm_id, entry["img_url"], _HIRISE_DATA_DIR,
                                lambda d, t: _download_progress(progress, d, t))

        elev, prev_w, prev_h, pxres, W, H = hirise.read_dtm_preview_keep_ratio(path, _HIRISE_PREV_MAX)
        z_min, z_max = hirise.finite_range(elev)
        dtm_bbox_prev = hirise.largest_finite_inner_bbox(elev, margin=2)
        if dtm_bbox_prev:
            bx0, by0, bx1, by1 = dtm_bbox_prev
            dtm_valid_bbox = (
                int(bx0 / max(prev_w, 1) * W),
                int(by0 / max(prev_h, 1) * H),
                int(bx1 / max(prev_w, 1) * W),
                int(by1 / max(prev_h, 1) * H),
            )
        else:
            dtm_valid_bbox = (0, 0, W, H)

        entry = _hirise_entry(dtm_id)
        if not entry:
            return _hirise_load_result(None, None, None, f"❌ CSV missing {dtm_id}")
        jp2_url = entry["jp2_urls"][0]
        if not jp2_url:
            return _hirise_load_result(None, None, None, "❌ No left JP2")
        jp2_path = hirise.find_ortho_path(jp2_url)
        if not jp2_path:
            progress(0, desc="Downloading JP2...")
            jp2_path = hirise.download_ortho(jp2_url, _HIRISE_DATA_DIR,
                                             lambda d, t: _download_progress(progress, d, t))
        preview, src_prev_w, src_prev_h, src_w, src_h, bands = hirise.read_ortho_preview_keep_ratio(
            jp2_path, _HIRISE_PREV_MAX)
        source_note = f"JP2 {src_w}×{src_h}, {bands} band"

        preview, preview_bbox, removed_ratio = _trim_preview_for_display(preview)
        preview = _rotate_display(preview)
        state = {
            "dtm_id": dtm_id,
            "dtm_path": path,
            "preview_image": preview,
            "source_prev_w": src_prev_w,
            "source_prev_h": src_prev_h,
            "preview_bbox": preview_bbox,
            "dtm_valid_bbox": dtm_valid_bbox,
            "dtm_w": W,
            "dtm_h": H,
            "pxres": pxres,
        }
        trim_note = f"\ntrim {removed_ratio:.0%}" if removed_ratio > 0.01 else ""
        status = f"{dtm_id}\nDTM {W}×{H}, {pxres:.2f}m/px, z {z_min:.0f}~{z_max:.0f}m\n{source_note}{trim_note}"
        return _hirise_load_result(preview, None, state, status)
    except Exception:
        import traceback
        return _hirise_load_result(None, None, None, f"❌ Load failed:\n{traceback.format_exc()}")


def hirise_on_select(preview_state, half_prev, evt: gr.SelectData):
    idx = evt.index
    if isinstance(idx, (list, tuple)) and len(idx) == 2 and idx[0] is not None and idx[1] is not None:
        cx, cy = int(idx[0]), int(idx[1])
    else:
        image = preview_state.get("preview_image") if preview_state else None
        cx, cy = (image.shape[1] // 2, image.shape[0] // 2) if image is not None else (None, None)
    overlay, crop, note = _hirise_update_crop(preview_state, cx, cy, half_prev)
    return overlay, crop, note, cx, cy, _hirise_load_result(None, None, None, "", cx, cy)[-1]


def generate_from_source(source, dtm_choice, cx, cy, half_prev, preview_state,
                         hm_name, perlin_resolution, perlin_seed, perlin_scale,
                         perlin_octaves, perlin_persistence, perlin_lacunarity,
                         terrain_size, terrain_height, terrain_downsample, gt_resolution, rock_k, rock_seed,
                         rock_sizes, rock_downsample, texture_resolution, texture_quilting, sky,
                         physics_table, output_name, progress=gr.Progress()):
    hm_png = None
    try:
        output_name = _clean_output_name(output_name, "terrain_name")
        terrain_size = _optional_positive_float(terrain_size, "terrain_size")
        terrain_height = _optional_positive_float(terrain_height, "terrain_height")
        terrain_downsample = _optional_positive_int(terrain_downsample, "terrain_downsample")
        gt_resolution = _optional_positive_float(gt_resolution, "gt_resolution")
        rock_k = _optional_nonnegative_float(rock_k, "rock_k")
        rock_seed = _optional_nonnegative_int(rock_seed, "rock_seed")
        rock_sizes = _float_list(rock_sizes, "rock_sizes")
        rock_downsample = _optional_positive_float(rock_downsample, "rock_downsample")
        texture_resolution = _optional_positive_float(texture_resolution, "texture_resolution")
        if sky not in ("mars", "lunar"):
            return None, None, "❌ sky must be mars or lunar"

        cli = _CliOverrides(
            hm=int(hm_name[2:]) if hm_name and hm_name.startswith("HM") and hm_name[2:].isdigit() else None,
            terrain_size=terrain_size,
            terrain_height=terrain_height,
            terrain_downsample=terrain_downsample,
            gt_resolution=gt_resolution,
            rock_downsample=rock_downsample,
            rock_k=rock_k,
            seed=rock_seed,
        )
        cfg = load_config(None, cli)
        source_note = source

        if source == "HiRISE DTM":
            dtm_id = _parse_dtm_choice(dtm_choice)
            if not dtm_id:
                return None, None, "❌ Select and load a DTM first"
            path = hirise.find_dtm_path(dtm_id)
            if not path:
                return None, None, "❌ DTM not downloaded — click \"Load preview\" first"
            if preview_state and preview_state.get("dtm_id") == dtm_id:
                pxres = float(preview_state["pxres"])
                window = _display_crop_to_dtm_window(preview_state, cx, cy, half_prev)
                if window is None:
                    return None, None, "❌ Invalid preview state — reload the preview"
                x0, y0, crop_side, _ = window
            else:
                _, prev_w, prev_h, pxres, W, H = hirise.read_dtm_preview_keep_ratio(path, _HIRISE_PREV_MAX)
                half_orig = int(10 * min(W / prev_w, H / prev_h))
                half_orig = max(8, min(half_orig, W // 2, H // 2))
                crop_side = 2 * half_orig
                x0 = W // 2 - half_orig
                y0 = H // 2 - half_orig

            if crop_side * crop_side > _MAX_HIRISE_CROP_PIXELS:
                return None, None, f"❌ Crop too large: {crop_side}x{crop_side} px"
            mesh_side = (crop_side + int(cfg.get("terrain_downsample", 1)) - 1) // int(cfg.get("terrain_downsample", 1))
            if mesh_side * mesh_side > _MAX_HIRISE_MESH_VERTICES:
                return None, None, f"❌ Mesh too dense: about {mesh_side}x{mesh_side} vertices after downsampling"

            progress(0.35, desc=f"Cropping DTM {crop_side}x{crop_side} px...")
            sub, _, _, _, _ = hirise.read_dtm_band(path, xoff=x0, yoff=y0,
                                                   xsize=crop_side, ysize=crop_side)
            hm_png = os.path.join(_HIRISE_DATA_DIR, f"_crop_{dtm_id}_{output_name}.png")
            info = hirise.to_heightmap_png(sub, hm_png)
            cfg["heightmap_path"] = hm_png
            cfg["terrain_size"] = terrain_size or crop_side * pxres
            cfg["terrain_height"] = terrain_height or info["terrain_height"]
            source_note = f"HiRISE {dtm_id}"

        elif source == "Local HM":
            if not hm_name:
                return None, None, "❌ Please select a heightmap"
            hm_path = _hm_path_from_name(hm_name)
            if hm_path:
                cfg["heightmap_path"] = hm_path
            source_note = hm_name

        elif source == "Perlin HM":
            hm_path = os.path.join(_GENERATED_HM_DIR, f"perlin_{output_name}.png")
            cfg["heightmap_path"] = write_perlin_heightmap(
                hm_path, resolution=perlin_resolution, seed=perlin_seed, scale=perlin_scale,
                octaves=perlin_octaves, persistence=perlin_persistence, lacunarity=perlin_lacunarity,
            )
            source_note = f"generated/perlin_{output_name}"

        else:
            return None, None, "❌ Unknown height source"

        cfg["rock_sizes"] = rock_sizes
        cfg["texture_resolution"] = texture_resolution
        cfg["texture_quilting"] = bool(texture_quilting)
        cfg["sky"] = sky
        cfg["texture_layers"] = _table_to_layers(physics_table)

        progress(0.65, desc="Generating terrain...")
        generate_terrain(cfg, output_name)

        out = os.path.join(_OUTPUT_DIR, output_name)
        blended = os.path.join(out, "terrain_blended.jpg")
        layers = os.path.join(out, "terrain_layers_debug.png")
        status = (f"✅ Done → terrain/{output_name}/\n"
                  f"  source={source_note}\n"
                  f"  terrain_size={cfg.get('terrain_size')}m terrain_height={cfg.get('terrain_height')}m\n"
                  f"  gt_resolution={cfg.get('gt_resolution')}m\n"
                  f"  rock_k={cfg.get('rock_k')} rock_seed={cfg.get('rock_seed')} "
                  f"rock_sizes={cfg.get('rock_sizes')}\n"
                  f"  sky={cfg.get('sky')} texture_resolution={cfg.get('texture_resolution')}m/px quilting={cfg.get('texture_quilting')} "
                  f"layers={len(cfg.get('texture_layers', []))}")
        return (blended if os.path.exists(blended) else None,
                layers if os.path.exists(layers) else None,
                status)
    except Exception:
        import traceback
        return None, None, f"❌ Generation failed:\n{traceback.format_exc()}"
    finally:
        if hm_png and os.path.exists(hm_png):
            try:
                os.remove(hm_png)
            except OSError:
                pass


def _source_visibility(source):
    return (
        gr.update(visible=source == "HiRISE DTM"),
        gr.update(visible=source == "Local HM"),
        gr.update(visible=source == "Perlin HM"),
    )


with gr.Blocks(title="Terrain Generator") as demo:
    gr.Markdown("# Terrain Generator")
    cfg0 = _default_cfg()
    hi_state = gr.State(None)
    cx_state = gr.State(None)
    cy_state = gr.State(None)

    with gr.Row():
        source = gr.Radio(
            choices=["HiRISE DTM", "Local HM", "Perlin HM"],
            value="HiRISE DTM",
            label="height source",
        )
        output_name = gr.Textbox(label="terrain_name")

    with gr.Accordion("HiRISE DTM", open=True, visible=True) as hirise_box:
        with gr.Row():
            dtm_choice = gr.Dropdown(choices=_hirise_choices(), label="DTM")
            load_btn = gr.Button("Load preview")
        load_status = gr.Textbox(label="Status", lines=3)
        with gr.Row():
            hi_preview = gr.Image(type="numpy", interactive=False, label="Ortho preview")
            hi_crop_preview = gr.Image(type="numpy", interactive=False, label="DTM crop")
        crop_status = gr.Textbox(label="Note", lines=1)
        with gr.Row():
            crop_coord = gr.Textbox(label="Center", value="Center: not selected", interactive=False)
            half_n = gr.Number(label="Half size (px)", value=10)

    with gr.Accordion("Local HM", open=True, visible=False) as local_box:
        hm = gr.Dropdown(choices=scan_heightmaps(), label="heightmap")
        local_hm_image = gr.Image(type="numpy", label="heightmap preview", height=360)
        local_hm_status = gr.Textbox(label="Status", lines=1)

    with gr.Accordion("Perlin HM", open=True, visible=False) as perlin_box:
        with gr.Row():
            perlin_resolution = gr.Number(label="perlin_resolution", value=_PERLIN_DEFAULTS["resolution"])
            perlin_seed = gr.Number(label="perlin_seed", value=_PERLIN_DEFAULTS["seed"])
            perlin_scale = gr.Number(label="perlin_scale", value=_PERLIN_DEFAULTS["scale"])
        with gr.Row():
            perlin_octaves = gr.Number(label="perlin_octaves", value=_PERLIN_DEFAULTS["octaves"])
            perlin_persistence = gr.Number(label="perlin_persistence", value=_PERLIN_DEFAULTS["persistence"])
            perlin_lacunarity = gr.Number(label="perlin_lacunarity", value=_PERLIN_DEFAULTS["lacunarity"])
        perlin_preview0, perlin_status0 = perlin_hm_preview(**_PERLIN_DEFAULTS)
        perlin_hm_image = gr.Image(type="numpy", label="heightmap preview", value=perlin_preview0, height=360)
        perlin_hm_status = gr.Textbox(label="Status", value=perlin_status0, lines=1)

    gr.Markdown("template: terrain_config.yaml")
    with gr.Row():
        terrain_size = gr.Number(label="terrain_size", value=cfg0.get("terrain_size"))
        terrain_height = gr.Number(label="terrain_height", value=cfg0.get("terrain_height"))
        terrain_downsample = gr.Number(label="terrain_downsample", value=cfg0.get("terrain_downsample"))
        gt_resolution = gr.Number(label="gt_resolution", value=cfg0.get("gt_resolution"))
    with gr.Row():
        rock_k = gr.Number(label="rock_k", value=cfg0.get("rock_k"))
        rock_seed = gr.Number(label="rock_seed", value=cfg0.get("rock_seed"))
        rock_sizes = gr.Textbox(label="rock_sizes", value=_list_text(cfg0.get("rock_sizes")))
        rock_downsample = gr.Number(label="rock_downsample", value=cfg0.get("rock_downsample"))
    with gr.Row():
        texture_resolution = gr.Number(label="texture_resolution (m/px)", value=cfg0.get("texture_resolution"))
        texture_quilting = gr.Checkbox(label="texture_quilting", value=cfg0.get("texture_quilting", False))
        sky = gr.Dropdown(choices=["mars", "lunar"], label="sky", value=cfg0.get("sky", "mars"))
    physics = gr.Dataframe(
        headers=_PHYSICS_HEADERS,
        value=_layers_to_table(cfg0),
        datatype=["str", "str"] + ["number"] * 10,
        row_count=(len(cfg0.get("texture_layers", [])), "dynamic"),
        col_count=(len(_PHYSICS_HEADERS), "fixed"),
        type="array",
        label="physics",
        interactive=True,
    )

    btn = gr.Button("Generate", variant="primary")
    status = gr.Textbox(label="Status", lines=5)
    with gr.Row():
        preview_blended = gr.Image(label="texture", height=360)
        preview_layers = gr.Image(label="layers", height=360)

    load_btn.click(hirise_load, dtm_choice,
                   [hi_preview, hi_crop_preview, hi_state, load_status,
                    cx_state, cy_state, crop_coord])
    source.change(_source_visibility, source, [hirise_box, local_box, perlin_box])
    hm.change(local_hm_preview, hm, [local_hm_image, local_hm_status])
    for perlin_input in [perlin_resolution, perlin_seed, perlin_scale,
                         perlin_octaves, perlin_persistence, perlin_lacunarity]:
        perlin_input.change(
            perlin_hm_preview,
            [perlin_resolution, perlin_seed, perlin_scale,
             perlin_octaves, perlin_persistence, perlin_lacunarity],
            [perlin_hm_image, perlin_hm_status],
        )
    hi_preview.select(hirise_on_select, [hi_state, half_n],
                      [hi_preview, hi_crop_preview, crop_status, cx_state, cy_state, crop_coord])
    half_n.change(_hirise_update_crop, [hi_state, cx_state, cy_state, half_n],
                  [hi_preview, hi_crop_preview, crop_status])
    btn.click(
        generate_from_source,
        [source, dtm_choice, cx_state, cy_state, half_n, hi_state,
         hm, perlin_resolution, perlin_seed, perlin_scale,
         perlin_octaves, perlin_persistence, perlin_lacunarity,
         terrain_size, terrain_height, terrain_downsample, gt_resolution, rock_k, rock_seed,
         rock_sizes, rock_downsample, texture_resolution, texture_quilting, sky,
         physics, output_name],
        [preview_blended, preview_layers, status],
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="localhost", server_port=7860)
