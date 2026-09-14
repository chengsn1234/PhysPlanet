# -*- coding: UTF-8 -*-
"""HiRISE DTM helpers for the Gradio terrain UI."""

import csv
import os
import sys
from urllib.parse import urlparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from osgeo import gdal

gdal.UseExceptions()
_TG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HIRISE_ROOT = os.path.join(_TG_DIR, "models", "HiRISE")
_HIRISE_DATA_DIR = os.path.join(_HIRISE_ROOT, "data")
_DEFAULT_CSV = os.path.join(_HIRISE_ROOT, "HiRISE_DTM_train_dataset.csv")


def _dtm_id(img_url):
    return os.path.splitext(os.path.basename(urlparse(img_url).path))[0]


def _url_basename(url):
    return os.path.basename(urlparse(url).path)


def parse_dtm_index(csv_path=None):
    csv_path = csv_path or _DEFAULT_CSV
    if not os.path.exists(csv_path):
        return []
    seen = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img_url = (row.get("DTM") or "").strip()
            if not img_url:
                continue
            dtm_id = _dtm_id(img_url)
            if dtm_id in seen:
                continue
            title = (row.get("Name") or "").strip() or dtm_id
            seen[dtm_id] = {
                "dtm_id": dtm_id,
                "title": title,
                "img_url": img_url,
                "jp2_urls": [(row.get("Left Image") or "").strip(), (row.get("Right Image") or "").strip()],
                "scale": float(row["Scale"]) if row.get("Scale") else None,
            }
    return list(seen.values())


def find_dtm_path(dtm_id, cache_dir=None):
    cache_dir = cache_dir or _HIRISE_DATA_DIR
    p = os.path.join(cache_dir, f"{dtm_id}.IMG")
    return p if os.path.exists(p) and os.path.getsize(p) > 0 else None


def download_file(url, dest, on_progress=None):
    import requests

    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    if os.path.exists(part):
        os.remove(part)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(part, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if on_progress:
                        on_progress(done, total)
    os.replace(part, dest)
    return dest


def download_dtm(dtm_id, img_url, cache_dir=None, on_progress=None):
    cache_dir = cache_dir or _HIRISE_DATA_DIR
    return download_file(img_url, os.path.join(cache_dir, f"{dtm_id}.IMG"), on_progress)


def find_ortho_path(url, cache_dir=None):
    cache_dir = cache_dir or _HIRISE_DATA_DIR
    p = os.path.join(cache_dir, _url_basename(url))
    return p if os.path.exists(p) and os.path.getsize(p) > 0 else None


def download_ortho(url, cache_dir=None, on_progress=None):
    cache_dir = cache_dir or _HIRISE_DATA_DIR
    return download_file(url, os.path.join(cache_dir, _url_basename(url)), on_progress)


def _gdal_dtype_to_numpy(dtype):
    return {
        gdal.GDT_Byte: np.uint8,
        gdal.GDT_UInt16: np.uint16,
        gdal.GDT_Int16: np.int16,
        gdal.GDT_UInt32: np.uint32,
        gdal.GDT_Int32: np.int32,
        gdal.GDT_Float32: np.float32,
        gdal.GDT_Float64: np.float64,
    }.get(dtype, np.float32)


def read_dtm_band(path, xoff=0, yoff=0, xsize=None, ysize=None,
                  buf_xsize=None, buf_ysize=None):
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"GDAL 打不开 {path}")
    W, H = ds.RasterXSize, ds.RasterYSize
    band = ds.GetRasterBand(1)
    xsize = xsize if xsize else W - xoff
    ysize = ysize if ysize else H - yoff
    if xoff < 0 or yoff < 0 or xsize <= 0 or ysize <= 0 or xoff + xsize > W or yoff + ysize > H:
        raise ValueError(f"DTM 读取窗口越界: ({xoff},{yoff},{xsize},{ysize}) not in {W}×{H}")

    nodata = band.GetNoDataValue()
    buf = band.ReadRaster(xoff, yoff, xsize, ysize,
                          buf_xsize, buf_ysize, buf_type=gdal.GDT_Float32)
    if buf is None:
        raise RuntimeError(f"GDAL ReadRaster failed: {path}")
    elev = np.frombuffer(buf, dtype=np.float32).reshape(buf_ysize or ysize, buf_xsize or xsize).copy()

    if nodata is not None:
        elev[elev == np.float32(nodata)] = np.nan

    gt = ds.GetGeoTransform()
    px_res = abs(gt[1]) if gt and gt[1] else 1.0
    py_res = abs(gt[5]) if gt and gt[5] else px_res
    return elev, px_res, py_res, W, H


def read_raster_band(path, band_index=1, xoff=0, yoff=0, xsize=None, ysize=None,
                     buf_xsize=None, buf_ysize=None):
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"GDAL 打不开 {path}")
    W, H = ds.RasterXSize, ds.RasterYSize
    if band_index < 1 or band_index > ds.RasterCount:
        raise ValueError(f"band_index={band_index} 超出范围 1..{ds.RasterCount}")
    band = ds.GetRasterBand(band_index)
    xsize = xsize if xsize else W - xoff
    ysize = ysize if ysize else H - yoff
    if xoff < 0 or yoff < 0 or xsize <= 0 or ysize <= 0 or xoff + xsize > W or yoff + ysize > H:
        raise ValueError(f"Raster 读取窗口越界: ({xoff},{yoff},{xsize},{ysize}) not in {W}×{H}")
    dtype = _gdal_dtype_to_numpy(band.DataType)
    buf = band.ReadRaster(xoff, yoff, xsize, ysize, buf_xsize, buf_ysize, buf_type=band.DataType)
    if buf is None:
        raise RuntimeError(f"GDAL ReadRaster failed: {path}")
    arr = np.frombuffer(buf, dtype=dtype).reshape(buf_ysize or ysize, buf_xsize or xsize).copy()
    nodata = band.GetNoDataValue()
    if nodata is not None:
        arr = arr.astype(np.float32, copy=False)
        arr[arr == np.float32(nodata)] = np.nan
    return arr, W, H, ds.RasterCount


def read_dtm_preview_keep_ratio(path, max_dim=700):
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"GDAL 打不开 {path}")
    W, H = ds.RasterXSize, ds.RasterYSize
    scale = max_dim / max(W, H)
    buf_xsize = max(1, int(W * scale))
    buf_ysize = max(1, int(H * scale))
    elev, px_res, py_res, _, _ = read_dtm_band(path, buf_xsize=buf_xsize, buf_ysize=buf_ysize)
    return elev, buf_xsize, buf_ysize, px_res, W, H


def read_ortho_preview_keep_ratio(path, max_dim=700):
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"GDAL 打不开 {path}")
    W, H = ds.RasterXSize, ds.RasterYSize
    scale = max_dim / max(W, H)
    buf_xsize = max(1, int(W * scale))
    buf_ysize = max(1, int(H * scale))
    bands = min(ds.RasterCount, 3)
    chans = []
    for i in range(1, bands + 1):
        arr, _, _, _ = read_raster_band(path, i, buf_xsize=buf_xsize, buf_ysize=buf_ysize)
        chans.append(stretch_to_uint8(arr))
    if len(chans) == 1:
        rgb = np.repeat(chans[0][..., None], 3, axis=2)
    else:
        rgb = np.stack(chans[:3], axis=2)
    return rgb, buf_xsize, buf_ysize, W, H, ds.RasterCount


def stretch_to_uint8(arr):
    out = np.zeros(arr.shape, dtype=np.uint8)
    mask = np.isfinite(arr)
    if not mask.any():
        return out
    lo, hi = np.nanpercentile(arr[mask], [2, 98])
    if hi - lo < 1e-6:
        hi = float(np.nanmax(arr[mask]))
        lo = float(np.nanmin(arr[mask]))
    if hi - lo > 1e-6:
        out[mask] = np.clip((arr[mask] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return out


def to_preview_uint8(elev):
    out = np.zeros(elev.shape, dtype=np.uint8)
    mask = ~np.isnan(elev)
    if mask.any():
        vmin, vmax = elev[mask].min(), elev[mask].max()
        if vmax - vmin > 1e-6:
            out[mask] = np.clip((elev[mask] - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    return out


def to_elevation_rgb(elev):
    gray = to_preview_uint8(elev)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[~np.isfinite(elev)] = 0
    return rgb


def ensure_rgb(image):
    if image is None:
        return None
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 4:
        return image[..., :3]
    return image


def valid_bbox(image, threshold=5, margin=2):
    img = ensure_rgb(image)
    if img is None:
        return None
    valid = np.any(img > threshold, axis=2)
    if not valid.any():
        return 0, 0, img.shape[1], img.shape[0]
    ys, xs = np.where(valid)
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(img.shape[1], int(xs.max()) + margin + 1)
    y1 = min(img.shape[0], int(ys.max()) + margin + 1)
    return x0, y0, x1, y1


def largest_valid_inner_bbox(image, threshold=5, margin=2):
    img = ensure_rgb(image)
    if img is None:
        return None
    valid = np.any(img > threshold, axis=2)
    return largest_mask_inner_bbox(valid, margin)


def largest_finite_inner_bbox(elev, margin=2):
    if elev is None:
        return None
    return largest_mask_inner_bbox(np.isfinite(elev), margin)


def largest_mask_inner_bbox(valid, margin=2):
    H, W = valid.shape
    lefts = np.full(H, W, dtype=np.int32)
    rights = np.full(H, -1, dtype=np.int32)
    for y in range(H):
        xs = np.flatnonzero(valid[y])
        if xs.size:
            lefts[y] = int(xs[0])
            rights[y] = int(xs[-1]) + 1

    best = None
    best_area = 0
    for y0 in range(H):
        if rights[y0] <= lefts[y0]:
            continue
        left = int(lefts[y0])
        right = int(rights[y0])
        for y1 in range(y0 + 1, H + 1):
            row = y1 - 1
            if rights[row] <= lefts[row]:
                break
            left = max(left, int(lefts[row]))
            right = min(right, int(rights[row]))
            width = right - left
            if width <= 0:
                break
            area = width * (y1 - y0)
            if area > best_area:
                best_area = area
                best = (left, y0, right, y1)

    if best is None:
        ys, xs = np.where(valid)
        if not xs.size:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    x0, y0, x1, y1 = best
    x0 = min(max(0, x0 + margin), W - 1)
    y0 = min(max(0, y0 + margin), H - 1)
    x1 = max(min(W, x1 - margin), x0 + 1)
    y1 = max(min(H, y1 - margin), y0 + 1)
    return x0, y0, x1, y1


def crop_to_bbox(image, bbox):
    if image is None or bbox is None:
        return image
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1].copy()


def draw_crop_box(image, cx=None, cy=None, half=None):
    out = ensure_rgb(image).copy()
    H, W = out.shape[:2]
    if cx is None or cy is None:
        return out
    half = int(half) if half else min(W, H) // 4
    half = max(1, min(half, W // 2, H // 2))
    cx = int(max(half, min(W - half, cx)))
    cy = int(max(half, min(H - half, cy)))
    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half, cy + half
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 230, 0), 2)
    cv2.line(out, (cx - 8, cy), (cx + 8, cy), (255, 80, 40), 2)
    cv2.line(out, (cx, cy - 8), (cx, cy + 8), (255, 80, 40), 2)
    return out


def crop_preview(image, cx=None, cy=None, half=None, max_dim=360):
    img = ensure_rgb(image)
    if img is None or cx is None or cy is None:
        return None
    H, W = img.shape[:2]
    half = int(half) if half else min(W, H) // 4
    half = max(1, min(half, W // 2, H // 2))
    cx = int(max(half, min(W - half, cx)))
    cy = int(max(half, min(H - half, cy)))
    crop = img[cy - half:cy + half, cx - half:cx + half].copy()
    h, w = crop.shape[:2]
    if max(h, w) < max_dim:
        scale = max_dim / max(h, w)
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    return crop


def black_pixel_ratio(image, threshold=5):
    img = ensure_rgb(image)
    if img is None:
        return 0.0
    black = np.all(img <= threshold, axis=2)
    return float(black.mean())


def finite_range(elev):
    mask = ~np.isnan(elev)
    if not mask.any():
        raise ValueError("DTM 区域全是 nodata")
    return float(elev[mask].min()), float(elev[mask].max())


def square_crop(elev, cx, cy, half):
    H, W = elev.shape
    half = int(min(half, W // 2, H // 2))
    if half < 4:
        raise ValueError(f"裁剪区域太小(half={half}),图 {W}×{H}")
    cx = int(max(half, min(W - half, cx)))
    cy = int(max(half, min(H - half, cy)))
    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half, cy + half
    return elev[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def to_heightmap_png(elev, out_path, min_terrain_height=0.5):
    mask = ~np.isnan(elev)
    if not mask.any():
        raise ValueError("crop 区域全是 nodata,换个位置裁剪")
    if not mask.all():
        if mask.mean() < 0.3:
            raise ValueError(f"crop 区域 nodata 过多({1 - mask.mean():.0%}),换个位置裁剪")
        fill = float(np.median(elev[mask]))
        elev = np.where(mask, elev, fill)
    z_min = float(elev.min())
    z_max = float(elev.max())
    terrain_height = max(z_max - z_min, min_terrain_height)
    png = np.clip(np.rint((elev - z_min) / terrain_height * 65535), 0, 65535).astype(np.uint16)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not cv2.imwrite(out_path, png):
        raise RuntimeError(f"写 heightmap PNG 失败: {out_path}")
    return {"terrain_height": terrain_height, "z_min": z_min, "z_max": z_max}
