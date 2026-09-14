"""把软土实验的训练日志、批量评测和演示遥测汇总成图。

子命令：
  curves   从 TensorBoard 事件文件画学习曲线（成功率 / 陷车率 / 沉陷惩罚 / 回报）
  terrain  画实验地形：土类分布 + 岩石 + 高程，用于说明土壤与高程解耦
  tracks   把演示轨迹叠在土类图上，按沉陷深度着色
  metrics  把多个评测 JSON 汇成对比条形图
  video    俯视对比动画：多个策略在同一土类图上同时推进
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

SOIL_NAMES = ("P10 firm", "P8 medium", "P4 dry sand", "P5 GRC-3")
SOIL_COLORS = ("#5a82c8", "#5abe6e", "#f0b43c", "#e1463c")
SOIL_CMAP = ListedColormap(SOIL_COLORS)
CONDITION_COLORS = {"v7": "#d62728", "v8": "#2ca02c", "v9": "#7f7f7f"}
CONDITION_LABELS = {
    "v7": "v7 baseline (no soil obs)",
    "v8": "v8 + soil physics obs",
    "v9": "v9 decorrelated control",
}


def condition_of(label: str) -> str:
    for key in CONDITION_LABELS:
        if key in label:
            return key
    return label


def load_soil_map(terrain_dir: str):
    classes = np.load(os.path.join(terrain_dir, "terrain_class_map.npy"))
    rocks = None
    rock_path = os.path.join(terrain_dir, "rock_map.npy")
    if os.path.exists(rock_path):
        rocks = np.load(rock_path)
    return classes, rocks


def draw_soil_background(ax, classes, rocks, terrain_size: float, alpha: float = 0.75):
    extent = (0.0, terrain_size, 0.0, terrain_size)
    ax.imshow(classes, origin="lower", extent=extent, cmap=SOIL_CMAP,
              vmin=1, vmax=4, alpha=alpha, interpolation="nearest")
    if rocks is not None:
        ax.contour(np.linspace(0, terrain_size, rocks.shape[1]),
                   np.linspace(0, terrain_size, rocks.shape[0]),
                   rocks > 0, levels=[0.5], colors="#2b2b2b", linewidths=0.6)
    ax.set_xlim(0, terrain_size)
    ax.set_ylim(0, terrain_size)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def soil_legend(extra=()):
    handles = [Line2D([], [], marker="s", linestyle="", markersize=9, color=c,
                      label=f"{n} soil") for n, c in zip(SOIL_NAMES, SOIL_COLORS)]
    handles.append(Line2D([], [], color="#2b2b2b", lw=1.0, label="rock outline"))
    return handles + list(extra)


# --------------------------------------------------------------------------- curves
TB_TAGS = {
    "Episode_Termination/is_success": "success rate",
    "Episode_Termination/bogged_down": "bogged-down rate",
    "Episode_Termination/rock_collision": "rock-collision rate",
    "Episode_Reward/sinkage": "sinkage penalty (per episode)",
    "Episode_Reward/progress_to_target": "progress reward",
    "Reward / Total reward (mean)": "total reward",
}


def read_tb(run_dir: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from tensorboard.backend.event_processing import event_accumulator

    files = sorted(glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True))
    if not files:
        return {}
    acc = event_accumulator.EventAccumulator(files[0], size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in TB_TAGS:
        if tag in acc.Tags()["scalars"]:
            events = acc.Scalars(tag)
            out[tag] = (np.array([e.step for e in events]), np.array([e.value for e in events]))
    return out


def smooth(values: np.ndarray, window: int = 9) -> np.ndarray:
    if len(values) < window or window < 2:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same") / np.convolve(np.ones_like(values), kernel, mode="same")


def cmd_curves(args):
    runs: dict[str, list[dict]] = {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        series = read_tb(path)
        if not series:
            print(f"[warn] no tfevents under {path}")
            continue
        runs.setdefault(condition_of(label), []).append(series)

    tags = [t for t in TB_TAGS if any(t in s for group in runs.values() for s in group)]
    cols = 3
    rows = int(np.ceil(len(tags) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.6 * rows), squeeze=False)
    for ax, tag in zip(axes.flat, tags):
        for cond, group in sorted(runs.items()):
            curves = [s[tag] for s in group if tag in s]
            if not curves:
                continue
            length = min(len(c[0]) for c in curves)
            steps = curves[0][0][:length]
            stack = np.stack([smooth(c[1][:length]) for c in curves])
            mean = stack.mean(axis=0)
            color = CONDITION_COLORS.get(cond, None)
            ax.plot(steps, mean, color=color, lw=1.8, label=f"{CONDITION_LABELS.get(cond, cond)} (n={len(curves)})")
            if len(stack) > 1:
                ax.fill_between(steps, stack.min(axis=0), stack.max(axis=0), color=color, alpha=0.18, lw=0)
        ax.set_title(TB_TAGS[tag], fontsize=11)
        ax.set_xlabel("timesteps")
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(tags):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=9)
    fig.suptitle("Soft-soil navigation: training curves (mean over seeds, band = min/max)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"[report] {args.out}")


# --------------------------------------------------------------------------- terrain
def cmd_terrain(args):
    classes, rocks = load_soil_map(args.terrain_dir)
    heights = np.load(os.path.join(args.terrain_dir, "height_gt_map.npy"))
    size = args.terrain_size

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    draw_soil_background(axes[0], classes, rocks, size)
    axes[0].set_title("soil class map (physics ground truth)")
    axes[0].legend(handles=soil_legend(), loc="upper right", fontsize=8, framealpha=0.9)

    im = axes[1].imshow(heights, origin="lower", extent=(0, size, 0, size), cmap="terrain")
    axes[1].set_title("elevation [m]")
    axes[1].set_xlabel("x [m]")
    axes[1].set_aspect("equal")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    # 解耦是实验成立的前提：策略不能靠高程图猜出软硬
    rows = np.linspace(0, heights.shape[0] - 1, classes.shape[0]).round().astype(int)
    cols = np.linspace(0, heights.shape[1] - 1, classes.shape[1]).round().astype(int)
    paired = heights[rows[:, None], cols[None, :]].ravel()
    flat_classes = classes.ravel()
    r = float(np.corrcoef(flat_classes, paired)[0, 1])
    axes[2].violinplot([paired[flat_classes == c] for c in range(1, 5)], showmeans=True)
    axes[2].set_xticks(range(1, 5))
    axes[2].set_xticklabels(SOIL_NAMES)
    axes[2].set_ylabel("elevation [m]")
    axes[2].set_title(f"elevation by soil class\nPearson r = {r:+.3f} (decorrelated by design)")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Experiment terrain: {os.path.basename(args.terrain_dir)}", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"[report] {args.out} (elevation/soil correlation r={r:+.3f})")


# --------------------------------------------------------------------------- tracks
MAX_STEP_DISTANCE = 2.0  # 一个控制步 0.2s，车速上限 ~0.35m/s，超过必然是重置传送


def episode_segments(data, env: int):
    """把一个环境的时间序列按 episode 编号切成若干段。

    除了按编号切，还按位置跳变兜底：早期录的数据里重置帧的编号有偏差，
    不拦掉会在轨迹图上画出一条横跨地图的直线。
    """
    episodes = data["episode"][:, env]
    xy = data["xy"][:, env]
    jumped = np.linalg.norm(np.diff(xy, axis=0), axis=1) > MAX_STEP_DISTANCE
    boundaries = np.flatnonzero((np.diff(episodes) != 0) | jumped) + 1
    return np.split(np.arange(len(episodes)), boundaries)


def cmd_tracks(args):
    datasets = []
    for spec in args.telemetry:
        label, path = spec.split("=", 1)
        datasets.append((label, np.load(path, allow_pickle=True)))

    terrain_dir = str(datasets[0][1]["terrain_dir"])
    size = float(datasets[0][1]["terrain_size"])
    classes, rocks = load_soil_map(terrain_dir)

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.4 * len(datasets), 6.6), squeeze=False)
    for ax, (label, data) in zip(axes[0], datasets):
        draw_soil_background(ax, classes, rocks, size, alpha=0.6)
        sinkage = data["sinkage"].mean(axis=2) * 100.0
        deep = 0.0
        total = 0
        for env in range(data["xy"].shape[1]):
            for segment in episode_segments(data, env):
                if len(segment) < 3:
                    continue
                xy = data["xy"][segment, env]
                sc = sinkage[segment, env]
                ax.scatter(xy[:, 0], xy[:, 1], c=sc, cmap="inferno_r", vmin=1.0, vmax=8.0, s=3.5, zorder=3)
                ax.plot(xy[0, 0], xy[0, 1], marker="o", color="white", mec="black", ms=6, zorder=4)
                deep += float((sc > args.safe_sinkage * 100.0).sum())
                total += len(sc)
        ax.set_title(f"{CONDITION_LABELS.get(condition_of(label), label)}\n"
                     f"time above {args.safe_sinkage * 100:.0f} cm sinkage: {deep / max(total, 1) * 100:.1f}%")
    scatter = axes[0][-1].collections[-1]
    fig.colorbar(scatter, ax=axes[0], fraction=0.03, label="mean sinkage [cm]")
    axes[0][0].legend(handles=soil_legend([
        Line2D([], [], marker="o", ls="", color="white", mec="black", label="episode start")]),
        loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle("Where each policy drives (held-out terrain), coloured by sinkage", fontsize=13)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"[report] {args.out}")


# --------------------------------------------------------------------------- metrics
METRIC_PANELS = [
    ("success_rate", "success rate", True),
    ("rate_bogged_down", "bogged-down rate", False),
    ("deep_fraction", "fraction of time above safe sinkage", False),
    ("sinkage_mean_cm", "mean sinkage [cm]", False),
    ("sinkage_p95_cm", "p95 of per-episode max sinkage [cm]", False),
    ("rate_rock_collision", "rock-collision rate", False),
]


def cmd_metrics(args):
    groups: dict[str, list[dict]] = {}
    for path in args.eval:
        with open(path) as f:
            payload = json.load(f)
        summary = payload["summary"] if "summary" in payload else payload
        groups.setdefault(condition_of(summary.get("label") or summary["task"]), []).append(summary)

    order = [c for c in ("v7", "v8", "v9") if c in groups] + [c for c in groups if c not in CONDITION_LABELS]
    fig, axes = plt.subplots(2, 4, figsize=(21, 8.5))

    for ax, (key, title, higher_better) in zip(axes.flat, METRIC_PANELS):
        for i, cond in enumerate(order):
            values = [g[key] for g in groups[cond] if key in g]
            if not values:
                continue
            ax.bar(i, np.mean(values), color=CONDITION_COLORS.get(cond, "#888"), width=0.6)
            ax.scatter([i] * len(values), values, color="black", zorder=3, s=22)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        ax.set_title(f"{title}  ({'higher' if higher_better else 'lower'} is better)", fontsize=10)
        ax.grid(alpha=0.3, axis="y")

    # 停留占比 vs 地形面积占比：直接反映策略是否主动挑硬地走
    ax = axes.flat[len(METRIC_PANELS)]
    width = 0.8 / (len(order) + 1)
    area = args.terrain_area or [0.30, 0.35, 0.25, 0.10]
    ax.bar(np.arange(4) - 0.4 + width / 2, area, width=width, color="#cccccc",
           edgecolor="black", label="terrain area")
    for i, cond in enumerate(order):
        fractions = np.mean([[g["soil_fraction"][n] for n in SOIL_NAMES] for g in groups[cond]], axis=0)
        ax.bar(np.arange(4) - 0.4 + width * (i + 1.5), fractions, width=width,
               color=CONDITION_COLORS.get(cond, "#888"), label=cond)
    ax.set_xticks(range(4))
    ax.set_xticklabels(SOIL_NAMES)
    ax.set_title("time spent on each soil vs. its share of the map", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    axes.flat[-1].axis("off")
    summary_lines = [f"{CONDITION_LABELS.get(c, c)}: n={len(groups[c])} runs" for c in order]
    axes.flat[-1].text(0.02, 0.95, "\n".join(summary_lines), va="top", fontsize=10)

    fig.suptitle("Held-out terrain evaluation (bars = mean over seeds, dots = individual seeds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"[report] {args.out}")


# --------------------------------------------------------------------------- video
def episode_outcome(data, env: int, episode: int = 0) -> str:
    """回合是怎么结束的；没在录制窗口内结束就返回空串。"""
    names = [str(n) for n in data["reason_names"]]
    for row in data["outcomes"]:
        if row[0] == env and row[1] == episode:
            fired = [n for i, n in enumerate(names) if row[2 + i]]
            return fired[0] if fired else "ended"
    return ""


def cmd_video(args):
    import imageio.v2 as imageio

    datasets = []
    for spec in args.telemetry:
        label, path = spec.split("=", 1)
        datasets.append((label, np.load(path, allow_pickle=True)))

    terrain_dir = str(datasets[0][1]["terrain_dir"])
    size = float(datasets[0][1]["terrain_size"])
    classes, rocks = load_soil_map(terrain_dir)
    env = args.env
    frames = min(int(d["xy"].shape[0]) for _, d in datasets)

    # 只画第一个 episode，保证各条件从同一出生点/目标点出发。
    # 动画长度取三者中最长的：截断到最短会让先结束的一方"看起来"也停了，
    # 掩盖掉"一方陷车、另一方还在走"这个关键差别。
    spans = []
    for _, data in datasets:
        segment = episode_segments(data, env)[0]
        spans.append(segment[: min(len(segment), frames)])
    span_len = max(len(s) for s in spans)

    # 各策略应当拿到同一场景；不一致说明 seed 对齐失效，图就不能横向比
    goals = np.stack([d["goal"][s[0], env] for (_, d), s in zip(datasets, spans)])
    starts = np.stack([d["xy"][s[0], env] for (_, d), s in zip(datasets, spans)])
    if len(datasets) > 1 and (goals.ptp(axis=0).max() > 0.5 or starts.ptp(axis=0).max() > 2.0):
        print(f"[warn] scenarios are not aligned across policies "
              f"(goal spread {goals.ptp(axis=0)}, start spread {starts.ptp(axis=0)}); "
              f"the panels show different tasks")

    # 所有面板共用同一视窗，否则各自缩放会让轨迹无法目视对比
    points = np.concatenate([d["xy"][s, env] for (_, d), s in zip(datasets, spans)]
                            + [goals, starts])
    pad = 10.0
    lo, hi = points.min(axis=0) - pad, points.max(axis=0) + pad
    # 正方形视窗，并平移到地形范围内，避免露出边界外的空白
    side = min(float(np.max(hi - lo)), size)
    center = np.clip((lo + hi) / 2.0, side / 2.0, size - side / 2.0)
    lo, hi = center - side / 2.0, center + side / 2.0

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 6.8), squeeze=False)
    artists = []
    for ax, (label, data), span in zip(axes[0], datasets, spans):
        xy = data["xy"][span, env]
        goal = data["goal"][span[0], env]
        draw_soil_background(ax, classes, rocks, size, alpha=0.65)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.plot(*goal, marker="*", ms=22, color="#00e000", mec="black", zorder=5)
        ax.plot(xy[0, 0], xy[0, 1], marker="o", ms=9, color="white", mec="black", zorder=5)
        trail, = ax.plot([], [], lw=2.4, color="black", zorder=6)
        rover, = ax.plot([], [], marker="o", ms=11, color="#ffd400", mec="black", zorder=7)
        title = ax.set_title("", fontsize=10)
        artists.append((data, span, xy, trail, rover, title, label,
                        episode_outcome(data, env)))

    fig.suptitle("Same start and goal, different policies (held-out terrain)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    canvas = fig.canvas

    with imageio.get_writer(args.out, fps=args.fps, macro_block_size=None) as writer:
        for t in range(span_len):
            for data, span, xy, trail, rover, title, label, outcome in artists:
                # 已结束的一方定格在最后一帧，并标出结局，避免看起来像"也停住了"
                k = min(t, len(span) - 1)
                done = t >= len(span)
                trail.set_data(xy[: k + 1, 0], xy[: k + 1, 1])
                rover.set_data([xy[k, 0]], [xy[k, 1]])
                rover.set_color("#b0b0b0" if done else "#ffd400")
                sinkage = float(data["sinkage"][span[k], env].mean()) * 100.0
                slip = float(np.abs(data["slip"][span[k], env]).mean())
                goal_dist = float(np.linalg.norm(data["goal"][span[k], env] - xy[k]))
                status = f"  <<< {outcome or 'still driving'}" if done else ""
                title.set_text(f"{CONDITION_LABELS.get(condition_of(label), label)}\n"
                               f"t {k * float(data['step_dt']):4.1f}s | sinkage {sinkage:4.1f} cm | "
                               f"slip {slip:4.2f} | to goal {goal_dist:4.1f} m{status}")
            canvas.draw()
            frame = np.asarray(canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)
    print(f"[report] {args.out} ({span_len} frames)")


# --------------------------------------------------------------------------- combined view
def _cam_strip(frames, labels, outcomes, done_flags, width: int):
    """把三路相机帧缩放拼成一条，并给已结束的一路压暗加标注。"""
    from PIL import Image, ImageDraw

    panel_w = width // len(frames)
    panel_h = round(panel_w * frames[0].shape[0] / frames[0].shape[1])
    strip = Image.new("RGB", (panel_w * len(frames), panel_h))
    for i, (frame, label, outcome, done) in enumerate(zip(frames, labels, outcomes, done_flags)):
        img = Image.fromarray(frame).resize((panel_w, panel_h), Image.BILINEAR)
        if done:  # 压暗到还能分辨土色为止，否则看不出它是陷在什么地形上
            img = Image.eval(img, lambda v: int(v * 0.62))
        draw = ImageDraw.Draw(img)
        caption = CONDITION_LABELS.get(condition_of(label), label)
        if done:
            caption += f"   <<< {outcome or 'still driving'}"
        draw.rectangle([0, 0, panel_w, 26], fill=(0, 0, 0))
        draw.text((8, 7), caption, fill=(255, 255, 255))
        strip.paste(img, (i * panel_w, 0))
    return np.asarray(strip)


def cmd_combined(args):
    """相机跟随视角（上）与俯视轨迹（下）同步合成为一个视频。

    单看俯视图不知道车在软土里具体怎么打滑，单看相机又看不出它相对目标跑偏多少，
    两者对齐到同一帧才说明问题。
    """
    import imageio.v2 as imageio

    datasets, readers = [], []
    cam_paths = dict(spec.split("=", 1) for spec in args.cam)
    for spec in args.telemetry:
        label, path = spec.split("=", 1)
        datasets.append((label, np.load(path, allow_pickle=True)))
        readers.append(imageio.get_reader(cam_paths[label]))

    terrain_dir = str(datasets[0][1]["terrain_dir"])
    size = float(datasets[0][1]["terrain_size"])
    classes, rocks = load_soil_map(terrain_dir)
    env = args.env

    spans = [episode_segments(d, env)[0] for _, d in datasets]
    span_len = max(len(s) for s in spans)
    outcomes = [episode_outcome(d, env) for _, d in datasets]

    goals = np.stack([d["goal"][s[0], env] for (_, d), s in zip(datasets, spans)])
    starts = np.stack([d["xy"][s[0], env] for (_, d), s in zip(datasets, spans)])
    points = np.concatenate([d["xy"][s, env] for (_, d), s in zip(datasets, spans)] + [goals, starts])
    lo, hi = points.min(axis=0) - 10.0, points.max(axis=0) + 10.0
    side = min(float(np.max(hi - lo)), size)
    center = np.clip((lo + hi) / 2.0, side / 2.0, size - side / 2.0)
    lo, hi = center - side / 2.0, center + side / 2.0

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 6.4), squeeze=False)
    artists = []
    for ax, (label, data), span in zip(axes[0], datasets, spans):
        xy = data["xy"][span, env]
        draw_soil_background(ax, classes, rocks, size, alpha=0.65)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.plot(*data["goal"][span[0], env], marker="*", ms=22, color="#00e000", mec="black", zorder=5)
        ax.plot(xy[0, 0], xy[0, 1], marker="o", ms=9, color="white", mec="black", zorder=5)
        trail, = ax.plot([], [], lw=2.4, color="black", zorder=6)
        rover, = ax.plot([], [], marker="o", ms=11, color="#ffd400", mec="black", zorder=7)
        artists.append((data, span, xy, trail, rover, ax.set_title("", fontsize=10)))
    fig.tight_layout()
    canvas = fig.canvas

    labels = [label for label, _ in datasets]
    latest = [None] * len(datasets)
    with imageio.get_writer(args.out, fps=args.fps, macro_block_size=None) as writer:
        for t in range(span_len):
            cam_frames, done_flags = [], []
            for i, span in enumerate(spans):
                done = t >= len(span)
                if not done:
                    latest[i] = readers[i].get_data(int(span[t]))
                cam_frames.append(latest[i])
                done_flags.append(done)

            for (data, span, xy, trail, rover, title), outcome in zip(artists, outcomes):
                k = min(t, len(span) - 1)
                done = t >= len(span)
                trail.set_data(xy[: k + 1, 0], xy[: k + 1, 1])
                rover.set_data([xy[k, 0]], [xy[k, 1]])
                rover.set_color("#b0b0b0" if done else "#ffd400")
                sinkage = float(data["sinkage"][span[k], env].mean()) * 100.0
                slip = float(np.abs(data["slip"][span[k], env]).mean())
                goal_dist = float(np.linalg.norm(data["goal"][span[k], env] - xy[k]))
                title.set_text(f"t {k * float(data['step_dt']):4.1f}s | sinkage {sinkage:4.1f} cm | "
                               f"slip {slip:4.2f} | to goal {goal_dist:4.1f} m")
            canvas.draw()
            below = np.asarray(canvas.buffer_rgba())[..., :3]

            above = _cam_strip(cam_frames, labels, outcomes, done_flags, below.shape[1])
            frame = np.concatenate([above, below], axis=0)
            if frame.shape[0] % 2:  # h264 要求偶数高
                frame = frame[:-1]
            writer.append_data(frame)
    for reader in readers:
        reader.close()
    print(f"[report] {args.out} ({span_len} frames, {frame.shape[1]}x{frame.shape[0]})")


STAT_METRICS = [
    ("deep_fraction", "time above safe sinkage", False),
    ("sinkage_mean_cm", "mean sinkage [cm]", False),
    ("sinkage_max_cm", "per-episode max sinkage [cm]", False),
    ("slip_mean", "mean slip", False),
    ("soft_fraction", "time on loose+drift soil", False),
    ("is_success", "success rate", True),
    ("bogged_down", "bogged-down rate", False),
]


def _episode_metric(record: dict, key: str) -> float:
    if key == "soft_fraction":
        return float(record["soil_fraction"][2] + record["soil_fraction"][3])
    if key in ("is_success", "bogged_down"):
        return float(record.get(f"end_{key}", False))
    return float(record[key])


def cmd_stats(args):
    """按回合自助采样给出置信区间，并单列种子间散布。

    只看条件均值会被单个异常种子带偏，两者必须一起报。
    """
    runs: dict[str, dict[str, list[dict]]] = {}
    for path in args.eval:
        payload = json.load(open(path))
        label = payload["summary"].get("label") or os.path.basename(path)
        runs.setdefault(condition_of(label), {})[label] = payload["episodes"]

    rng = np.random.default_rng(0)
    order = [c for c in ("v7", "v8", "v9") if c in runs]
    print(f"{'metric':30}" + "".join(f"{CONDITION_LABELS.get(c, c).split()[0]:>26}" for c in order))
    table: dict[str, dict[str, tuple]] = {}
    for key, title, higher_better in STAT_METRICS:
        cells = []
        for cond in order:
            per_seed = {lab: np.array([_episode_metric(r, key) for r in rows])
                        for lab, rows in runs[cond].items()}
            pooled = np.concatenate(list(per_seed.values()))
            boot = rng.choice(pooled, size=(args.bootstrap, len(pooled)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            seed_means = np.array([v.mean() for v in per_seed.values()])
            table.setdefault(key, {})[cond] = (pooled.mean(), lo, hi, seed_means)
            cells.append(f"{pooled.mean():7.3f} [{lo:.3f},{hi:.3f}]")
        arrow = "higher better" if higher_better else "lower better"
        print(f"{title[:28]:30}" + "".join(f"{c:>26}" for c in cells) + f"   ({arrow})")
        print(f"{'   per-seed':30}" + "".join(
            f"{'  '.join(f'{m:.3f}' for m in table[key][c][3]):>26}" for c in order))

    print("\n成对比较（种子区间是否重叠比 p 值更能说明问题）：")
    for key, title, higher_better in STAT_METRICS:
        for a, b in (("v8", "v7"), ("v8", "v9")):
            if a not in table[key] or b not in table[key]:
                continue
            sa, sb = table[key][a][3], table[key][b][3]
            better = (sa.max() < sb.min()) if not higher_better else (sa.min() > sb.max())
            worse = (sa.min() > sb.max()) if not higher_better else (sa.max() < sb.min())
            verdict = "✓ 全部种子占优" if better else ("✗ 全部种子劣于" if worse else "— 种子区间重叠")
            print(f"  {title[:30]:32} {a} vs {b}: {verdict}"
                  f"  ({sa.min():.3f}~{sa.max():.3f} vs {sb.min():.3f}~{sb.max():.3f})")


def _least_cost_path(cost, start_rc, goal_rc):
    """8 邻域 Dijkstra，返回 (路径格点, 几何长度[格])。cost=inf 表示不可通行。"""
    import heapq

    rows, cols = cost.shape
    dist = np.full(cost.shape, np.inf)
    prev = np.full(cost.shape + (2,), -1, dtype=np.int32)
    dist[start_rc] = 0.0
    heap = [(0.0, start_rc)]
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    while heap:
        d, (r, c) = heapq.heappop(heap)
        if (r, c) == goal_rc:
            break
        if d > dist[r, c]:
            continue
        for dr, dc in steps:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or not np.isfinite(cost[nr, nc]):
                continue
            nd = d + cost[nr, nc] * math.hypot(dr, dc)
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                prev[nr, nc] = (r, c)
                heapq.heappush(heap, (nd, (nr, nc)))
    if not np.isfinite(dist[goal_rc]):
        return None, np.inf
    path, node = [goal_rc], goal_rc
    while node != start_rc:
        node = tuple(prev[node])
        if node == (-1, -1):
            return None, np.inf
        path.append(node)
    path = np.array(path[::-1])
    length = float(np.hypot(*np.diff(path, axis=0).T).sum()) if len(path) > 1 else 0.0
    return path, length


def scenario_avoidability(classes, rocks, size, start, goal, soft_class, cell=0.5):
    """判断"软土是可以绕开的"还是"无论如何都得踩"。

    这是挑演示场景的关键：如果起点或终点本身就陷在软土里、或者周围没有硬路，
    那三个策略都只能硬闯，画面上看不出任何决策差异，这种 case 证明不了东西。
    """
    mpp = size / max(classes.shape[0] - 1, 1)
    stride = max(int(round(cell / mpp)), 1)
    grid = classes[::stride, ::stride]
    cell_m = stride * mpp

    blocked = np.zeros(grid.shape, dtype=bool)
    if rocks is not None:
        blocked = rocks[::stride, ::stride][: grid.shape[0], : grid.shape[1]] > 0

    # 软土不是禁行，只是很贵：允许"绕不过去就硬闯"，代价体现在路径的软土占比上
    weight = np.where(grid >= soft_class + 1, 40.0, np.where(grid >= soft_class, 6.0, 1.0))
    weight[blocked] = np.inf

    def to_rc(p):
        return (int(np.clip(p[1] / cell_m, 0, grid.shape[0] - 1)),
                int(np.clip(p[0] / cell_m, 0, grid.shape[1] - 1)))

    start_rc, goal_rc = to_rc(start), to_rc(goal)
    line = start + np.linspace(0, 1, 80)[:, None] * (goal - start)
    line_rc = np.stack([np.clip((line[:, 1] / cell_m).astype(int), 0, grid.shape[0] - 1),
                        np.clip((line[:, 0] / cell_m).astype(int), 0, grid.shape[1] - 1)], axis=1)
    direct_soft = float((grid[line_rc[:, 0], line_rc[:, 1]] >= soft_class).mean())

    path, path_cells = _least_cost_path(weight, start_rc, goal_rc)
    if path is None:
        return None
    path_soft = float((grid[path[:, 0], path[:, 1]] >= soft_class).mean())
    direct_m = float(np.linalg.norm(goal - start))
    detour = (path_cells * cell_m) / max(direct_m, 1.0e-6)
    return {
        "start_class": int(classes[to_rc(start)[0] * stride, to_rc(start)[1] * stride]),
        "goal_class": int(classes[to_rc(goal)[0] * stride, to_rc(goal)[1] * stride]),
        "direct_soft": direct_soft,
        "path_soft": path_soft,
        "detour": detour,
        "direct_m": direct_m,
        "path": path * cell_m,
    }


def audit_cost_grid(classes, rocks, size, cell, robot_radius, impassable_class, soft_class):
    """审计用的代价网格：最软土层和岩石按车半宽膨胀成墙，次软土层记 4 倍代价。

    审计与可达性可视化必须用同一套规则，否则图上画的路不是审计判定的那条路。
    """
    from scipy import ndimage

    mpp = size / max(classes.shape[0] - 1, 1)
    stride = max(int(round(cell / mpp)), 1)
    cell_m = stride * mpp
    grid = classes[::stride, ::stride]
    footprint = np.ones((2 * int(round(robot_radius / cell_m)) + 1,) * 2)
    blocked = ndimage.binary_dilation(grid >= impassable_class, footprint)
    if rocks is not None:
        rock_grid = rocks[::stride, ::stride][: grid.shape[0], : grid.shape[1]] > 0
        blocked |= ndimage.binary_dilation(rock_grid, footprint)

    weight = np.where(grid >= soft_class, 4.0, 1.0)
    weight[blocked] = np.inf
    return weight, grid, blocked, cell_m


def cmd_audit(args):
    """审计命令项实际采样出的起点/目标：是否可达、要绕多远、时间够不够。

    采样只校验了"起终点不在岩石和 drift 上"和直线方向的岩石遮挡，并没有做连通性检查。
    目标一旦被 drift 或岩石围死，那个回合对任何策略都是死局，训练信号是噪声。
    这个脚本用真实录到的起终点对做事后验证，地形参数改动后应当重跑。
    """
    datasets = [np.load(spec.split("=", 1)[1], allow_pickle=True) for spec in args.telemetry]
    terrain_dir = str(datasets[0]["terrain_dir"])
    size = float(datasets[0]["terrain_size"])
    classes, rocks = load_soil_map(terrain_dir)

    pairs = []
    for data in datasets:
        for env in range(data["xy"].shape[1]):
            for seg in episode_segments(data, env):
                if len(seg) >= 3:
                    pairs.append((data["xy"][seg[0], env], data["goal"][seg[0], env]))

    weight, grid, blocked, cell = audit_cost_grid(
        classes, rocks, size, args.cell, args.robot_radius,
        args.impassable_class, args.soft_class,
    )

    def to_rc(p):
        return (int(np.clip(p[1] / cell, 0, grid.shape[0] - 1)),
                int(np.clip(p[0] / cell, 0, grid.shape[1] - 1)))

    bad_start = bad_goal = unreachable = 0
    detours, times = [], []
    for start, goal in pairs:
        rc_s, rc_g = to_rc(start), to_rc(goal)
        if not np.isfinite(weight[rc_s]):
            bad_start += 1
            continue
        if not np.isfinite(weight[rc_g]):
            bad_goal += 1
            continue
        path, _ = _least_cost_path(weight, rc_s, rc_g)
        if path is None:
            unreachable += 1
            continue
        length = float(np.hypot(*np.diff(path, axis=0).T).sum()) * cell
        detours.append(length / max(float(np.linalg.norm(goal - start)), 1.0e-6))
        times.append(length / args.speed)

    n = max(len(pairs), 1)
    print(f"terrain      : {terrain_dir}")
    print(f"pairs        : {len(pairs)} start/goal pairs from {len(datasets)} recordings")
    print(f"blocked area : {blocked.mean() * 100:.1f}% "
          f"(soil class >= {args.impassable_class} and rocks, inflated {args.robot_radius} m)")
    print(f"start blocked: {bad_start:4d} ({bad_start / n * 100:.1f}%)")
    print(f"goal  blocked: {bad_goal:4d} ({bad_goal / n * 100:.1f}%)")
    print(f"unreachable  : {unreachable:4d} ({unreachable / n * 100:.1f}%)")
    if detours:
        det, tim = np.array(detours), np.array(times)
        print(f"detour ratio : median {np.median(det):.2f}  p90 {np.percentile(det, 90):.2f}  "
              f"max {det.max():.2f}")
        for thr in (1.2, 1.5, 2.0):
            print(f"   above {thr}x : {(det > thr).mean() * 100:5.1f}%")
        over = float((tim > args.time_limit).mean()) * 100.0
        print(f"time needed  : median {np.median(tim):.0f}s  p90 {np.percentile(tim, 90):.0f}s "
              f"at {args.speed} m/s  ->  {over:.1f}% exceed the {args.time_limit}s episode limit")
    verdict = (bad_start + bad_goal + unreachable) / n
    print(f"\n{'PASS' if verdict <= args.tolerance else 'FAIL'}: "
          f"{verdict * 100:.2f}% of sampled episodes are unsolvable "
          f"(tolerance {args.tolerance * 100:.2f}%)")
    return 0 if verdict <= args.tolerance else 1


def cmd_paths(args):
    """把审计判定为可达的采样场景连同绕行路线画出来，供人工复核。

    审计只输出一句"0% 不可达"，无法据此判断判定是否合理。这里按绕行系数分层抽样，
    把最直达、中等绕行、最坏绕行的场景都画出来，路线用的就是审计那套代价网格。
    """
    data = np.load(args.telemetry.split("=", 1)[1], allow_pickle=True)
    terrain_dir = str(data["terrain_dir"])
    size = float(data["terrain_size"])
    classes, rocks = load_soil_map(terrain_dir)

    weight, grid, blocked, cell = audit_cost_grid(
        classes, rocks, size, args.cell, args.robot_radius,
        args.impassable_class, args.soft_class,
    )

    def to_rc(p):
        return (int(np.clip(p[1] / cell, 0, grid.shape[0] - 1)),
                int(np.clip(p[0] / cell, 0, grid.shape[1] - 1)))

    rounds, envs = data["xy"].shape[0], data["xy"].shape[1]
    solved = []
    for t in range(0, rounds, 3):  # 收割时每轮复制了 3 帧
        for env in range(envs):
            start, goal = data["xy"][t, env], data["goal"][t, env]
            rc_s, rc_g = to_rc(start), to_rc(goal)
            if not (np.isfinite(weight[rc_s]) and np.isfinite(weight[rc_g])):
                continue
            path, cells = _least_cost_path(weight, rc_s, rc_g)
            if path is None:
                continue
            length = float(np.hypot(*np.diff(path, axis=0).T).sum()) * cell
            direct = float(np.linalg.norm(goal - start))
            line = start + np.linspace(0, 1, 80)[:, None] * (goal - start)
            line_rc = np.stack([np.clip((line[:, 1] / cell).astype(int), 0, grid.shape[0] - 1),
                                np.clip((line[:, 0] / cell).astype(int), 0, grid.shape[1] - 1)], axis=1)
            solved.append({
                "env": env, "round": t // 3, "start": start, "goal": goal,
                "path": path * cell, "detour": length / max(direct, 1.0e-6),
                "direct": direct, "length": length,
                "direct_soft": float((grid[line_rc[:, 0], line_rc[:, 1]] >= args.soft_class).mean()),
                "direct_blocked": float(blocked[line_rc[:, 0], line_rc[:, 1]].mean()),
                "path_soft": float((grid[path[:, 0], path[:, 1]] >= args.soft_class).mean()),
            })
        if len(solved) >= args.pool:
            break

    if not solved:
        print("[report] no solvable pair found; nothing to draw")
        return 1

    # 按绕行系数排序后分层取样：既要看直达的常见情形，也要看最坏情形站不站得住
    solved.sort(key=lambda s: s["detour"])
    n = args.count
    picks = [solved[min(int(round(q * (len(solved) - 1))), len(solved) - 1)]
             for q in np.linspace(0.0, 1.0, n)]
    # 最直达的一档改挑"直线穿软土最多"的，否则第一格是一条毫无信息的空地直线
    head = max(solved[: max(len(solved) // 4, 1)], key=lambda s: s["direct_soft"])
    picks[0] = head

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 6.3 * rows), squeeze=False)
    for ax, s in zip(axes.flat, picks):
        draw_soil_background(ax, classes, rocks, size, alpha=0.72)
        # 把审计认定的墙（最软土层+岩石，按车半宽膨胀）勾出来，让"绕开"有据可依
        ax.contour(np.arange(grid.shape[1]) * cell, np.arange(grid.shape[0]) * cell,
                   blocked, levels=[0.5], colors="#7a0000", linewidths=1.2, linestyles=":")
        ax.plot([s["start"][0], s["goal"][0]], [s["start"][1], s["goal"][1]], "k--", lw=2.2,
                zorder=5, label=f"straight {s['direct']:.1f} m "
                                f"({s['direct_soft']:.0%} soft, {s['direct_blocked']:.0%} blocked)")
        ax.plot(s["path"][:, 1], s["path"][:, 0], color="#0040ff", lw=3.0, zorder=5,
                label=f"audit route {s['length']:.1f} m "
                      f"(x{s['detour']:.2f}, {s['path_soft']:.0%} soft, {s['length'] / args.speed:.0f}s)")
        ax.plot(*s["start"], marker="o", ms=12, color="white", mec="black", mew=2, zorder=6)
        ax.plot(*s["goal"], marker="*", ms=24, color="#00e000", mec="black", mew=1.5, zorder=6)
        # 视野必须框住整条绕行路线：大绕行的路线会甩到起终点连线之外很远
        pts = np.vstack([s["start"][None, :], s["goal"][None, :], s["path"][:, ::-1]])
        center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        half = max((pts.max(axis=0) - pts.min(axis=0)).max() * 0.5 + args.margin_m, 8.0)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
        ax.set_title(f"env {s['env']} round {s['round']}: detour x{s['detour']:.2f}", fontsize=10)
    for ax in axes.flat[len(picks):]:
        ax.axis("off")

    fig.legend(handles=soil_legend([
        Line2D([], [], color="#7a0000", lw=1.2, ls=":", label="audit wall (P5+rock, inflated 0.6 m)"),
    ]), loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle(f"sampled scenarios judged reachable  —  {os.path.basename(terrain_dir)}  "
                 f"({len(solved)} pairs checked)", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.subplots_adjust(hspace=0.22)
    fig.savefig(args.out, dpi=125, bbox_inches="tight")
    print(f"[report] {args.out}")
    for s in picks:
        print(f"  env {s['env']:3d} round {s['round']}: direct {s['direct']:5.1f}m "
              f"({s['direct_soft']:4.0%} soft, {s['direct_blocked']:4.0%} blocked) -> "
              f"route {s['length']:5.1f}m x{s['detour']:.2f} "
              f"({s['path_soft']:4.0%} soft, {s['length'] / args.speed:3.0f}s)")
    return 0


def cmd_scenario(args):
    """画出演示场景的可绕性依据：直线穿软土，但存在一条走硬地的绕行路线。

    没有这张图，读者无法判断演示是不是挑了个"怎么走都得陷"的场景。
    """
    datasets = [(spec.split("=", 1)[0], np.load(spec.split("=", 1)[1], allow_pickle=True))
                for spec in args.telemetry]
    terrain_dir = str(datasets[0][1]["terrain_dir"])
    size = float(datasets[0][1]["terrain_size"])
    classes, rocks = load_soil_map(terrain_dir)
    env = args.env

    span = episode_segments(datasets[0][1], env)[0]
    start = datasets[0][1]["xy"][span[0], env]
    goal = datasets[0][1]["goal"][span[0], env]
    info = scenario_avoidability(classes, rocks, size, start, goal, args.soft_class)

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.2))
    for ax in axes:
        draw_soil_background(ax, classes, rocks, size, alpha=0.7)
        center = (start + goal) / 2.0
        ax.set_xlim(center[0] - args.span_m, center[0] + args.span_m)
        ax.set_ylim(center[1] - args.span_m, center[1] + args.span_m)
        ax.plot(*start, marker="o", ms=13, color="white", mec="black", mew=2, zorder=6)
        ax.plot(*goal, marker="*", ms=26, color="#00e000", mec="black", mew=1.5, zorder=6)
        ax.set_xlabel("x [m]")

    axes[0].plot([start[0], goal[0]], [start[1], goal[1]], "k--", lw=2.5, zorder=5,
                 label=f"straight line: {info['direct_soft']:.0%} soft soil")
    axes[0].plot(info["path"][:, 1], info["path"][:, 0], color="#0040ff", lw=3, zorder=5,
                 label=f"firm detour: {info['path_soft']:.0%} soft, x{info['detour']:.2f} longer")
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.92)
    axes[0].set_ylabel("y [m]")
    axes[0].set_title(f"env {env}: the soft soil here is avoidable\n"
                      f"direct {info['direct_m']:.1f} m, both endpoints on firm ground")

    # 短轨迹画粗、长轨迹画细并压在上层，否则先陷车的一方会被完全盖住
    traces = [(label, data, episode_segments(data, env)[0]) for label, data in datasets]
    widths = {label: w for (label, _, _), w in
              zip(sorted(traces, key=lambda t: -len(t[2])), (2.4, 3.4, 4.4))}
    for (label, data, seg), color, dash in zip(traces, ("#d62728", "#2ca02c", "#404040"),
                                               ((None, None), (None, None), (5, 2))):
        xy = data["xy"][seg, env]
        axes[1].plot(xy[:, 0], xy[:, 1], lw=widths[label], color=color, zorder=5,
                     dashes=dash if dash[0] else (1, 0),
                     label=f"{CONDITION_LABELS.get(condition_of(label), label)}"
                           f"  ({len(seg) * float(data['step_dt']):.0f}s, {episode_outcome(data, env)})")
        axes[1].plot(xy[-1, 0], xy[-1, 1], marker="X", ms=11, color=color, mec="black", zorder=7)
    axes[1].legend(loc="upper left", fontsize=9, framealpha=0.92)
    axes[1].set_title("what each policy actually did")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[report] {args.out} (direct_soft={info['direct_soft']:.2f}, "
          f"path_soft={info['path_soft']:.2f}, detour=x{info['detour']:.2f})")


def cmd_pick(args):
    """挑一个演示场景：直线要穿软土，但确实存在一条基本走硬地的绕行路线。

    只按地形几何打分，不看策略跑得怎么样——按结果挑就是 cherry-pick。
    """
    datasets = [(spec.split("=", 1)[0], np.load(spec.split("=", 1)[1], allow_pickle=True))
                for spec in args.telemetry]
    terrain_dir = str(datasets[0][1]["terrain_dir"])
    size = float(datasets[0][1]["terrain_size"])
    classes, rocks = load_soil_map(terrain_dir)

    rows = []
    for env in range(datasets[0][1]["xy"].shape[1]):
        spans = [episode_segments(d, env)[0] for _, d in datasets]
        if min(len(s) for s in spans) < args.min_steps:
            continue
        start = datasets[0][1]["xy"][spans[0][0], env]
        goal = datasets[0][1]["goal"][spans[0][0], env]
        info = scenario_avoidability(classes, rocks, size, start, goal, args.soft_class)
        if info is None:
            continue
        firm_ends = max(info["start_class"], info["goal_class"]) < args.soft_class
        # 贴着地图边界的场景，绕行会沿边界走，画面上看不出是在避软土
        inside = float(np.min([start, goal, size - start, size - goal])) >= args.edge_margin
        # 硬地起终点 + 直线穿软土 + 绕行既干净又不太远，四者缺一这个 case 就不成立
        ok = (firm_ends and inside and info["direct_soft"] >= args.min_direct_soft
              and info["path_soft"] <= args.max_path_soft
              and info["detour"] <= args.max_detour)
        info.update(env=env, ok=ok,
                    score=(info["direct_soft"] - info["path_soft"]) / max(info["detour"], 1.0))
        rows.append(info)

    good = [r for r in rows if r["ok"]]
    if args.verbose:
        for r in sorted(rows, key=lambda r: -r["score"])[: args.verbose]:
            print(f"  env {r['env']:3d} start_cls {r['start_class']} goal_cls {r['goal_class']} "
                  f"| direct {r['direct_m']:5.1f}m soft {r['direct_soft']:.2f} "
                  f"| detour x{r['detour']:.2f} soft {r['path_soft']:.2f} "
                  f"| score {r['score']:.2f} {'OK' if r['ok'] else ''}", file=sys.stderr)
        print(f"  {len(good)}/{len(rows)} scenarios are genuinely avoidable", file=sys.stderr)
    if not good:
        print(f"[pick] no avoidable scenario among {len(rows)} candidates; "
              f"relax --max_path_soft/--max_detour or record more envs", file=sys.stderr)
        print(-1)
        return
    print(max(good, key=lambda r: r["score"])["env"])


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("curves", help="learning curves from tensorboard")
    p.add_argument("--run", nargs="+", required=True, metavar="LABEL=DIR")
    p.add_argument("--out", default="docs/softsoil/curves.png")
    p.set_defaults(func=cmd_curves)

    p = sub.add_parser("terrain", help="soil map / elevation / decorrelation")
    p.add_argument("--terrain_dir", required=True)
    p.add_argument("--terrain_size", type=float, default=100.0)
    p.add_argument("--out", default="docs/softsoil/terrain.png")
    p.set_defaults(func=cmd_terrain)

    p = sub.add_parser("tracks", help="trajectories over the soil map")
    p.add_argument("--telemetry", nargs="+", required=True, metavar="LABEL=NPZ")
    p.add_argument("--safe_sinkage", type=float, default=0.02,
                   help="与 env_cfg 的 SAFE_SINKAGE 一致，用于统计超阈时间占比")
    p.add_argument("--out", default="docs/softsoil/tracks.png")
    p.set_defaults(func=cmd_tracks)

    p = sub.add_parser("metrics", help="bar comparison from eval json")
    p.add_argument("--eval", nargs="+", required=True)
    p.add_argument("--terrain_area", nargs=4, type=float, default=None)
    p.add_argument("--out", default="docs/softsoil/metrics.png")
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("combined", help="follow-cam strip stacked on the top-down animation")
    p.add_argument("--telemetry", nargs="+", required=True, metavar="LABEL=NPZ")
    p.add_argument("--cam", nargs="+", required=True, metavar="LABEL=MP4")
    p.add_argument("--env", type=int, default=0)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_combined)

    p = sub.add_parser("stats", help="bootstrap CIs and per-seed spread from eval json")
    p.add_argument("--eval", nargs="+", required=True)
    p.add_argument("--bootstrap", type=int, default=4000)
    p.add_argument("--out", default="/dev/null")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("audit", help="check that sampled goals are actually reachable")
    p.add_argument("--telemetry", nargs="+", required=True, metavar="LABEL=NPZ")
    p.add_argument("--impassable_class", type=int, default=4, help="该类及以上视为进去出不来")
    p.add_argument("--soft_class", type=int, default=3)
    p.add_argument("--robot_radius", type=float, default=0.6)
    p.add_argument("--cell", type=float, default=0.5)
    p.add_argument("--speed", type=float, default=0.29, help="实测有效行进速度 [m/s]")
    p.add_argument("--time_limit", type=float, default=150.0)
    p.add_argument("--tolerance", type=float, default=0.01)
    p.add_argument("--out", default="/dev/null")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("paths", help="draw sampled scenarios with the audit's detour route")
    p.add_argument("--telemetry", required=True, metavar="LABEL=NPZ",
                   help="harvest_spawn_goal.py 产出的采样起终点")
    p.add_argument("--count", type=int, default=6, help="画几个场景（按绕行系数分层取样）")
    p.add_argument("--pool", type=int, default=512, help="最多解算多少对再分层")
    p.add_argument("--impassable_class", type=int, default=4)
    p.add_argument("--soft_class", type=int, default=3)
    p.add_argument("--robot_radius", type=float, default=0.6)
    p.add_argument("--cell", type=float, default=0.5)
    p.add_argument("--speed", type=float, default=0.29)
    p.add_argument("--margin_m", type=float, default=6.0, help="视野在起终点外扩多少米")
    p.add_argument("--out", default="docs/softsoil/reachability.png")
    p.set_defaults(func=cmd_paths)

    p = sub.add_parser("scenario", help="show that the demo scenario's soft soil is avoidable")
    p.add_argument("--telemetry", nargs="+", required=True, metavar="LABEL=NPZ")
    p.add_argument("--env", type=int, required=True)
    p.add_argument("--soft_class", type=int, default=3)
    p.add_argument("--span_m", type=float, default=16.0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_scenario)

    p = sub.add_parser("pick", help="print an env whose soft soil is actually avoidable")
    p.add_argument("--telemetry", nargs="+", required=True, metavar="LABEL=NPZ")
    p.add_argument("--soft_class", type=int, default=3, help="土类 >= 该值视为软土（3=loose）")
    p.add_argument("--min_steps", type=int, default=40)
    p.add_argument("--min_direct_soft", type=float, default=0.35, help="直线至少这么多软土才有绕的必要")
    p.add_argument("--max_path_soft", type=float, default=0.20, help="绕行路线上软土占比上限")
    p.add_argument("--max_detour", type=float, default=1.8, help="绕行长度相对直线的上限")
    p.add_argument("--edge_margin", type=float, default=8.0, help="起终点距地图边界的最小距离 [m]")
    p.add_argument("--verbose", type=int, default=0, help="打印前 N 个候选到 stderr")
    p.add_argument("--out", default="/dev/null")
    p.set_defaults(func=cmd_pick)

    p = sub.add_parser("video", help="top-down comparison animation")
    p.add_argument("--telemetry", nargs="+", required=True, metavar="LABEL=NPZ")
    p.add_argument("--env", type=int, default=0)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--out", default="docs/softsoil/topdown.mp4")
    p.set_defaults(func=cmd_video)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(arguments.out)), exist_ok=True)
    arguments.func(arguments)
