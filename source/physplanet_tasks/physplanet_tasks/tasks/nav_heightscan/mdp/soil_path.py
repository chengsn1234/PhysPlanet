# Dijkstra 最小代价路径进度奖励(全局物理信息 -> 局部观测策略的蒸馏信号).
#
# 动机: v14/v15 的策略拿着 8x8m 局部物理扫描也不会绕开大软区 —— 训练场景软
# patch 中位弦长 ~6m, 欧氏进度奖励(+2.0)下"直穿"永远比"绕行"划算, 避软行为
# 从未被奖励过. 本模块在训练侧引入全局信息:
#   - 以 n0 场构造通行代价图(硬=1, 软按 n0 三次方恶化), Dijkstra 求 spawn->goal
#     的最小代价路径;
#   - 奖励 = 沿该路径剩余弧长的下降量(势形塑形): 贴着代价图给出的好路线走才有
#     进度分, 直插软区不得分且吃沉陷/滑转惩罚;
#   - 策略观测不变(仍是局部扫描), 部署时不需要任何规划器 —— 规划器的"选路知识"
#     在训练中被蒸馏进策略权重.
# 代价曲线标定(alpha=86, n_ref=1.0, p=3): n0=0.92->1.01, 1.11(sand)->1.06,
# 1.18(S1 硬沙)->1.50(可穿), 1.40(坑缘)->6.5, 1.60(loose)->19.6(必绕).

import heapq
import math
import os

import numpy as np
import torch

from ..terramechanics import SOIL_KEYS

__all__ = ["soil_path_progress_reward", "soft_proximity_penalty"]

_PATH_SPACING_M = 1.5     # 折线简化间距
_TRACK_WINDOW = 12        # 最近段跟踪窗口(±点数)


class _SoilCostPlanner:
    """n0 场 -> 1m 格网通行代价 -> 8 邻域 Dijkstra."""

    def __init__(self, terrain_dir: str, terrain_size: float,
                 alpha: float, n_ref: float, pool: int = 10,
                 inflate_m: float = 2.0):
        soil = np.load(os.path.join(terrain_dir, "soil_params_map.npy"))
        n0 = soil[:, :, SOIL_KEYS.index("n0")].astype(np.float64)
        soft = np.clip(n0 - n_ref, 0.0, None)
        if inflate_m > 0:
            # 软区膨胀: 先在高分辨率上 max 膨胀 inflate_m, 再 max 池化到规划格网.
            # 不能先均值池化再膨胀 —— 坑缘外是硬类(如基床 0.6), 均值会把 4m 温和带
            # 边缘稀释到软度 0, 膨胀就没东西可扩; max 保序, 晕圈不会被稀释掉.
            px_res = terrain_size / soft.shape[1]
            it = int(math.ceil(inflate_m / px_res))
            for _ in range(it):
                p = np.pad(soft, 1, mode="edge")
                # 注意 np.maximum 只支持二元(第三位置参数是 out), 必须链式两两取大
                m = np.maximum(np.maximum(p[:-2, :-2], p[:-2, 1:-1]), p[:-2, 2:])
                m = np.maximum(m, np.maximum(
                    np.maximum(p[1:-1, :-2], p[1:-1, 1:-1]), p[1:-1, 2:]))
                m = np.maximum(m, np.maximum(
                    np.maximum(p[2:, :-2], p[2:, 1:-1]), p[2:, 2:]))
                soft = m
        g = self._max_pool(soft, pool)
        self.cost = 1.0 + alpha * g ** 3
        self.ny, self.nx = g.shape
        self.res = terrain_size / self.nx
        self.origin = 0.0   # 训练侧地图角对齐: 世界 [0, size], 像素0在(0,0)

    @staticmethod
    def _max_pool(a: np.ndarray, pool: int) -> np.ndarray:
        h = (a.shape[0] // pool) * pool
        w = (a.shape[1] // pool) * pool
        return a[:h, :w].reshape(h // pool, pool, w // pool, pool).max(axis=(1, 3))

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        c = int(np.clip((x - self.origin) / self.res, 0, self.nx - 1))
        r = int(np.clip((y - self.origin) / self.res, 0, self.ny - 1))
        return r, c

    def _world(self, r: int, c: int) -> tuple[float, float]:
        return (self.origin + (c + 0.5) * self.res,
                self.origin + (r + 0.5) * self.res)

    def plan(self, start_xy, goal_xy):
        """返回 (折线 (K,2) world xy, 累计弧长 (K,)). 目标不可达时退化为直线."""
        sr, sc = self._cell(start_xy[0], start_xy[1])
        gr, gc = self._cell(goal_xy[0], goal_xy[1])
        ny, nx = self.ny, self.nx
        cost = self.cost
        s, t = sr * nx + sc, gr * nx + gc
        if s == t:
            poly = np.array([start_xy, goal_xy], dtype=np.float64)
            L = float(math.hypot(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1]))
            return poly, np.array([0.0, L])
        dist = np.full(ny * nx, np.inf)
        prev = np.full(ny * nx, -1, dtype=np.int64)
        settled = np.zeros(ny * nx, dtype=bool)
        dist[s] = 0.0
        heap = [(0.0, s)]
        nbrs = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
                (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)))
        while heap:
            d, u = heapq.heappop(heap)
            if settled[u]:
                continue
            if u == t:
                break
            settled[u] = True
            ur, uc = divmod(u, nx)
            cu = cost[ur, uc]
            for dr, dc, seg in nbrs:
                vr, vc = ur + dr, uc + dc
                if not (0 <= vr < ny and 0 <= vc < nx):
                    continue
                v = vr * nx + vc
                if settled[v]:
                    continue
                nd = d + 0.5 * (cu + cost[vr, vc]) * seg
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        if not np.isfinite(dist[t]):
            poly = np.array([start_xy, goal_xy], dtype=np.float64)
            return poly, np.array([0.0, float(math.hypot(goal_xy[0] - start_xy[0],
                                                         goal_xy[1] - start_xy[1]))])
        cells = []
        u = t
        while u != -1:
            cells.append(u)
            u = prev[u]
        cells.reverse()
        pts = np.array([self._world(u // nx, u % nx) for u in cells])
        keep = [0]
        for i in range(1, len(pts) - 1):
            if math.hypot(pts[i][0] - pts[keep[-1]][0],
                          pts[i][1] - pts[keep[-1]][1]) >= _PATH_SPACING_M:
                keep.append(i)
        keep.append(len(pts) - 1)
        poly = pts[keep]
        seg = np.hypot(np.diff(poly[:, 0]), np.diff(poly[:, 1]))
        return poly, np.concatenate([[0.0], np.cumsum(seg)])


def _nearest_on_polyline(p, poly, start_idx: int, full: bool = False):
    """返回 (段索引, 到折线距离); 窗口限制在 start_idx±_TRACK_WINDOW."""
    n = len(poly)
    lo, hi = 0, n - 1
    if not full:
        lo = max(0, start_idx - _TRACK_WINDOW)
        hi = min(n - 1, start_idx + _TRACK_WINDOW)
    bi, bd = lo, float("inf")
    for i in range(lo, hi):
        ax, ay = poly[i]
        abx, aby = poly[i + 1][0] - ax, poly[i + 1][1] - ay
        L2 = abx * abx + aby * aby + 1e-12
        t = ((p[0] - ax) * abx + (p[1] - ay) * aby) / L2
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        dx = p[0] - (ax + t * abx)
        dy = p[1] - (ay + t * aby)
        d = dx * dx + dy * dy
        if d < bd:
            bi, bd = i, d
    return bi, math.sqrt(bd)


def _remaining(entry, p) -> float:
    i0 = min(entry["idx"], len(entry["cum"]) - 2)
    ax, ay = entry["poly"][i0]
    bx, by = entry["poly"][i0 + 1]
    abx, aby = bx - ax, by - ay
    L2 = abx * abx + aby * aby + 1e-12
    t = ((p[0] - ax) * abx + (p[1] - ay) * aby) / L2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return float(entry["cum"][-1] - entry["cum"][i0] - t * math.hypot(abx, aby))


def soil_path_progress_reward(env, command_name: str, terrain_dir: str,
                              terrain_size: float, alpha: float = 86.0,
                              n_ref: float = 1.0, replan_dist: float = 3.0,
                              inflate_m: float = 2.0) -> torch.Tensor:
    """沿 Dijkstra 最小代价路径的剩余弧长下降量 (m/step).

    每环境维护 {目标, 折线, 累计弧长, 跟踪段, 上步剩余}: 目标变更(reset)或偏离
    折线超 replan_dist 时从当前位置重规划; 重规划步奖励置 0, 不产生跳变.
    """
    state = getattr(env, "_soil_path_state", None)
    if state is None or state["terrain_dir"] != terrain_dir:
        state = {"planner": _SoilCostPlanner(terrain_dir, terrain_size, alpha, n_ref,
                                             inflate_m=inflate_m),
                 "cache": [None] * env.num_envs,
                 "terrain_dir": terrain_dir}
        env._soil_path_state = state
    planner, cache = state["planner"], state["cache"]

    cmd = env.command_manager.get_term(command_name)
    goal_w = cmd.pos_command_w[:, :2].cpu().numpy()
    pos_w = env.scene["robot"].data.root_pos_w[:, :2].cpu().numpy()

    reward = np.zeros(env.num_envs, dtype=np.float64)
    for i in range(env.num_envs):
        p, g = pos_w[i], goal_w[i]
        entry = cache[i]
        replanned = False
        if (entry is None or env.episode_length_buf[i] <= 1
                or math.hypot(g[0] - entry["goal"][0], g[1] - entry["goal"][1]) > 0.5):
            poly, cum = planner.plan(p, g)
            idx, _ = _nearest_on_polyline(p, poly, 0, full=True)
            entry = {"goal": (float(g[0]), float(g[1])), "poly": poly,
                     "cum": cum, "idx": idx, "prev": 0.0}
            cache[i] = entry
            replanned = True
        else:
            idx, d = _nearest_on_polyline(p, entry["poly"], entry["idx"])
            entry["idx"] = idx
            if d > replan_dist:
                poly, cum = planner.plan(p, g)
                idx, _ = _nearest_on_polyline(p, poly, 0, full=True)
                entry["poly"], entry["cum"] = poly, cum
                entry["idx"] = idx
                replanned = True
        rem = _remaining(entry, p)
        if not replanned:
            reward[i] = entry["prev"] - rem
        entry["prev"] = rem
    return torch.as_tensor(reward, dtype=torch.float32, device=env.device)


def soft_proximity_penalty(env, terrain_dir: str, terrain_size: float,
                           margin_m: float = 3.5, weight_scale: float = 0.5,
                           n_soft: float = 1.30, pool: int = 10) -> torch.Tensor:
    """软区邻近惩罚: 距 n0>=n_soft 区域越近扣分越多(线性, margin_m 内 0->weight_scale).

    与路径进度不同, 这是策略"看得见"的信号(物理扫描里就有软区) —— 教"离软区
    保持 margin"的习惯, 不依赖不可见的规划路. S1(n0=1.18) 在阈值外, 照常穿越.
    """
    state = getattr(env, "_soft_prox_state", None)
    if state is None or state["terrain_dir"] != terrain_dir:
        soil = np.load(os.path.join(terrain_dir, "soil_params_map.npy"))
        n0 = soil[:, :, SOIL_KEYS.index("n0")].astype(np.float64)
        px = terrain_size / n0.shape[1]
        mask = n0 >= n_soft
        covered = mask.copy()
        lvl = np.zeros(mask.shape, dtype=np.float32)   # 到软区分级距离 (m)
        step_px = max(1, int(round(0.5 / px)))         # 每级 0.5m(棋盘膨胀主方向即 1px/迭代)
        levels = int(math.ceil(margin_m / 0.5))
        cur = mask.copy()
        for i in range(1, levels + 1):
            for _ in range(step_px):
                p = np.pad(cur, 1, mode="edge")
                cur = (p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:] |
                       p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:] |
                       p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:])
            lvl[cur & ~covered] = i * 0.5
            covered = cur
        h = (lvl.shape[0] // pool) * pool
        w = (lvl.shape[1] // pool) * pool
        d = lvl[:h, :w].reshape(h // pool, pool, w // pool, pool).max(axis=(1, 3))
        state = {"d": d, "terrain_dir": terrain_dir,
                 "res": terrain_size / d.shape[1], "origin": 0.0}
        env._soft_prox_state = state
    st = state
    pos = env.scene["robot"].data.root_pos_w[:, :2].cpu().numpy()
    nx, ny = st["d"].shape[1], st["d"].shape[0]
    gx = np.clip((pos[:, 0] - st["origin"]) / st["res"], 0, nx - 1.001)
    gy = np.clip((pos[:, 1] - st["origin"]) / st["res"], 0, ny - 1.001)
    x0 = gx.astype(int)
    y0 = gy.astype(int)
    fx, fy = gx - x0, gy - y0
    dval = (st["d"][y0, x0] * (1 - fx) * (1 - fy) + st["d"][y0, x0 + 1] * fx * (1 - fy)
            + st["d"][y0 + 1, x0] * (1 - fx) * fy + st["d"][y0 + 1, x0 + 1] * fx * fy)
    pen = weight_scale * np.clip(1.0 - dval / margin_m, 0.0, 1.0)
    return torch.as_tensor(pen, dtype=torch.float32, device=env.device)
