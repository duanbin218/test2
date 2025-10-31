"""环带巡边寻路核心实现模块。

本模块提供构建矩形环带掩膜、在栅格地图内寻找最近环带点、对环带点按极角排序、
执行八方向 A* 寻路、以及生成完整绕圈路径与绘制函数。所有函数都围绕
二维网格坐标 (x, y)（x 为列、y 为行）展开。
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2
except ImportError as exc:
    raise ImportError("绘制路径需要安装 OpenCV(cv2)，请先安装依赖。") from exc
import numpy as np

Coordinate = Tuple[int, int]


def build_ring_mask(
    ax: int,
    ay: int,
    inner_r: int,
    outer_r: int,
    width: int,
    height: int,
) -> np.ndarray:
    """根据切比雪夫距离构建矩形环带布尔掩膜。

    参数说明：
        ax, ay: 环带中心点 A 的坐标，x 为列、y 为行。
        inner_r: 内边框半径，必须为非负整数。
        outer_r: 外边框半径，必须大于 inner_r。
        width, height: 栅格图像的宽与高，用于限定掩膜大小。

    返回值：
        与图像同尺寸的布尔数组，True 表示位于环带区域内。
    """

    if not (0 <= ax < width and 0 <= ay < height):
        raise ValueError("中心点 A 超出图像范围，无法构建环带掩膜。")
    if inner_r < 0 or outer_r <= inner_r:
        raise ValueError("环带半径参数非法，需满足 outer_r > inner_r >= 0。")

    # 使用 numpy 构造所有坐标与中心点的差值，便于一次性计算切比雪夫距离。
    yy, xx = np.ogrid[:height, :width]
    chebyshev = np.maximum(np.abs(xx - ax), np.abs(yy - ay))
    ring_mask = (chebyshev > inner_r) & (chebyshev <= outer_r)
    return ring_mask


def _iter_neighbors() -> Iterable[Tuple[int, int, float]]:
    """返回八方向邻居及其代价的迭代器。"""

    # 八方向移动：水平和垂直的代价为 1，斜向代价为 sqrt(2)。
    for dx, dy in [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ]:
        cost = math.sqrt(2) if dx != 0 and dy != 0 else 1.0
        yield dx, dy, cost


def _can_step(
    walkable_mask: np.ndarray,
    x: int,
    y: int,
    nx: int,
    ny: int,
) -> bool:
    """判断从 (x, y) 到 (nx, ny) 的移动是否合法。"""

    height, width = walkable_mask.shape
    if not (0 <= nx < width and 0 <= ny < height):
        return False
    if not walkable_mask[ny, nx]:
        return False
    # 处理斜向移动的“角落穿越”情况：若斜向移动，两条直角边都需可走。
    if nx != x and ny != y:
        if not (walkable_mask[y, nx] and walkable_mask[ny, x]):
            return False
    return True


def find_nearest_point_on_ring(
    a: Coordinate,
    ring_mask: np.ndarray,
    walkable_mask: np.ndarray,
) -> Coordinate:
    """从起点 A 出发，找到最近的环带内可走点。

    实现方式：使用 Dijkstra（优先队列）在全局可走网格上扩张，
    第一次弹出位于环带且可走的点即为最近点。
    """

    ay, ax = a[1], a[0]
    height, width = walkable_mask.shape
    if not (0 <= ax < width and 0 <= ay < height):
        raise ValueError("起点 A 超出地图范围，无法寻路。")
    if not walkable_mask[ay, ax]:
        raise ValueError("起点 A 所在像素不可走，无法启动寻路。")

    visited = set()
    heap: List[Tuple[float, Coordinate]] = [(0.0, (ax, ay))]
    distances: Dict[Coordinate, float] = {(ax, ay): 0.0}

    while heap:
        dist, (cx, cy) = heapq.heappop(heap)
        if (cx, cy) in visited:
            continue
        visited.add((cx, cy))
        if ring_mask[cy, cx] and walkable_mask[cy, cx]:
            return (cx, cy)

        for dx, dy, cost in _iter_neighbors():
            nx, ny = cx + dx, cy + dy
            if not _can_step(walkable_mask, cx, cy, nx, ny):
                continue
            new_dist = dist + cost
            if new_dist < distances.get((nx, ny), float("inf")):
                distances[(nx, ny)] = new_dist
                heapq.heappush(heap, (new_dist, (nx, ny)))

    raise ValueError("未能找到环带内的可走点，请检查环带参数或障碍布局。")


def order_ring_points_cw_or_ccw(
    points: Sequence[Coordinate],
    a: Coordinate,
    clockwise: bool = True,
) -> List[Coordinate]:
    """根据极角对环带候选点进行顺/逆时针排序。"""

    if not points:
        return []
    ax, ay = a

    def _angle(pt: Coordinate) -> float:
        # atan2 的参数顺序为 (y, x)，这里统一转换到 0-2π 范围，便于排序。
        raw = math.atan2(pt[1] - ay, pt[0] - ax)
        angle = raw if raw >= 0 else raw + 2 * math.pi
        return angle

    with_angles = [(pt, _angle(pt)) for pt in points]
    with_angles.sort(key=lambda item: item[1], reverse=clockwise)
    return [pt for pt, _ in with_angles]


def _heuristic(a: Coordinate, b: Coordinate) -> float:
    """八方向距离（Octile Distance）启发式函数。"""

    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    diag = min(dx, dy)
    straight = max(dx, dy) - diag
    return math.sqrt(2) * diag + straight


def astar(
    walkable_mask: np.ndarray,
    start_xy: Coordinate,
    goal_xy: Coordinate,
) -> List[Coordinate]:
    """在布尔可走掩膜上执行八方向 A* 寻路。

    若起点或终点不可达，会抛出 ValueError，并提供中文提示。
    """

    height, width = walkable_mask.shape
    sx, sy = start_xy
    gx, gy = goal_xy
    if not (0 <= sx < width and 0 <= sy < height):
        raise ValueError("起点超出地图范围，无法执行 A* 寻路。")
    if not (0 <= gx < width and 0 <= gy < height):
        raise ValueError("终点超出地图范围，无法执行 A* 寻路。")
    if not walkable_mask[sy, sx]:
        raise ValueError("起点处不可走，无法执行 A* 寻路。")
    if not walkable_mask[gy, gx]:
        raise ValueError("终点处不可走，无法执行 A* 寻路。")

    open_heap: List[Tuple[float, Coordinate]] = []
    heapq.heappush(open_heap, (0.0, start_xy))
    came_from: Dict[Coordinate, Optional[Coordinate]] = {start_xy: None}
    g_score: Dict[Coordinate, float] = {start_xy: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal_xy:
            # 回溯路径
            path: List[Coordinate] = []
            node: Optional[Coordinate] = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        cx, cy = current
        for dx, dy, move_cost in _iter_neighbors():
            nx, ny = cx + dx, cy + dy
            if not _can_step(walkable_mask, cx, cy, nx, ny):
                continue
            tentative_g = g_score[current] + move_cost
            if tentative_g >= g_score.get((nx, ny), float("inf")):
                continue
            came_from[(nx, ny)] = current
            g_score[(nx, ny)] = tentative_g
            f_score = tentative_g + _heuristic((nx, ny), goal_xy)
            heapq.heappush(open_heap, (f_score, (nx, ny)))

    raise ValueError("A* 寻路失败，可能是起终点被障碍完全隔离。")


def trace_ring_path(
    a: Coordinate,
    ring_mask: np.ndarray,
    walkable_mask: np.ndarray,
    clockwise: bool = True,
    anchor_count: int = 8,
) -> List[Coordinate]:
    """生成环绕 A 点的稀疏巡边路径点。

    本函数在内部仍使用 A* 连接各个锚点，并在 `trace_ring_path.last_metadata`
    中记录完整的密集路径。但最终返回值只包含指定数量的环带锚点，
    以满足“仅需要上下左右/对角方向几个点”的需求。

    参数说明：
        a: 起点 A 的坐标，格式为 (x, y)。
        ring_mask: 由 `build_ring_mask` 构建的环带布尔掩膜。
        walkable_mask: 全局可走掩膜，True 表示可通行。
        clockwise: 是否按顺时针排序锚点，False 则为逆时针。
        anchor_count: 希望返回的环带锚点数量，仅支持 4 或 8。

    返回值：
        指定数量的环带锚点列表，不含起点 A，本函数的密集行走路径请从
        `trace_ring_path.last_metadata["密集路径"]` 获取，完整锚点列表亦同步存于
        `trace_ring_path.last_metadata["锚点列表"]` 便于外部复用。
    """

    height, width = walkable_mask.shape
    if ring_mask.shape != walkable_mask.shape:
        raise ValueError("环带掩膜与可走掩膜尺寸不一致，无法巡边。")
    if not (0 <= a[0] < width and 0 <= a[1] < height):
        raise ValueError("起点 A 超出地图范围，无法巡边。")
    if anchor_count not in (4, 8):
        raise ValueError("锚点数量仅支持 4 或 8，请根据需求选择上下左右或包含对角线。")

    # 收集环带可走点。
    ring_points = np.argwhere(ring_mask & walkable_mask)
    ring_points = [(int(x), int(y)) for y, x in ring_points]
    if not ring_points:
        raise ValueError("环带区域内不存在可走像素，无法执行巡边。")
    if len(ring_points) < anchor_count:
        raise ValueError("环带可走像素数量不足，无法选出足够的锚点。")

    nearest = find_nearest_point_on_ring(a, ring_mask, walkable_mask)

    ax, ay = a

    def _angle(pt: Coordinate) -> float:
        """将点的极角转换到 [0, 2π) 区间，便于比较。"""

        raw = math.atan2(pt[1] - ay, pt[0] - ax)
        return raw if raw >= 0 else raw + 2 * math.pi

    def _angle_diff(a1: float, a2: float) -> float:
        """计算两个角度的最小差值。"""

        diff = abs(a1 - a2) % (2 * math.pi)
        return min(diff, 2 * math.pi - diff)

    points_with_angle = [(pt, _angle(pt)) for pt in ring_points]
    # 先按逆时针方向（角度递增）排序，后续再根据 clockwise 调整顺序。
    points_with_angle.sort(key=lambda item: item[1])

    if anchor_count == 4:
        target_angles = [0.0, 0.5 * math.pi, 1.0 * math.pi, 1.5 * math.pi]
    else:
        target_angles = [
            0.0,
            0.25 * math.pi,
            0.5 * math.pi,
            0.75 * math.pi,
            1.0 * math.pi,
            1.25 * math.pi,
            1.5 * math.pi,
            1.75 * math.pi,
        ]

    used_indices: set[int] = set()
    selected: List[Tuple[float, Coordinate]] = []
    for target in target_angles:
        best_idx: Optional[int] = None
        best_diff = float("inf")
        best_pt: Optional[Coordinate] = None
        for idx, (pt, ang) in enumerate(points_with_angle):
            if idx in used_indices:
                continue
            diff = _angle_diff(ang, target)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx
                best_pt = pt
        if best_idx is None or best_pt is None:
            raise ValueError("未能为全部方向选出环带锚点，请检查障碍布局。")
        used_indices.add(best_idx)
        selected.append((points_with_angle[best_idx][1], best_pt))

    # selected 按照 target_angles（逆时针）排列，如需顺时针则反转。
    if clockwise:
        selected = list(reversed(selected))

    # 根据起点到锚点的启发式距离选择第一个锚点，使行走更自然。
    if selected:
        start_idx = min(
            range(len(selected)),
            key=lambda i: _heuristic(a, selected[i][1]),
        )
        selected = selected[start_idx:] + selected[:start_idx]

    anchor_points: List[Coordinate] = [pt for _, pt in selected]

    global_path: List[Coordinate] = []
    segment_count = 0
    fallback_count = 0
    skipped_points: List[Coordinate] = []

    # 第一个段落：从 A 到环带最近点，使用全局可走掩膜。
    try:
        first_segment = astar(walkable_mask, a, nearest)
    except ValueError as exc:
        raise ValueError(f"A 点无法到达环带：{exc}") from exc
    global_path.extend(first_segment)
    segment_count += 1

    current = nearest
    local_mask = ring_mask & walkable_mask

    # 若第一个锚点与最近点不同，先连到第一个锚点。
    if anchor_points:
        first_anchor = anchor_points[0]
        if first_anchor != current:
            try:
                segment = astar(local_mask, current, first_anchor)
            except ValueError:
                try:
                    segment = astar(walkable_mask, current, first_anchor)
                    fallback_count += 1
                except ValueError:
                    skipped_points.append(first_anchor)
                    segment = []
            if segment:
                global_path.extend(segment[1:])
                current = first_anchor
                segment_count += 1

    # 依次连接其余锚点。
    for target in anchor_points[1:]:
        if target == current:
            continue
        try:
            segment = astar(local_mask, current, target)
        except ValueError:
            try:
                segment = astar(walkable_mask, current, target)
                fallback_count += 1
            except ValueError:
                skipped_points.append(target)
                continue
        # 拼接路径时去掉第一个点，避免重复。
        global_path.extend(segment[1:])
        current = target
        segment_count += 1

    # 回到第一个锚点以闭合环带。
    if anchor_points:
        first_anchor = anchor_points[0]
        if current != first_anchor:
            try:
                segment = astar(local_mask, current, first_anchor)
            except ValueError:
                try:
                    segment = astar(walkable_mask, current, first_anchor)
                    fallback_count += 1
                except ValueError:
                    skipped_points.append(first_anchor)
                    segment = []
            if segment:
                global_path.extend(segment[1:])
                current = first_anchor
                segment_count += 1

    # 最后从首个锚点返回 A，形成完整闭环。
    if current != a:
        try:
            segment = astar(walkable_mask, current, a)
        except ValueError:
            skipped_points.append(a)
        else:
            global_path.extend(segment[1:])
            current = a
            segment_count += 1

    trace_ring_path.last_metadata = {
        "环带候选点数": len(ring_points),
        "锚点数量": len(anchor_points),
        "成功段数": segment_count,
        "回退次数": fallback_count,
        "跳过点数": len(skipped_points),
        "跳过点列表": skipped_points,
        "最近环带点": nearest,
        "是否闭合": (global_path[-1] == a) if global_path else False,
        "总步长": len(global_path),
        "密集路径": global_path,
        "锚点列表": anchor_points,
    }

    if len(global_path) < 2:
        raise ValueError("巡边路径过短，可能没有成功绕行。")

    return anchor_points


def draw_path_on_image(
    bgr_img: np.ndarray,
    path: Sequence[Coordinate],
    out_path: str,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> None:
    """在原始图像上绘制巡边路径并保存。

    `color` 使用 BGR 颜色顺序。函数会额外标记起点与终点。
    建议传入 `trace_ring_path.last_metadata["密集路径"]` 这样的密集坐标序列。
    """

    if bgr_img.ndim != 3 or bgr_img.shape[2] != 3:
        raise ValueError("绘图函数仅支持三通道 BGR 图像。")
    if len(path) < 2:
        raise ValueError("路径点数量不足，无法绘制。")

    canvas = bgr_img.copy()
    pts = np.array([(x, y) for x, y in path], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, color, thickness, lineType=cv2.LINE_AA)

    start_pt = tuple(map(int, path[0]))
    end_pt = tuple(map(int, path[-1]))
    cv2.circle(canvas, start_pt, 3, (0, 255, 0), -1)
    cv2.putText(canvas, "A", (start_pt[0] + 2, start_pt[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.circle(canvas, end_pt, 3, (255, 0, 0), -1)

    cv2.imwrite(out_path, canvas)


# 为静态类型工具设置默认的元数据字典，防止属性不存在。
trace_ring_path.last_metadata = {
    "环带候选点数": 0,
    "锚点数量": 0,
    "成功段数": 0,
    "回退次数": 0,
    "跳过点数": 0,
    "跳过点列表": [],
    "最近环带点": (0, 0),
    "是否闭合": False,
    "总步长": 0,
    "密集路径": [],
    "锚点列表": [],
}
