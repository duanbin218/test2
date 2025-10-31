"""环带寻路核心实现模块。

此模块严格按照任务要求提供图像网格转换、正方形环带轨迹生成、八方向 A* 搜索、
最近可达点查找、完整路径组装与路径绘制等功能。全部说明均使用中文，
以便在中文环境下维护与扩展。
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np

# 为了便于类型检查，提前定义常用类型别名
Point = Tuple[int, int]


def image_to_grid(img: np.ndarray) -> np.ndarray:
    """将三通道图像转换为布尔网格。

    参数说明：
        img: 输入的三通道图像，需满足纯黑(0,0,0)表示可走，纯白(255,255,255)表示不可走，
             其他颜色一律视为不可走。

    返回：
        布尔网格，True 表示可走，False 表示不可走。

    设计原因：
        直接在布尔网格上进行 A* 搜索能大幅简化逻辑，并且 NumPy 布尔数组内存紧凑、
        运算速度快。函数中仅检查纯黑像素，其余情况全部置为不可走以避免路径穿越
        未知区域。
    """

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("输入图像必须是三通道彩色图像")

    # 构建可走区域：完全等于 0 的像素
    walkable = np.all(img == 0, axis=2)
    return walkable


def _validate_radii(r_outer: int, r_inner: int) -> None:
    """检查外半径与内半径是否合法，不合法时抛出中文异常。"""

    if r_outer <= 0 or r_inner <= 0:
        raise ValueError("外半径与内半径都必须为正整数")
    if r_outer <= r_inner:
        raise ValueError("外半径必须严格大于内半径")


def _is_within(x: int, y: int, w: int, h: int) -> bool:
    """判断像素坐标是否在图像范围内。"""

    return 0 <= x < w and 0 <= y < h


def rect_track_points(
    ax: int,
    ay: int,
    r_outer: int,
    r_inner: int,
    cw: bool,
    img_w: int,
    img_h: int,
    grid: np.ndarray,
) -> List[Point]:
    """生成环带中线轨迹上的可行走点序列。

    参数说明：
        ax, ay: 环带中心点 A 的坐标。
        r_outer, r_inner: 环带外半径与内半径，要求外半径大于内半径。
        cw: 若为 True 表示顺时针遍历轨迹，False 表示逆时针。
        img_w, img_h: 图像宽高，用于过滤越界点。
        grid: 布尔可走网格，用来筛除不可走点。

    返回：
        一个按照环行顺序排列的点列表，所有点均在图像范围内且可走。

    设计说明：
        为了确保环行轨迹尽可能居中，按照要求选取 r_track = floor((r_outer + r_inner)/2)
        生成“中线正方形”。该正方形周长路径即为理想的绕行轨迹。当轨迹上的某些
        像素越界或被障碍阻挡时，将其剔除，但仍保留原有顺序，便于后续在断点处尝试
        使用 A* 补齐连接；若最终无法形成完整闭环，则在路径构建阶段抛出异常。
    """

    _validate_radii(r_outer, r_inner)

    r_track = (r_outer + r_inner) // 2
    if r_track <= 0:
        raise ValueError("环行轨迹半径计算结果不合法")

    # 若中线半径为 0，说明外半径与内半径十分接近，需要提醒用户调整
    if r_track == 0:
        raise ValueError("环行轨迹半径为 0，请调大外半径或内半径差值")

    # 构造正方形四个边界范围
    left = ax - r_track
    right = ax + r_track
    top = ay - r_track
    bottom = ay + r_track

    edges: List[Point] = []

    def add_segment(points: Iterable[Point]) -> None:
        """辅助函数：按顺序将点加入列表并过滤越界/不可走点。"""

        for x, y in points:
            if not _is_within(x, y, img_w, img_h):
                continue
            if not grid[y, x]:
                continue
            edges.append((x, y))

    # 根据顺时针或逆时针分别生成边界点顺序
    if cw:
        # 顺时针：上边从左到右，右边从上到下，下边从右到左，左边从下到上
        add_segment((x, top) for x in range(left, right + 1))
        add_segment((right, y) for y in range(top + 1, bottom + 1))
        add_segment((x, bottom) for x in range(right - 1, left - 1, -1))
        add_segment((left, y) for y in range(bottom - 1, top, -1))
    else:
        # 逆时针：上边从右到左，左边从上到下，下边从左到右，右边从下到上
        add_segment((x, top) for x in range(right, left - 1, -1))
        add_segment((left, y) for y in range(top + 1, bottom + 1))
        add_segment((x, bottom) for x in range(left + 1, right + 1))
        add_segment((right, y) for y in range(bottom - 1, top, -1))

    # 若轨迹完全被剔除，直接报错提示用户调整参数或清理障碍
    if not edges:
        raise ValueError("环行轨迹在图像范围内没有任何可走点，请调整半径或清理障碍")

    # 消除连续重复点，避免后续环路中出现停顿
    cleaned: List[Point] = []
    for pt in edges:
        if not cleaned or cleaned[-1] != pt:
            cleaned.append(pt)

    # 若最后一个点与第一个点相同也去掉，后续会显式闭环
    if cleaned and cleaned[0] == cleaned[-1]:
        cleaned.pop()

    return cleaned


# 定义八方向移动及其代价
_NEIGHBORS: List[Tuple[int, int, float]] = [
    (-1, -1, math.sqrt(2)),
    (0, -1, 1.0),
    (1, -1, math.sqrt(2)),
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (-1, 1, math.sqrt(2)),
    (0, 1, 1.0),
    (1, 1, math.sqrt(2)),
]


def _heuristic(a: Point, b: Point) -> float:
    """八方向一致的启发函数：采用 Octile Distance（八方向距离）。"""

    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    # octile = dx + dy + (sqrt(2) - 2) * min(dx, dy)
    return dx + dy + (math.sqrt(2) - 2) * min(dx, dy)


def a_star(grid: np.ndarray, start: Point, goal: Point) -> List[Point]:
    """在布尔网格上执行八方向 A* 搜索。

    参数说明：
        grid: 布尔可走网格。
        start: 起点坐标。
        goal: 终点坐标。

    返回：
        若可达，返回包含起点与终点的路径列表；若不可达，返回空列表。

    设计说明：
        使用优先队列保存候选节点，启发函数采用八方向距离（Octile Distance），
        这样能够在八邻接网格中保持一致性与可接受的收敛速度。由于移动代价存在直线与
        对角线两种情况，因此使用 Dijkstra 兼容的加权最短路框架。
    """

    if not grid[start[1], start[0]] or not grid[goal[1], goal[0]]:
        return []

    h = _heuristic(start, goal)
    open_heap: List[Tuple[float, Point]] = [(h, start)]
    came_from: Dict[Point, Optional[Point]] = {start: None}
    g_score: Dict[Point, float] = {start: 0.0}

    img_h, img_w = grid.shape

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            # 回溯构建路径
            path: List[Point] = []
            node: Optional[Point] = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        current_g = g_score[current]

        for dx, dy, cost in _NEIGHBORS:
            nx = current[0] + dx
            ny = current[1] + dy
            if not _is_within(nx, ny, img_w, img_h):
                continue
            if not grid[ny, nx]:
                continue

            tentative_g = current_g + cost
            neighbor = (nx, ny)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_score = tentative_g + _heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))

    return []


def find_nearest_reachable_on_ring(
    grid: np.ndarray,
    start: Point,
    ring_points: List[Point],
) -> Point:
    """在环带轨迹上寻找对起点可达的最短代价点。

    参数说明：
        grid: 布尔可走网格。
        start: 起点 A 的坐标。
        ring_points: 环带轨迹上按顺序排列的点列表。

    返回：
        与 A 最近且可达的环带轨迹点。

    设计说明：
        为避免对每个候选点分别执行 A* 导致性能开销巨大，采用一次性 Dijkstra
        （在代价相同的情况下等价于多目标 BFS），自起点向外扩散，当首次弹出
        属于环带集合的节点时即获得最短实际代价点。这种策略符合任务说明中
        “建议从 A 做一次 BFS” 的建议。
    """

    if not ring_points:
        raise ValueError("环行轨迹为空，无法寻找最近点")

    img_h, img_w = grid.shape
    if not _is_within(start[0], start[1], img_w, img_h):
        raise ValueError("A 点越界，请检查输入")
    if not grid[start[1], start[0]]:
        raise ValueError("A 点所在像素不可走，请选择可走位置")

    ring_set: Set[Point] = set(ring_points)

    # 使用优先队列实现 Dijkstra，代价与 a_star 中一致
    heap: List[Tuple[float, Point]] = [(0.0, start)]
    visited: Dict[Point, float] = {start: 0.0}

    while heap:
        cost, current = heapq.heappop(heap)

        if current in ring_set:
            return current

        # 若取出的节点已经有更优代价，则跳过
        if cost > visited.get(current, float("inf")):
            continue

        for dx, dy, step_cost in _NEIGHBORS:
            nx = current[0] + dx
            ny = current[1] + dy
            if not _is_within(nx, ny, img_w, img_h):
                continue
            if not grid[ny, nx]:
                continue

            new_cost = cost + step_cost
            neighbor = (nx, ny)
            if new_cost < visited.get(neighbor, float("inf")):
                visited[neighbor] = new_cost
                heapq.heappush(heap, (new_cost, neighbor))

    raise ValueError("A 点无法到达任何环行轨迹上的可走点，请调整半径或清理障碍")


def _connect_path_segments(grid: np.ndarray, points: List[Point]) -> List[Point]:
    """按照给定顺序连接点列，必要时调用 A* 填补缺口。"""

    if len(points) < 2:
        return points[:]

    full_path: List[Point] = [points[0]]
    for idx in range(1, len(points)):
        prev = full_path[-1]
        target = points[idx]
        if prev == target:
            continue
        dx = abs(prev[0] - target[0])
        dy = abs(prev[1] - target[1])
        if max(dx, dy) == 1:
            # 两个点相邻时直接附加即可
            full_path.append(target)
            continue
        # 若不相邻则调用 A* 搜索，尝试在障碍附近绕行
        segment = a_star(grid, prev, target)
        if not segment:
            raise ValueError(
                "环行轨迹在地图上不可连续绕行，可能被障碍切断，请尝试调整半径或清理障碍"
            )
        # 拼接时去掉首节点避免重复
        full_path.extend(segment[1:])

    return full_path


def build_full_path_from_A(
    grid: np.ndarray,
    A: Point,
    ax: int,
    ay: int,
    r_outer: int,
    r_inner: int,
    cw: bool,
) -> List[Point]:
    """构建从 A 出发绕环一圈的完整路径。"""

    img_h, img_w = grid.shape
    if not _is_within(A[0], A[1], img_w, img_h):
        raise ValueError("A 点越界，请检查输入参数")
    if not grid[A[1], A[0]]:
        raise ValueError("A 点处不可走，请选择可走像素")

    ring_points = rect_track_points(ax, ay, r_outer, r_inner, cw, img_w, img_h, grid)

    nearest = find_nearest_reachable_on_ring(grid, A, ring_points)

    # 第一段路径：A 到最近的环带点
    first_segment = a_star(grid, A, nearest)
    if not first_segment:
        raise ValueError("A 点到环行轨迹之间被障碍阻挡，无法到达")

    # 构建环行顺序，确保完整一圈
    if nearest not in ring_points:
        raise ValueError("最近点不在环行轨迹中，数据出现异常")

    start_idx = ring_points.index(nearest)
    ordered_ring = ring_points[start_idx:] + ring_points[: start_idx + 1]
    if len(ordered_ring) < 2:
        raise ValueError("环行轨迹点数不足，无法绕行一圈")

    ring_path = _connect_path_segments(grid, ordered_ring)

    # 若环路最后一个点不是起始点，则尝试补回起点以形成完整闭环
    if ring_path[-1] != nearest:
        closure = a_star(grid, ring_path[-1], nearest)
        if not closure:
            raise ValueError(
                "环行轨迹无法回到起点，可能被障碍切断，请尝试调整参数或清理障碍"
            )
        ring_path.extend(closure[1:])

    # 合并两段路径，避免重复最近点
    full_path = first_segment + ring_path[1:]
    return full_path


def draw_path(img: np.ndarray, path: List[Point], A: Point, save_path: str = "output_path.png") -> None:
    """在图像上绘制路径并保存。"""

    if not path:
        raise ValueError("路径为空，无法绘制")

    # 将路径转换为 OpenCV 所需的形状 (n, 1, 2)
    pts = np.array(path, dtype=np.int32).reshape(-1, 1, 2)

    # 在图像上绘制路径，使用亮色区分
    cv2.polylines(img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

    # 标记 A 点位置
    cv2.circle(img, A, radius=3, color=(0, 255, 0), thickness=-1)

    cv2.imwrite(save_path, img)
    print(f"保存路径图：{save_path}")

