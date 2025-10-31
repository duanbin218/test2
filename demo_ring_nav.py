"""环带巡边寻路示例脚本。"""

import time
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    raise SystemExit("未检测到 OpenCV(cv2)，请先安装依赖：pip install opencv-python") from exc
import numpy as np

from ring_nav import (
    build_ring_mask,
    draw_path_on_image,
    trace_ring_path,
)


def _build_demo_map(width: int, height: int) -> np.ndarray:
    """生成一张包含障碍的测试图像（BGR 三通道）。"""

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    # 绘制几块障碍区域，使路径更加多样。
    cv2.rectangle(canvas, (10, 10), (20, 25), (0, 0, 0), -1)
    cv2.rectangle(canvas, (35, 5), (45, 15), (0, 0, 0), -1)
    cv2.line(canvas, (5, 40), (50, 40), (0, 0, 0), 3)
    return canvas


def _compute_walkable_mask(img: np.ndarray) -> np.ndarray:
    """将彩色图像转为布尔可走掩膜。"""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary > 127


def run_demo_case(
    img: np.ndarray,
    a: tuple[int, int],
    inner_r: int,
    outer_r: int,
    clockwise: bool,
    anchor_count: int,
    color: tuple[int, int, int],
    label: str,
) -> tuple[list[tuple[int, int]], dict]:
    """执行单组参数的巡边示例，返回锚点路径与诊断信息。"""

    walkable_mask = _compute_walkable_mask(img)
    height, width = walkable_mask.shape
    ring_mask = build_ring_mask(a[0], a[1], inner_r, outer_r, width, height)

    start_time = time.time()
    path = trace_ring_path(
        a,
        ring_mask,
        walkable_mask,
        clockwise=clockwise,
        anchor_count=anchor_count,
    )
    duration = time.time() - start_time

    stats = dict(trace_ring_path.last_metadata)
    stats["耗时秒"] = duration
    stats["参数标签"] = label
    stats["方向"] = "顺时针" if clockwise else "逆时针"
    stats["锚点数量参数"] = anchor_count
    print(
        f"【{label}】环带点数={stats['环带候选点数']}，锚点数={stats['锚点数量']}，"
        f"最近点={stats['最近环带点']}，锚点列表={path}，"
        f"段数={stats['成功段数']}，回退={stats['回退次数']}，跳过={stats['跳过点数']}，"
        f"路径长度={stats['总步长']}，耗时={duration:.4f} 秒"
    )
    return path, stats


def main() -> None:
    """运行两组巡边示例并生成可视化结果。"""

    width, height = 60, 60
    img = _build_demo_map(width, height)

    cases = [
        {
            "a": (30, 30),
            "inner_r": 5,
            "outer_r": 10,
            "clockwise": True,
            "anchor_count": 4,
            "color": (0, 0, 255),
            "label": "示例一",
        },
        {
            "a": (24, 18),
            "inner_r": 3,
            "outer_r": 8,
            "clockwise": False,
            "anchor_count": 8,
            "color": (255, 0, 0),
            "label": "示例二",
        },
    ]

    all_paths = []
    for case in cases:
        path, stats = run_demo_case(
            img,
            case["a"],
            case["inner_r"],
            case["outer_r"],
            case["clockwise"],
            case["anchor_count"],
            case["color"],
            case["label"],
        )
        all_paths.append((stats["密集路径"], case["color"], case["label"]))

    output = img.copy()
    for path, color, label in all_paths:
        draw_path_on_image(output, path, "tmp.png", color=color, thickness=2)
        # draw_path_on_image 会保存图像，这里仅获取绘制结果。
        output = cv2.imread("tmp.png")
        if output is None:
            raise RuntimeError("绘图过程中临时文件读取失败。")
    Path("tmp.png").unlink(missing_ok=True)

    out_path = "demo_output.png"
    cv2.putText(output, "示例一=红色(顺时针,4点)", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(output, "示例二=蓝色(逆时针,8点)", (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.imwrite(out_path, output)
    print(f"结果图已保存至 {out_path}")


if __name__ == "__main__":
    main()
