"""命令行演示脚本：生成或读取地图，并执行环带绕行寻路。"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from ring_path import (
    Point,
    build_full_path_from_A,
    draw_path,
    image_to_grid,
)


def _str_to_bool(value: str) -> bool:
    """将字符串转换为布尔值，支持大小写混合的 true/false。"""

    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("布尔参数只能为 true/false")


def _generate_demo_map(width: int = 200, height: int = 200) -> np.ndarray:
    """生成一张默认的黑白测试图，方便快速验证算法。"""

    # 背景初始化为纯黑，可行走区域
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 在图中绘制若干白色障碍，用于制造绕行需求
    cv2.rectangle(img, (40, 40), (80, 160), (255, 255, 255), thickness=2)
    cv2.rectangle(img, (120, 20), (160, 180), (255, 255, 255), thickness=2)
    cv2.circle(img, (100, 100), 15, (255, 255, 255), thickness=2)

    return img


def parse_args() -> argparse.Namespace:
    """解析命令行参数，所有帮助信息均为中文。"""

    parser = argparse.ArgumentParser(description="在正方形环带上绕行一圈的寻路演示")
    parser.add_argument("--map", type=str, default="", help="可选，指定黑白底图路径")
    parser.add_argument("--ax", type=int, default=100, help="A 点的 x 坐标")
    parser.add_argument("--ay", type=int, default=100, help="A 点的 y 坐标")
    parser.add_argument("--r_outer", type=int, default=30, help="环带外半径")
    parser.add_argument("--r_inner", type=int, default=15, help="环带内半径")
    parser.add_argument("--cw", type=_str_to_bool, default=True, help="是否顺时针绕行，true/false")
    return parser.parse_args()


def load_map(path_str: str) -> np.ndarray:
    """根据参数读取或生成地图，所有异常都提供中文提示。"""

    if not path_str:
        print("未指定地图，使用默认生成的 200x200 测试图")
        return _generate_demo_map()

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError("指定的地图文件不存在")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法读取地图文件，请确认路径与格式正确")

    return img


def main() -> None:
    """命令行入口：准备地图、执行寻路并保存可视化结果。"""

    args = parse_args()

    try:
        img = load_map(args.map)
    except Exception as exc:  # noqa: BLE001
        print(f"读取地图失败：{exc}")
        sys.exit(1)

    # 保存原始图像的拷贝用于绘制结果
    img_copy = img.copy()

    grid = image_to_grid(img)
    img_h, img_w = grid.shape
    print(f"地图尺寸：{img_w}x{img_h}")

    A: Point = (args.ax, args.ay)
    print(f"A 点位置：{A}")
    print(
        f"环带参数：外半径={args.r_outer} 内半径={args.r_inner} 绕行方向={'顺时针' if args.cw else '逆时针'}"
    )

    try:
        path = build_full_path_from_A(
            grid,
            A,
            args.ax,
            args.ay,
            args.r_outer,
            args.r_inner,
            args.cw,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"寻路失败：{exc}")
        sys.exit(1)

    print(f"最终路径长度：{len(path)}")
    draw_path(img_copy, path, A)


if __name__ == "__main__":
    main()

