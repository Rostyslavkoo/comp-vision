"""
Геометричні перетворення зображень: масштабування, поворот, перспектива.
Лабораторна робота №4 — Тиждень 7.
"""

from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Масштабування ─────────────────────────────────────────────────────────────

def scale_image(
    image: np.ndarray,
    fx: float = 1.0,
    fy: float = 1.0,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Масштабує зображення з коефіцієнтами fx (ширина) та fy (висота).
    interpolation: INTER_LINEAR (за замовч.), INTER_NEAREST, INTER_CUBIC, INTER_AREA.
    """
    return cv2.resize(image, None, fx=fx, fy=fy, interpolation=interpolation)


def scale_to_size(
    image: np.ndarray,
    width: int,
    height: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Масштабує зображення до точного розміру (width × height)."""
    return cv2.resize(image, (width, height), interpolation=interpolation)


# ── Поворот ───────────────────────────────────────────────────────────────────

def rotate_image(
    image: np.ndarray,
    angle: float,
    center: Optional[Tuple[int, int]] = None,
    scale: float = 1.0,
    expand: bool = False,
) -> np.ndarray:
    """
    Повертає зображення на кут angle (градуси, проти годинникової стрілки).
    center — центр повороту (за замовчуванням — центр зображення).
    expand=True — автоматично розширює полотно, щоб уникнути обрізання.
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    if expand:
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        return cv2.warpAffine(image, M, (new_w, new_h))

    return cv2.warpAffine(image, M, (w, h))


# ── Перспективне перетворення ─────────────────────────────────────────────────

def perspective_transform(
    image: np.ndarray,
    src_points: List[Tuple[int, int]],
    dst_points: List[Tuple[int, int]],
    output_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Перспективне перетворення (homography): 4 точки джерела → 4 точки призначення.
    src_points, dst_points — списки 4 точок [(x,y), ...].
    output_size — (width, height) вихідного зображення; якщо None — розмір оригіналу.
    """
    h, w = image.shape[:2]
    if output_size is None:
        output_size = (w, h)

    src = np.float32(src_points)
    dst = np.float32(dst_points)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, output_size)


def perspective_transform_auto(
    image: np.ndarray,
    skew_factor: float = 0.15,
) -> np.ndarray:
    """
    Демонстраційне перспективне перетворення: імітує погляд збоку (трапецієвидне спотворення).
    skew_factor контролює ступінь спотворення (0.0 – 0.4).
    """
    h, w = image.shape[:2]
    dx = int(w * skew_factor)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(dx, 0), (w - dx, 0), (w, h), (0, h)]
    return perspective_transform(image, src, dst)


# ── Афінне перетворення ───────────────────────────────────────────────────────

def affine_transform(
    image: np.ndarray,
    src_pts: List[Tuple[int, int]],
    dst_pts: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Афінне перетворення: 3 точки джерела → 3 точки призначення.
    Зберігає паралельність ліній (на відміну від перспективи).
    """
    h, w = image.shape[:2]
    src = np.float32(src_pts[:3])
    dst = np.float32(dst_pts[:3])
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(image, M, (w, h))


# ── Порівняльний grid ─────────────────────────────────────────────────────────

def compare_transformations(
    image: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Показати порівняльний grid усіх геометричних перетворень.
    """
    h, w = image.shape[:2]

    # Афінні точки для демонстрації
    src_affine = [(0, 0), (w - 1, 0), (0, h - 1)]
    dst_affine = [(int(w * 0.1), int(h * 0.1)),
                  (w - 1, int(h * 0.05)),
                  (int(w * 0.15), h - 1)]

    rows = [
        ("Оригінал", image),
        ("Масштаб ×0.5", scale_image(image, fx=0.5, fy=0.5)),
        ("Масштаб ×1.5", scale_image(image, fx=1.5, fy=1.5)),
        ("Поворот 30°", rotate_image(image, 30)),
        ("Поворот 90°", rotate_image(image, 90)),
        ("Поворот 45° + розш.", rotate_image(image, 45, expand=True)),
        ("Перспектива", perspective_transform_auto(image, skew_factor=0.15)),
        ("Афінне", affine_transform(image, src_affine, dst_affine)),
    ]

    # Нормалізуємо розміри для відображення у grid
    target_h, target_w = h, w
    normalized = []
    for title, img in rows:
        resized = scale_to_size(img, target_w, target_h)
        normalized.append((title, resized))

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (title, img) in zip(axes.ravel(), normalized):
        ax.imshow(_to_rgb(img))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.suptitle("Лаб4 — Геометричні перетворення", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
