"""
Морфологічні операції: ерозія, дилатація, відкриття, закриття.
Лабораторна робота №4 — Тиждень 8.
"""

from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from segmentation import threshold_otsu


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _make_kernel(kernel_size: int, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    return cv2.getStructuringElement(shape, (kernel_size, kernel_size))


# ── Базові операції ───────────────────────────────────────────────────────────

def erode(
    image: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
    kernel_shape: int = cv2.MORPH_ELLIPSE,
) -> np.ndarray:
    """
    Морфологічна ерозія: стискає світлі об'єкти, видаляє дрібний шум.
    kernel_shape: MORPH_ELLIPSE (за замовч.), MORPH_RECT, MORPH_CROSS.
    """
    kernel = _make_kernel(kernel_size, kernel_shape)
    return cv2.erode(image, kernel, iterations=iterations)


def dilate(
    image: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
    kernel_shape: int = cv2.MORPH_ELLIPSE,
) -> np.ndarray:
    """
    Морфологічна дилатація: розширює світлі об'єкти, заповнює дрібні прогалини.
    """
    kernel = _make_kernel(kernel_size, kernel_shape)
    return cv2.dilate(image, kernel, iterations=iterations)


def opening(
    image: np.ndarray,
    kernel_size: int = 3,
    kernel_shape: int = cv2.MORPH_ELLIPSE,
) -> np.ndarray:
    """
    Відкриття (opening) = ерозія → дилатація.
    Видаляє дрібний шум і тонкі з'єднання між об'єктами.
    """
    kernel = _make_kernel(kernel_size, kernel_shape)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def closing(
    image: np.ndarray,
    kernel_size: int = 3,
    kernel_shape: int = cv2.MORPH_ELLIPSE,
) -> np.ndarray:
    """
    Закриття (closing) = дилатація → ерозія.
    Заповнює дрібні отвори всередині об'єктів.
    """
    kernel = _make_kernel(kernel_size, kernel_shape)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def morphological_gradient(
    image: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Морфологічний градієнт = дилатація − ерозія.
    Виділяє контури об'єктів.
    """
    kernel = _make_kernel(kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def top_hat(image: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """
    Top-hat = оригінал − відкриття. Виділяє дрібні яскраві деталі на темному фоні.
    """
    kernel = _make_kernel(kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def black_hat(image: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """
    Black-hat = закриття − оригінал. Виділяє дрібні темні деталі на світлому фоні.
    """
    kernel = _make_kernel(kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)


# ── Покращення маски сегментації ─────────────────────────────────────────────

def improve_segmentation(
    mask: np.ndarray,
    open_kernel: int = 3,
    close_kernel: int = 5,
) -> np.ndarray:
    """
    Покращує бінарну маску з segmentation.py (лаб2):
    1. Opening (3×3) — видаляє шум і дрібні артефакти
    2. Closing (5×5) — заповнює прогалини всередині об'єктів
    Повертає очищену маску.
    """
    cleaned = opening(mask, kernel_size=open_kernel)
    cleaned = closing(cleaned, kernel_size=close_kernel)
    return cleaned


def segment_and_improve(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Допоміжна функція для інтеграційного демо:
    1. Otsu-сегментація (лаб2) → маска
    2. Морфологічне покращення маски
    Повертає (raw_mask, improved_mask).
    """
    raw_mask = threshold_otsu(image)
    improved_mask = improve_segmentation(raw_mask)
    return raw_mask, improved_mask


# ── Порівняльні grid-и ────────────────────────────────────────────────────────

def compare_morphology(
    image: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Показати порівняльний grid усіх морфологічних операцій.
    Операції застосовуються до бінарної маски (Otsu з лаб2).
    """
    mask = threshold_otsu(image)

    items = [
        ("Оригінал (BGR)", image),
        ("Otsu-маска", mask),
        ("Ерозія (3×3)", erode(mask, 3)),
        ("Дилатація (3×3)", dilate(mask, 3)),
        ("Відкриття (3×3)", opening(mask, 3)),
        ("Закриття (5×5)", closing(mask, 5)),
        ("Градієнт", morphological_gradient(mask, 3)),
        ("Покращена маска", improve_segmentation(mask)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (title, img) in zip(axes.ravel(), items):
        display = _to_rgb(img) if len(img.shape) == 3 else img
        ax.imshow(display, cmap=None if len(img.shape) == 3 else "gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.suptitle("Лаб4 — Морфологічні операції", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def compare_segmentation_improvement(
    image: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Інтеграційне демо: порівняння маски до і після морфологічного покращення.
    Демонструє зв'язок лаб2 (сегментація) + лаб4 (морфологія).
    """
    raw_mask, improved_mask = segment_and_improve(image)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, img, title in zip(
        axes,
        [image, raw_mask, improved_mask],
        ["Оригінал", "Otsu-маска (лаб2)", "Після морфології (лаб4)"],
    ):
        display = _to_rgb(img) if len(img.shape) == 3 else img
        ax.imshow(display, cmap=None if len(img.shape) == 3 else "gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.suptitle("Лаб4 — Покращення сегментації морфологією", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
