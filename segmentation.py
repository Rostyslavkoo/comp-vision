"""
Сегментація зображень: порогова, Otsu, Watershed, GrabCut.
Лабораторна робота №2 — Тиждень 4.
"""

from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def threshold_binary(image: np.ndarray, thresh: int = 127) -> np.ndarray:
    """
    Проста бінарна порогова сегментація.
    Повертає бінарну маску (0 або 255).
    """
    gray = _to_gray(image)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return mask


def threshold_otsu(image: np.ndarray) -> np.ndarray:
    """
    Автоматична порогова сегментація методом Otsu.
    Оптимальний поріг знаходиться автоматично за гістограмою.
    Повертає бінарну маску.
    """
    gray = _to_gray(image)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def segment_watershed(image: np.ndarray) -> np.ndarray:
    """
    Сегментація методом Watershed для розділення об'єктів, що торкаються.
    Пайплайн: grayscale → Otsu → морфологія → distance transform → markers → watershed.
    Повертає кольорове зображення з розфарбованими сегментами.
    """
    gray = _to_gray(image)

    # Бінаризація + морфологічне очищення
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Визначення явно-фонових та явно-переднього-планових пікселів
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)

    # Маркери для watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed потребує BGR-зображення
    if len(image.shape) == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image.copy()

    markers = cv2.watershed(bgr, markers)

    # Візуалізація: різні кольори для різних сегментів
    result = bgr.copy()
    result[markers == -1] = [0, 0, 255]  # межі — червоні

    # Розфарбовуємо сегменти
    unique = np.unique(markers)
    rng = np.random.default_rng(42)
    for label in unique:
        if label <= 1:
            continue
        color = rng.integers(50, 230, size=3).tolist()
        result[markers == label] = color

    return result


def segment_grabcut(
    image: np.ndarray,
    rect: Optional[Tuple[int, int, int, int]] = None,
    iterations: int = 5,
) -> np.ndarray:
    """
    Сегментація методом GrabCut із прямокутником ROI.
    rect — (x, y, w, h); якщо None — використовується центральні 60% зображення.
    Повертає зображення з виділеним переднім планом.
    """
    if len(image.shape) == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image.copy()

    h, w = bgr.shape[:2]
    if rect is None:
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    mask = np.zeros((h, w), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(bgr, mask, rect, bg_model, fg_model, iterations, cv2.GC_INIT_WITH_RECT)

    # 0=явний-фон, 2=можливий-фон → маска переднього плану
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    result = cv2.bitwise_and(bgr, bgr, mask=fg_mask)
    return result


def compare_segmentation(
    image: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Показати порівняльний grid усіх методів сегментації.
    """
    titles = [
        "Оригінал",
        "Порогова (thresh=127)",
        "Otsu",
        "Watershed",
        "GrabCut",
    ]
    results = [
        image,
        threshold_binary(image, thresh=127),
        threshold_otsu(image),
        segment_watershed(image),
        segment_grabcut(image),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, img, title in zip(axes, results, titles):
        ax.imshow(_to_rgb(img))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.suptitle("Лаб2 — Методи сегментації", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
