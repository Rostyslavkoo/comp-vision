"""
Витяг ознак: контури, ключові точки (SIFT, ORB), HOG-дескриптори.
Лабораторна робота №3 — Тиждень 5.
"""

from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from segmentation import threshold_otsu


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Контури ──────────────────────────────────────────────────────────────────

def find_contours(
    image: np.ndarray,
    mode: int = cv2.RETR_EXTERNAL,
    method: int = cv2.CHAIN_APPROX_SIMPLE,
) -> List[np.ndarray]:
    """
    Знаходить контури на зображенні.
    Перед пошуком застосовує Otsu-бінаризацію з segmentation.py (лаб2).
    Повертає список контурів (кожен — масив точок).
    """
    binary = threshold_otsu(image)
    contours, _ = cv2.findContours(binary, mode, method)
    return list(contours)


def draw_contours(
    image: np.ndarray,
    contours: List[np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Малює контури на копії зображення. Повертає BGR-зображення.
    """
    if len(image.shape) == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result = image.copy()
    cv2.drawContours(result, contours, -1, color, thickness)
    return result


def get_contour_stats(contours: List[np.ndarray]) -> dict:
    """
    Повертає базову статистику по контурах: кількість, площі, периметри.
    """
    if not contours:
        return {"count": 0, "areas": [], "perimeters": []}
    areas = [cv2.contourArea(c) for c in contours]
    perimeters = [cv2.arcLength(c, True) for c in contours]
    return {
        "count": len(contours),
        "areas": areas,
        "perimeters": perimeters,
        "max_area": max(areas),
        "total_area": sum(areas),
    }


# ── SIFT ──────────────────────────────────────────────────────────────────────

def detect_sift(
    image: np.ndarray,
    n_features: int = 500,
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Детектує ключові точки та обчислює SIFT-дескриптори.
    Повертає (keypoints, descriptors).
    SIFT — масштабно-інваріантне перетворення; стійкий до зміни масштабу та повороту.
    """
    gray = _to_gray(image)
    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def draw_keypoints_sift(image: np.ndarray, keypoints: List[cv2.KeyPoint]) -> np.ndarray:
    """Малює SIFT-ключові точки з масштабом та орієнтацією."""
    return cv2.drawKeypoints(
        image, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


# ── ORB ───────────────────────────────────────────────────────────────────────

def detect_orb(
    image: np.ndarray,
    n_features: int = 500,
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Детектує ключові точки та обчислює ORB-дескриптори.
    ORB — швидша і безкоштовна альтернатива SIFT/SURF:
    комбінує FAST-детектор + BRIEF-дескриптор із орієнтацією.
    SURF не підтримується у вільній opencv-python через патентні обмеження.
    """
    gray = _to_gray(image)
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def draw_keypoints_orb(image: np.ndarray, keypoints: List[cv2.KeyPoint]) -> np.ndarray:
    """Малює ORB-ключові точки."""
    return cv2.drawKeypoints(image, keypoints, None, color=(255, 165, 0))


# ── HOG ───────────────────────────────────────────────────────────────────────

def compute_hog(
    image: np.ndarray,
    win_size: Tuple[int, int] = (64, 64),
    cell_size: Tuple[int, int] = (8, 8),
    block_size: Tuple[int, int] = (16, 16),
    n_bins: int = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Обчислює HOG (Histogram of Oriented Gradients) дескриптор зображення.
    Масштабує до win_size перед обчисленням.
    Повертає (feature_vector, hog_visualization_image).
    """
    gray = _to_gray(image)
    resized = cv2.resize(gray, win_size)

    hog = cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=block_size,
        _blockStride=(cell_size[0] // 2, cell_size[1] // 2),
        _cellSize=cell_size,
        _nbins=n_bins,
    )
    features = hog.compute(resized)

    # Візуалізація HOG через градієнти
    vis = _hog_visualization(resized, cell_size, n_bins)
    return features.flatten(), vis


def _hog_visualization(
    gray: np.ndarray,
    cell_size: Tuple[int, int],
    n_bins: int,
) -> np.ndarray:
    """Генерує візуалізацію HOG-градієнтів."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    h, w = gray.shape
    cy, cx = cell_size
    vis = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - cy, cy):
        for x in range(0, w - cx, cx):
            cell_mag = magnitude[y:y + cy, x:x + cx]
            cell_ang = angle[y:y + cy, x:x + cx] % 180
            hist, _ = np.histogram(cell_ang, bins=n_bins, range=(0, 180), weights=cell_mag)
            dominant = np.argmax(hist) * (180 // n_bins)
            strength = hist.max()
            if strength > 0:
                rad = np.deg2rad(dominant)
                dx = int(np.cos(rad) * (cx // 2 - 1))
                dy = int(np.sin(rad) * (cy // 2 - 1))
                cx_center = x + cx // 2
                cy_center = y + cy // 2
                vis[
                    max(0, cy_center - abs(dy)):min(h, cy_center + abs(dy) + 1),
                    max(0, cx_center - abs(dx)):min(w, cx_center + abs(dx) + 1),
                ] = min(strength, 255)

    vis_norm = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return vis_norm


# ── Порівняльний grid ─────────────────────────────────────────────────────────

def compare_descriptors(
    image: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Показати порівняльний grid: контури, SIFT, ORB, HOG.
    """
    # Контури
    contours = find_contours(image)
    img_contours = draw_contours(image, contours)

    # SIFT
    kp_sift, _ = detect_sift(image)
    img_sift = draw_keypoints_sift(image, kp_sift)

    # ORB
    kp_orb, _ = detect_orb(image)
    img_orb = draw_keypoints_orb(image, kp_orb)

    # HOG
    _, hog_vis = compute_hog(image)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    data = [
        (image, "Оригінал"),
        (img_contours, f"Контури ({len(contours)})"),
        (img_sift, f"SIFT ({len(kp_sift)} точок)"),
        (img_orb, f"ORB ({len(kp_orb)} точок)"),
        (hog_vis, "HOG"),
    ]
    for ax, (img, title) in zip(axes, data):
        ax.imshow(_to_rgb(img) if len(img.shape) == 3 else img, cmap="gray" if len(img.shape) == 2 else None)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.suptitle("Лаб3 — Витяг ознак", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
