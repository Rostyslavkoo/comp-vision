"""
Гістограма яскравості та методи покращення контрастності.
Лабораторна робота №1 — Тиждень 2: Аналіз цифрових зображень.
"""

from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_grayscale(image: np.ndarray) -> np.ndarray:
    """Конвертувати зображення в відтінки сірого (якщо потрібно)."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def build_brightness_histogram(
    image: np.ndarray,
    bins: int = 256,
    range_vals: Tuple[int, int] = (0, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Побудувати гістограму яскравості зображення.
    Повертає (значення пікселів 0..255, кількість пікселів для кожного рівня).
    """
    gray = get_grayscale(image)
    hist, bin_edges = np.histogram(gray.ravel(), bins=bins, range=range_vals)
    return np.arange(bins), hist


def plot_histogram(
    image: np.ndarray,
    title: str = "Гістограма яскравості",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Побудувати та відобразити гістограму яскравості за допомогою matplotlib.
    """
    x, hist = build_brightness_histogram(image)
    plt.figure(figsize=(8, 4))
    plt.bar(x, hist, color="gray", width=1.0, edgecolor="none")
    plt.title(title)
    plt.xlabel("Рівень яскравості")
    plt.ylabel("Кількість пікселів")
    plt.xlim(0, 256)
    plt.grid(axis="y", alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def enhance_contrast_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Покращення контрастності методом гістограмної еквіалізації (OpenCV).
    Застосовується до сірого зображення; для кольорового — до каналу L (HSV) або Y.
    """
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    # Кольорове: перетворюємо в YCrCb, еквіалізуємо Y, збираємо назад
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def enhance_contrast_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Покращення контрастності методом CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Краще зберігає локальні деталі, менше шуму.
    """
    gray = get_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(image.shape) == 2:
        return clahe.apply(image)
    # Кольорове: CLAHE тільки на канал яскравості (Y в YCrCb)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def enhance_contrast_normalize(image: np.ndarray) -> np.ndarray:
    """
    Покращення контрастності лінійною нормалізацією (розтягування діапазону 0–255).
    """
    gray = get_grayscale(image)
    min_val = np.min(gray)
    max_val = np.max(gray)
    if max_val <= min_val:
        return image.copy()
    normalized = np.clip((gray.astype(np.float32) - min_val) / (max_val - min_val) * 255, 0, 255)
    result = normalized.astype(np.uint8)
    if len(image.shape) == 3:
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


def show_contrast_comparison(
    image: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Показати оригінал та три варіанти покращення контрастності в одному figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Для відображення перетворюємо BGR -> RGB
    def to_rgb(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    axes[0, 0].imshow(to_rgb(image))
    axes[0, 0].set_title("Оригінал")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(to_rgb(enhance_contrast_histogram_equalization(image)))
    axes[0, 1].set_title("Гістограмна еквіалізація")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(to_rgb(enhance_contrast_clahe(image)))
    axes[1, 0].set_title("CLAHE")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(to_rgb(enhance_contrast_normalize(image)))
    axes[1, 1].set_title("Лінійна нормалізація")
    axes[1, 1].axis("off")

    plt.suptitle("Методи покращення контрастності", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
