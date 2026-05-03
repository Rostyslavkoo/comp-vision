"""
Фільтрація зображень: шумозаглушення та підвищення різкості.
Лабораторна робота №2 — Тиждень 3.
"""

from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def apply_gaussian(
    image: np.ndarray,
    ksize: Tuple[int, int] = (5, 5),
    sigma: float = 0,
) -> np.ndarray:
    """Розмиття Гауса для зглажування шуму."""
    return cv2.GaussianBlur(image, ksize, sigma)


def apply_median(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Медіанний фільтр — ефективний проти імпульсного (salt-pepper) шуму."""
    return cv2.medianBlur(image, ksize)


def apply_bilateral(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """
    Білатеральний фільтр — зберігає краї, прибирає шум.
    d — діаметр сусідства, sigma_color/sigma_space — ширина гаусіан.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def sharpen_unsharp_mask(
    image: np.ndarray,
    ksize: Tuple[int, int] = (5, 5),
    sigma: float = 1.0,
    amount: float = 1.5,
) -> np.ndarray:
    """
    Підвищення різкості методом unsharp mask:
    result = original + amount * (original - blurred)
    """
    blurred = cv2.GaussianBlur(image, ksize, sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened


def sharpen_laplacian(image: np.ndarray) -> np.ndarray:
    """
    Підвищення різкості через лапласіан:
    обчислюємо краї Лапласом, додаємо до оригіналу.
    """
    gray_input = len(image.shape) == 2
    if gray_input:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.clip(np.abs(lap), 0, 255).astype(np.uint8)

    if gray_input:
        sharpened = cv2.add(image, lap_abs)
    else:
        lap_bgr = cv2.cvtColor(lap_abs, cv2.COLOR_GRAY2BGR)
        sharpened = cv2.add(image, lap_bgr)
    return sharpened


def add_synthetic_noise(
    image: np.ndarray,
    kind: str = "gaussian",
    sigma: float = 25.0,
    salt_pepper_ratio: float = 0.02,
) -> np.ndarray:
    """
    Додати синтетичний шум для демонстрації роботи фільтрів.
    kind='gaussian' — гаусівський шум; kind='salt_pepper' — імпульсний шум.
    """
    noisy = image.copy().astype(np.float32)
    if kind == "gaussian":
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)
    elif kind == "salt_pepper":
        noisy = image.copy()
        total = image.size
        n_salt = int(total * salt_pepper_ratio / 2)
        n_pepper = int(total * salt_pepper_ratio / 2)
        # salt (білі пікселі)
        coords = [np.random.randint(0, d, n_salt) for d in image.shape[:2]]
        noisy[coords[0], coords[1]] = 255
        # pepper (чорні пікселі)
        coords = [np.random.randint(0, d, n_pepper) for d in image.shape[:2]]
        noisy[coords[0], coords[1]] = 0
    else:
        raise ValueError(f"Невідомий тип шуму: {kind}. Використовуйте 'gaussian' або 'salt_pepper'.")
    return noisy


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compare_filters(
    image: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Показати порівняльний grid: оригінал, +шум, Гаус, медіана, білатеральний,
    unsharp mask, Лапласіан.
    """
    noisy_g = add_synthetic_noise(image, kind="gaussian")
    noisy_sp = add_synthetic_noise(image, kind="salt_pepper")

    titles = [
        "Оригінал",
        "Гаусівський шум",
        "Salt-Pepper шум",
        "Гаус (σ=1, 5×5)",
        "Медіана (5×5)",
        "Білатеральний",
        "Unsharp Mask",
        "Лапласіан",
    ]
    images = [
        image,
        noisy_g,
        noisy_sp,
        apply_gaussian(noisy_g, ksize=(5, 5), sigma=1),
        apply_median(noisy_sp, ksize=5),
        apply_bilateral(noisy_g),
        sharpen_unsharp_mask(image),
        sharpen_laplacian(image),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, img, title in zip(axes.ravel(), images, titles):
        ax.imshow(_to_rgb(img))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.suptitle("Лаб2 — Фільтрація зображень", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
