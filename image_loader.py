"""
Модуль завантаження та відображення зображень літальних об'єктів.
Лабораторна робота №1 — Вступ до комп'ютерного бачення.
Підтримка форматів: PNG, JPG, BMP.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


# Підтримувані формати зображень
SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".bmp")


def load_image_opencv(path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Завантажити зображення за допомогою OpenCV.
    Повертає BGR-масив (стандарт OpenCV) або None при помилці.
    """
    path = str(path)
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path)
    return img


def load_image_pil(path: Union[str, Path]) -> Optional[Image.Image]:
    """
    Завантажити зображення за допомогою PIL/Pillow.
    Зручно для отримання метаданих та роботи з режимами (RGB, L тощо).
    """
    path = str(path)
    if not os.path.isfile(path):
        return None
    try:
        return Image.open(path).copy()
    except Exception:
        return None


def load_image(path: Union[str, Path], use_rgb: bool = False) -> Optional[np.ndarray]:
    """
    Універсальне завантаження зображення (OpenCV).
    use_rgb=True — конвертує BGR -> RGB для відображення в matplotlib.
    """
    img = load_image_opencv(path)
    if img is None:
        return None
    if use_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def display_image(
    image: np.ndarray,
    title: str = "Зображення",
    wait: bool = True,
    scale: Optional[float] = None,
) -> None:
    """
    Відобразити зображення у вікні OpenCV.
    scale — коефіцієнт зміни розміру (наприклад 0.5 для зменшення вдвічі).
    """
    if image is None or image.size == 0:
        print("Порожнє зображення, відображення пропущено.")
        return
    disp = image.copy()
    if scale is not None and scale != 1.0:
        disp = cv2.resize(disp, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # Якщо RGB (3 канали), конвертувати в BGR для коректного кольору в OpenCV
    if len(disp.shape) == 3 and disp.shape[2] == 3:
        disp = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
    cv2.imshow(title, disp)
    if wait:
        cv2.waitKey(0)
        cv2.destroyWindow(title)


def get_image_characteristics(path: Union[str, Path]) -> Optional[dict]:
    """
    Отримати характеристики зображення (розмір, формат, тип даних, діапазон яскравості).
    Використовує OpenCV та NumPy для аналізу.
    """
    img = load_image_opencv(path)
    if img is None:
        return None

    ext = Path(path).suffix.lower()
    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    dtype = str(img.dtype)

    # Статистика яскравості (по усіх каналах або по сірому)
    if channels == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    min_val = int(np.min(gray))
    max_val = int(np.max(gray))
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    return {
        "path": str(path),
        "format": ext,
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": dtype,
        "min_brightness": min_val,
        "max_brightness": max_val,
        "mean_brightness": round(mean_val, 2),
        "std_brightness": round(std_val, 2),
        "file_size_bytes": os.path.getsize(path) if os.path.isfile(path) else None,
    }


def print_characteristics(path: Union[str, Path]) -> bool:
    """Вивести характеристики зображення в консоль. Повертає True, якщо файл прочитано."""
    info = get_image_characteristics(path)
    if info is None:
        print(f"Не вдалося завантажити: {path}")
        return False
    print("--- Характеристики зображення ---")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print("--------------------------------")
    return True


def list_flying_object_images(directory: Union[str, Path]) -> list:
    """
    Знайти всі зображення літальних об'єктів у заданій директорії
    (файли з розширеннями PNG, JPG, BMP).
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = []
    for ext in SUPPORTED_FORMATS:
        files.extend(directory.glob(f"*{ext}"))
    return sorted(files)
