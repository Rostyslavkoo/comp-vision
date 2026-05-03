#!/usr/bin/env python3
"""
Лабораторна робота №2: Фільтрація та сегментація зображень.
Система розпізнавання літаючих об'єктів — Варіант 1.

Тиждень 3: Gaussian, Median, Bilateral фільтри; sharpening (unsharp mask, Laplacian).
Тиждень 4: Порогова сегментація, Otsu, Watershed, GrabCut.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import cv2
import numpy as np

# Лаб1 — повторне використання
from image_loader import load_image, print_characteristics

# Лаб2 — нові модулі
from filtering import (
    apply_gaussian,
    apply_median,
    apply_bilateral,
    sharpen_unsharp_mask,
    sharpen_laplacian,
    add_synthetic_noise,
    compare_filters,
)
from segmentation import (
    threshold_binary,
    threshold_otsu,
    segment_watershed,
    segment_grabcut,
    compare_segmentation,
)


def run_lab2(
    image_path: str,
    use_gui: bool = False,
    output_dir: Optional[str] = None,
) -> None:
    out_dir = Path(output_dir) if output_dir else Path("lab2_output")
    filters_dir = out_dir / "filters"
    seg_dir = out_dir / "segmentation"
    filters_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("Лабораторна робота №2 — Фільтрація та сегментація")
    print("=" * 55)

    print_characteristics(image_path)

    img = load_image(image_path)
    if img is None:
        print(f"Помилка завантаження: {image_path}")
        sys.exit(1)

    # ── Тиждень 3: Фільтрація ──────────────────────────────────────
    print("\n[Тиждень 3] Фільтрація зображень")

    noisy_g = add_synthetic_noise(img, kind="gaussian")
    noisy_sp = add_synthetic_noise(img, kind="salt_pepper")

    pairs = [
        ("gaussian_blur", apply_gaussian(noisy_g)),
        ("median_blur", apply_median(noisy_sp)),
        ("bilateral_filter", apply_bilateral(noisy_g)),
        ("sharpen_unsharp", sharpen_unsharp_mask(img)),
        ("sharpen_laplacian", sharpen_laplacian(img)),
        ("noisy_gaussian", noisy_g),
        ("noisy_salt_pepper", noisy_sp),
    ]
    for name, result in pairs:
        path = filters_dir / f"{name}.png"
        cv2.imwrite(str(path), result)
        print(f"  Збережено: {path.relative_to(out_dir.parent)}")

    compare_path = filters_dir / "filters_comparison.png"
    compare_filters(img, save_path=str(compare_path), show=use_gui)
    print(f"  Порівняння фільтрів: {compare_path.relative_to(out_dir.parent)}")

    # ── Тиждень 4: Сегментація ────────────────────────────────────
    print("\n[Тиждень 4] Сегментація зображень")

    seg_pairs = [
        ("threshold_binary", threshold_binary(img)),
        ("threshold_otsu", threshold_otsu(img)),
        ("watershed", segment_watershed(img)),
        ("grabcut", segment_grabcut(img)),
    ]
    for name, result in seg_pairs:
        path = seg_dir / f"{name}.png"
        cv2.imwrite(str(path), result)
        print(f"  Збережено: {path.relative_to(out_dir.parent)}")

    seg_compare_path = seg_dir / "segmentation_comparison.png"
    compare_segmentation(img, save_path=str(seg_compare_path), show=use_gui)
    print(f"  Порівняння сегментації: {seg_compare_path.relative_to(out_dir.parent)}")

    print("\nЛабораторна робота №2 виконана.")
    print(f"Результати збережено у: {out_dir}/")


def _make_demo_image() -> np.ndarray:
    """Синтетичне зображення 'літак' для демонстрації без реального файлу."""
    h, w = 300, 400
    img = np.ones((h, w, 3), dtype=np.uint8) * 200

    # Небо (градієнт)
    for y in range(h):
        val = int(200 - y * 0.3)
        img[y, :] = [max(val - 20, 0), max(val - 10, 0), min(val + 30, 255)]

    # Тіло літака
    cv2.rectangle(img, (80, 130), (320, 170), (80, 80, 80), -1)
    # Крила
    pts_wing = np.array([[140, 150], [200, 90], [260, 150]], np.int32)
    cv2.fillPoly(img, [pts_wing], (100, 100, 100))
    # Хвіст
    pts_tail = np.array([[290, 140], [330, 110], [330, 150]], np.int32)
    cv2.fillPoly(img, [pts_tail], (90, 90, 90))
    # Ілюмінатори
    for x in [150, 190, 230]:
        cv2.circle(img, (x, 148), 8, (180, 210, 240), -1)

    return img


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Лабораторна робота №2: фільтрація та сегментація зображень"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Шлях до зображення (PNG, JPG або BMP)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Не відкривати графічні вікна (лише збереження у файли)",
    )
    parser.add_argument(
        "--output-dir",
        default="lab2_output",
        help="Директорія для збереження результатів",
    )
    args = parser.parse_args()

    use_gui = not args.no_gui

    if args.image:
        path = Path(args.image)
        if not path.is_file():
            print(f"Файл не знайдено: {path}")
            sys.exit(1)
        run_lab2(str(path), use_gui=use_gui, output_dir=args.output_dir)
        return

    # Демо без аргументів
    print("Шлях до зображення не вказано. Використовується синтетичне демо-зображення.")
    demo_dir = Path(args.output_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)
    demo_path = demo_dir / "demo_aircraft.png"
    demo_img = _make_demo_image()
    cv2.imwrite(str(demo_path), demo_img)
    print(f"Демо-зображення збережено: {demo_path}")
    run_lab2(str(demo_path), use_gui=use_gui, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
