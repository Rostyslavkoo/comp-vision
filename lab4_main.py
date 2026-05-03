#!/usr/bin/env python3
"""
Лабораторна робота №4: Геометричні перетворення та морфологічні операції.
Система розпізнавання літаючих об'єктів — Варіант 1.

Тиждень 7: Масштабування, поворот, перспективне та афінне перетворення.
Тиждень 8: Ерозія, дилатація, відкриття, закриття; покращення сегментації.
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

# Лаб4 — нові модулі
from geometry import (
    scale_image,
    scale_to_size,
    rotate_image,
    perspective_transform_auto,
    affine_transform,
    compare_transformations,
)
from morphology import (
    erode,
    dilate,
    opening,
    closing,
    morphological_gradient,
    top_hat,
    black_hat,
    improve_segmentation,
    segment_and_improve,
    compare_morphology,
    compare_segmentation_improvement,
)


def run_lab4(
    image_path: str,
    use_gui: bool = False,
    output_dir: Optional[str] = None,
) -> None:
    out_dir = Path(output_dir) if output_dir else Path("lab4_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Лабораторна робота №4 — Геометричні перетворення та морфологія")
    print("=" * 60)

    print_characteristics(image_path)
    img = load_image(image_path)
    if img is None:
        print(f"Помилка завантаження: {image_path}")
        sys.exit(1)

    h, w = img.shape[:2]

    # ── Тиждень 7: Геометричні перетворення ──────────────────────────
    print("\n[Тиждень 7] Геометричні перетворення")

    # Масштабування
    img_half = scale_image(img, fx=0.5, fy=0.5)
    img_double = scale_image(img, fx=1.5, fy=1.5)
    cv2.imwrite(str(out_dir / "scale_0.5x.png"), img_half)
    cv2.imwrite(str(out_dir / "scale_1.5x.png"), img_double)
    print(f"  Масштаб ×0.5: {img_half.shape[1]}×{img_half.shape[0]} пкс → збережено")
    print(f"  Масштаб ×1.5: {img_double.shape[1]}×{img_double.shape[0]} пкс → збережено")

    # Поворот
    img_rot30 = rotate_image(img, 30)
    img_rot90 = rotate_image(img, 90)
    img_rot45e = rotate_image(img, 45, expand=True)
    cv2.imwrite(str(out_dir / "rotate_30.png"), img_rot30)
    cv2.imwrite(str(out_dir / "rotate_90.png"), img_rot90)
    cv2.imwrite(str(out_dir / "rotate_45_expand.png"), img_rot45e)
    print(f"  Поворот 30°, 90°, 45° (з розш. полотна) → збережено")

    # Перспектива
    img_persp = perspective_transform_auto(img, skew_factor=0.15)
    cv2.imwrite(str(out_dir / "perspective.png"), img_persp)
    print(f"  Перспективне перетворення (skew=0.15) → збережено")

    # Афінне
    src_pts = [(0, 0), (w - 1, 0), (0, h - 1)]
    dst_pts = [(int(w * 0.1), int(h * 0.1)),
               (w - 1, int(h * 0.05)),
               (int(w * 0.15), h - 1)]
    img_affine = affine_transform(img, src_pts, dst_pts)
    cv2.imwrite(str(out_dir / "affine.png"), img_affine)
    print(f"  Афінне перетворення → збережено")

    # Порівняльний grid
    trans_path = out_dir / "transformations_comparison.png"
    compare_transformations(img, save_path=str(trans_path), show=use_gui)
    print(f"  Порівняння перетворень: {trans_path.relative_to(out_dir.parent)}")

    # ── Тиждень 8: Морфологічні операції ─────────────────────────────
    print("\n[Тиждень 8] Морфологічні операції")

    # Отримуємо бінарну маску через Otsu (лаб2)
    from segmentation import threshold_otsu
    mask = threshold_otsu(img)

    morph_pairs = [
        ("erode_3x3", erode(mask, 3)),
        ("dilate_3x3", dilate(mask, 3)),
        ("opening_3x3", opening(mask, 3)),
        ("closing_5x5", closing(mask, 5)),
        ("gradient_3x3", morphological_gradient(mask, 3)),
        ("top_hat", top_hat(img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))),
        ("black_hat", black_hat(img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))),
        ("improved_mask", improve_segmentation(mask)),
    ]
    for name, result in morph_pairs:
        cv2.imwrite(str(out_dir / f"{name}.png"), result)
    print(f"  Збережено {len(morph_pairs)} морфологічних результатів")

    # Порівняльний grid морфології
    morph_path = out_dir / "morphology_comparison.png"
    compare_morphology(img, save_path=str(morph_path), show=use_gui)
    print(f"  Порівняння морфології: {morph_path.relative_to(out_dir.parent)}")

    # Інтеграційне демо: лаб2 + лаб4
    seg_improve_path = out_dir / "segmentation_improvement.png"
    compare_segmentation_improvement(img, save_path=str(seg_improve_path), show=use_gui)
    print(f"  Покращення сегментації (лаб2→лаб4): {seg_improve_path.relative_to(out_dir.parent)}")

    print("\nЛабораторна робота №4 виконана.")
    print(f"Результати збережено у: {out_dir}/")


def _make_demo_image() -> np.ndarray:
    """Синтетичне зображення для демонстрації (аналогічно лаб2)."""
    h, w = 300, 400
    img = np.ones((h, w, 3), dtype=np.uint8) * 200
    for y in range(h):
        val = int(200 - y * 0.3)
        img[y, :] = [max(val - 20, 0), max(val - 10, 0), min(val + 30, 255)]
    cv2.rectangle(img, (80, 130), (320, 170), (80, 80, 80), -1)
    pts_wing = np.array([[140, 150], [200, 90], [260, 150]], np.int32)
    cv2.fillPoly(img, [pts_wing], (100, 100, 100))
    pts_tail = np.array([[290, 140], [330, 110], [330, 150]], np.int32)
    cv2.fillPoly(img, [pts_tail], (90, 90, 90))
    for x in [150, 190, 230]:
        cv2.circle(img, (x, 148), 8, (180, 210, 240), -1)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Лабораторна робота №4: геометричні перетворення та морфологія"
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
        default="lab4_output",
        help="Директорія для збереження результатів",
    )
    args = parser.parse_args()

    use_gui = not args.no_gui

    if args.image:
        path = Path(args.image)
        if not path.is_file():
            print(f"Файл не знайдено: {path}")
            sys.exit(1)
        run_lab4(str(path), use_gui=use_gui, output_dir=args.output_dir)
        return

    print("Шлях до зображення не вказано. Використовується синтетичне демо-зображення.")
    demo_dir = Path(args.output_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)
    demo_path = demo_dir / "demo_aircraft.png"
    cv2.imwrite(str(demo_path), _make_demo_image())
    print(f"Демо-зображення збережено: {demo_path}")
    run_lab4(str(demo_path), use_gui=use_gui, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
