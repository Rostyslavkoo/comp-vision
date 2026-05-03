#!/usr/bin/env python3
"""
Лабораторна робота №3: Витяг ознак та обробка відео.
Система розпізнавання літаючих об'єктів — Варіант 1.

Тиждень 5: Контури, SIFT, ORB, HOG-дескриптори.
Тиждень 6: Оптичний потік (Lucas-Kanade, Farneback), Віднімання фону (MOG2, KNN).
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

# Лаб3 — нові модулі
from features import (
    find_contours,
    draw_contours,
    get_contour_stats,
    detect_sift,
    detect_orb,
    compute_hog,
    compare_descriptors,
)
from video_processing import (
    make_synthetic_video,
    optical_flow_lk,
    optical_flow_dense,
    background_subtraction,
    demo_optical_flow_static,
    demo_background_subtraction_static,
)


def run_lab3(
    image_path: str,
    video_path: Optional[str] = None,
    use_gui: bool = False,
    output_dir: Optional[str] = None,
) -> None:
    out_dir = Path(output_dir) if output_dir else Path("lab3_output")
    video_dir = out_dir / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("Лабораторна робота №3 — Витяг ознак та обробка відео")
    print("=" * 55)

    # ── Тиждень 5: Витяг ознак ────────────────────────────────────────
    print("\n[Тиждень 5] Витяг ознак із зображення")

    print_characteristics(image_path)
    img = load_image(image_path)
    if img is None:
        print(f"Помилка завантаження: {image_path}")
        sys.exit(1)

    # Контури
    contours = find_contours(img)
    stats = get_contour_stats(contours)
    img_contours = draw_contours(img, contours)
    contours_path = out_dir / "contours.png"
    cv2.imwrite(str(contours_path), img_contours)
    print(f"  Контури: знайдено {stats['count']} контурів, "
          f"макс. площа = {stats.get('max_area', 0):.0f} пкс²")
    print(f"  Збережено: {contours_path.relative_to(out_dir.parent)}")

    # SIFT
    kp_sift, desc_sift = detect_sift(img)
    print(f"  SIFT: {len(kp_sift)} ключових точок, "
          f"дескриптор {desc_sift.shape if desc_sift is not None else 'None'}")

    # ORB
    kp_orb, desc_orb = detect_orb(img)
    print(f"  ORB: {len(kp_orb)} ключових точок, "
          f"дескриптор {desc_orb.shape if desc_orb is not None else 'None'}")

    # HOG
    hog_features, hog_vis = compute_hog(img)
    hog_path = out_dir / "hog_visualization.png"
    cv2.imwrite(str(hog_path), hog_vis)
    print(f"  HOG: вектор ознак {hog_features.shape[0]} елементів")
    print(f"  Збережено: {hog_path.relative_to(out_dir.parent)}")

    # Порівняльний grid
    descriptors_path = out_dir / "descriptors.png"
    compare_descriptors(img, save_path=str(descriptors_path), show=use_gui)
    print(f"  Порівняння дескрипторів: {descriptors_path.relative_to(out_dir.parent)}")

    # ── Тиждень 6: Обробка відео ──────────────────────────────────────
    print("\n[Тиждень 6] Оптичний потік та віднімання фону")

    # Визначаємо джерело відео
    if video_path and Path(video_path).is_file():
        vid = video_path
        print(f"  Відео: {video_path}")
    else:
        vid = str(out_dir / "synthetic_video.mp4")
        print("  Відео не вказано — генерується синтетичне відео")
        make_synthetic_video(vid)
        print(f"  Синтетичне відео збережено: {Path(vid).relative_to(out_dir.parent)}")

    # Lucas-Kanade оптичний потік
    lk_dir = str(video_dir)
    try:
        n_lk = optical_flow_lk(vid, save_frames_dir=lk_dir)
        print(f"  Lucas-Kanade: збережено {n_lk} кадрів у {video_dir.name}/lk_flow_*.png")
    except Exception as e:
        print(f"  Lucas-Kanade: помилка — {e}")

    # Farneback щільний потік
    try:
        n_dense = optical_flow_dense(vid, save_frames_dir=lk_dir)
        print(f"  Farneback dense: збережено {n_dense} кадрів у {video_dir.name}/dense_flow_*.png")
    except Exception as e:
        print(f"  Farneback dense: помилка — {e}")

    # Віднімання фону MOG2
    try:
        n_mog2 = background_subtraction(vid, method="MOG2", save_frames_dir=lk_dir)
        print(f"  Background Subtraction MOG2: збережено {n_mog2} кадрів")
    except Exception as e:
        print(f"  Background Subtraction MOG2: помилка — {e}")

    # Віднімання фону KNN
    try:
        n_knn = background_subtraction(vid, method="KNN", save_frames_dir=lk_dir)
        print(f"  Background Subtraction KNN: збережено {n_knn} кадрів")
    except Exception as e:
        print(f"  Background Subtraction KNN: помилка — {e}")

    # Статичні демо-зображення (завжди генеруються для звіту)
    flow_demo_path = out_dir / "optical_flow_demo.png"
    demo_optical_flow_static(save_path=str(flow_demo_path), show=use_gui)
    print(f"  Демо оптичного потоку: {flow_demo_path.relative_to(out_dir.parent)}")

    bg_demo_path = out_dir / "bg_subtraction_demo.png"
    demo_background_subtraction_static(save_path=str(bg_demo_path), show=use_gui)
    print(f"  Демо віднімання фону: {bg_demo_path.relative_to(out_dir.parent)}")

    print("\nЛабораторна робота №3 виконана.")
    print(f"Результати збережено у: {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Лабораторна робота №3: витяг ознак та обробка відео"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Шлях до зображення (PNG, JPG або BMP)",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Шлях до відеофайлу (MP4, AVI тощо)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Не відкривати графічні вікна (лише збереження у файли)",
    )
    parser.add_argument(
        "--output-dir",
        default="lab3_output",
        help="Директорія для збереження результатів",
    )
    args = parser.parse_args()

    use_gui = not args.no_gui

    if args.image:
        path = Path(args.image)
        if not path.is_file():
            print(f"Файл не знайдено: {path}")
            sys.exit(1)
        run_lab3(str(path), video_path=args.video, use_gui=use_gui, output_dir=args.output_dir)
        return

    # Демо без аргументів — використовуємо синтетичне зображення з лаб2
    print("Шлях до зображення не вказано. Використовується синтетичне демо-зображення.")
    demo_dir = Path(args.output_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)

    demo_path = demo_dir / "demo_aircraft.png"
    if not demo_path.exists():
        # Генеруємо аналогічне зображення як у лаб2
        h, w = 300, 400
        img = np.ones((h, w, 3), dtype=np.uint8) * 200
        for y in range(h):
            val = int(200 - y * 0.3)
            img[y, :] = [max(val - 20, 0), max(val - 10, 0), min(val + 30, 255)]
        import cv2 as _cv2
        _cv2.rectangle(img, (80, 130), (320, 170), (80, 80, 80), -1)
        pts_wing = np.array([[140, 150], [200, 90], [260, 150]], np.int32)
        _cv2.fillPoly(img, [pts_wing], (100, 100, 100))
        pts_tail = np.array([[290, 140], [330, 110], [330, 150]], np.int32)
        _cv2.fillPoly(img, [pts_tail], (90, 90, 90))
        for x in [150, 190, 230]:
            _cv2.circle(img, (x, 148), 8, (180, 210, 240), -1)
        _cv2.imwrite(str(demo_path), img)
        print(f"Демо-зображення збережено: {demo_path}")

    run_lab3(str(demo_path), video_path=args.video, use_gui=use_gui, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
