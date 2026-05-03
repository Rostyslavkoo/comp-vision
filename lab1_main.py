#!/usr/bin/env python3
"""
Лабораторна робота №1: Вступ до комп'ютерного бачення та аналіз цифрових зображень.
Система розпізнавання літаючих об'єктів — Варіант 1.

Тиждень 1: Огляд OpenCV, PIL, NumPy; модуль завантаження та відображення.
Тиждень 2: Завантаження PNG/JPG/BMP, характеристики, гістограма яскравості, покращення контрастності.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # без графічного середовища
import cv2

from histogram_contrast import plot_histogram, show_contrast_comparison
from image_loader import (
    display_image,
    get_image_characteristics,
    load_image,
    list_flying_object_images,
    print_characteristics,
)


def run_lab1(image_path: str, use_gui: bool = True, output_dir: Optional[str] = None) -> None:
    """
    Виконати повний цикл лабораторної №1 для одного зображення.
    """
    path = Path(image_path)
    if not path.is_file():
        print(f"Файл не знайдено: {path}")
        sys.exit(1)

    ext = path.suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".bmp"):
        print(f"Підтримуються лише PNG, JPG, BMP. Отримано: {ext}")
        sys.exit(1)

    out_dir = Path(output_dir) if output_dir else Path("lab1_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Лабораторна робота №1 — Аналіз зображення")
    print("=" * 50)

    # 1. Характеристики зображення (Тиждень 2)
    print_characteristics(path)

    # 2. Завантаження та перевірка
    img = load_image(path)
    if img is None:
        print("Помилка завантаження зображення.")
        sys.exit(1)
    print("Зображення успішно завантажено (OpenCV).")

    if use_gui:
        # 3. Відображення (Тиждень 1 — модуль відображення)
        display_image(img, title="Літальний об'єкт (оригінал)", wait=True, scale=0.8)

    # 4. Гістограма яскравості (Тиждень 2)
    hist_path = out_dir / "histogram.png"
    plot_histogram(img, title="Гістограма яскравості", save_path=str(hist_path), show=use_gui)
    print(f"Гістограму збережено: {hist_path}")

    # 5. Покращення контрастності — порівняння (Тиждень 2)
    comparison_path = out_dir / "contrast_comparison.png"
    show_contrast_comparison(img, save_path=str(comparison_path), show=use_gui)
    print(f"Порівняння контрастності збережено: {comparison_path}")

    print("\nЛабораторна робота №1 виконана.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Лабораторна робота №1: завантаження, характеристики, гістограма, контрастність"
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
        default="lab1_output",
        help="Директорія для збереження гістограм та звітних зображень",
    )
    parser.add_argument(
        "--list",
        metavar="DIR",
        help="Показати список зображень у директорії та їхні характеристики",
    )
    args = parser.parse_args()

    if args.list:
        directory = Path(args.list)
        if not directory.is_dir():
            print(f"Директорія не знайдена: {directory}")
            sys.exit(1)
        images = list_flying_object_images(directory)
        print(f"Знайдено зображень у {directory}: {len(images)}")
        for p in images:
            print(f"\n{p.name}")
            print_characteristics(p)
        return

    if args.image:
        run_lab1(args.image, use_gui=not args.no_gui, output_dir=args.output_dir)
        return

    # Якщо шлях не передано — створити тестове зображення та продемонструвати
    print("Шлях до зображення не вказано. Створюємо тестове зображення для демонстрації.")
    try:
        import numpy as np
        # Просте тестове зображення "літака" (сірий прямокутник + трикутник)
        h, w = 200, 300
        img = np.ones((h, w, 3), dtype=np.uint8) * 180
        cv2.rectangle(img, (50, 80), (250, 120), (100, 100, 100), -1)
        pts = np.array([[150, 50], [200, 100], [100, 100]], np.int32)
        cv2.fillPoly(img, [pts], (120, 120, 120))
        test_path = Path("lab1_output") / "test_flying_object.png"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(test_path), img)
        print(f"Тестове зображення збережено: {test_path}")
        run_lab1(str(test_path), use_gui=not args.no_gui, output_dir=args.output_dir)
    except Exception as e:
        print(f"Помилка: {e}")
        print("Запустіть: python lab1_main.py <шлях_до_зображення.png>")


if __name__ == "__main__":
    main()
