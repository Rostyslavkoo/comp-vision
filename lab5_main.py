#!/usr/bin/env python3
"""
Лабораторна робота №5: Класифікація зображень.
Система розпізнавання літаючих об'єктів — Варіант 1.

Тиждень 9: Класичне ML — SVM, Random Forest, k-NN на HOG-ознаках.
Тиждень 10: CNN з нуля (PyTorch): Conv→ReLU→MaxPool×3 → FC→FC.
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

# Лаб5 — нові модулі
from classical_ml import (
    build_dataset,
    build_synthetic_dataset,
    train_svm,
    train_random_forest,
    train_knn,
    evaluate,
    compare_classifiers,
)
from cnn_model import (
    FlyingObjectCNN,
    FlyingObjectDataset,
    SyntheticFlyingDataset,
    train_cnn,
    evaluate_cnn,
    plot_training_history,
    IMG_SIZE,
)


def run_lab5(
    data_dir: Optional[str],
    use_gui: bool = False,
    output_dir: str = "lab5_output",
    cnn_epochs: int = 20,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Лабораторна робота №5 — Класифікація зображень")
    print("=" * 60)

    use_synthetic = data_dir is None or not Path(data_dir).is_dir()

    # ── Тиждень 9: Класичне ML ────────────────────────────────────────────────
    print("\n[Тиждень 9] Класичне ML (SVM / Random Forest / k-NN)")

    if use_synthetic:
        print("  Реальний датасет не знайдено → синтетичний демо-датасет (80 зображень/клас)")
        X_train, X_test, y_train, y_test, label_map = build_synthetic_dataset(n_per_class=80)
    else:
        print(f"  Завантаження датасету з {data_dir} …")
        X_train, X_test, y_train, y_test, label_map = build_dataset(data_dir)

    print(f"  Train: {len(X_train)} | Test: {len(X_test)} | Класи: {list(label_map.keys())}")
    print(f"  Розмір HOG-вектора: {X_train.shape[1]}")

    # Тренування
    print("  Тренування SVM …")
    svm = train_svm(X_train, y_train)
    svm_metrics = evaluate(svm, X_test, y_test, label_map)

    print("  Тренування Random Forest …")
    rf = train_random_forest(X_train, y_train)
    rf_metrics = evaluate(rf, X_test, y_test, label_map)

    print("  Тренування k-NN …")
    knn = train_knn(X_train, y_train)
    knn_metrics = evaluate(knn, X_test, y_test, label_map)

    # Результати
    print("\n  ── Результати ──────────────────────────────────────")
    for name, metrics in [("SVM", svm_metrics), ("Random Forest", rf_metrics), ("k-NN", knn_metrics)]:
        print(f"  {name:<16} Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print()
    print("  Детальний звіт SVM:")
    print(svm_metrics["report"])

    # Порівняльний графік
    ml_plot_path = out_dir / "classifiers_comparison.png"
    accuracies = compare_classifiers(
        X_train, X_test, y_train, y_test, label_map,
        save_path=str(ml_plot_path),
        show=use_gui,
    )
    print(f"  Порівняння класифікаторів збережено: {ml_plot_path}")

    # ── Тиждень 10: CNN ───────────────────────────────────────────────────────
    print("\n[Тиждень 10] CNN з нуля (PyTorch)")

    import torch
    from torch.utils.data import DataLoader, random_split

    if use_synthetic:
        print("  Використовується синтетичний датасет (100 зображень/клас для CNN)")
        full_ds = SyntheticFlyingDataset(n_per_class=100, img_size=IMG_SIZE)
        num_classes = 3
        cnn_label_map = {"aircraft": 0, "bird": 1, "helicopter": 2}
    else:
        full_ds = FlyingObjectDataset(data_dir, img_size=IMG_SIZE)
        num_classes = len(full_ds.label_map)
        cnn_label_map = full_ds.label_map

    # Розбиваємо на train/val/test = 70/15/15
    total = len(full_ds)
    n_test = max(1, int(total * 0.15))
    n_val = max(1, int(total * 0.15))
    n_train = total - n_val - n_test

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"  Архітектура: Conv(32)→ReLU→MaxPool → Conv(64)→ReLU→MaxPool → Conv(128)→ReLU→MaxPool → FC(256)→FC({num_classes})")
    print(f"  Пристрій: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Тренування {cnn_epochs} епох …\n")

    cnn_model, history = train_cnn(
        train_loader, val_loader,
        num_classes=num_classes,
        epochs=cnn_epochs,
    )

    cnn_acc, _, _ = evaluate_cnn(cnn_model, test_loader)
    print(f"\n  CNN Test Accuracy: {cnn_acc:.4f} ({cnn_acc*100:.1f}%)")

    # Зберігаємо ваги
    weights_path = out_dir / "cnn_weights.pth"
    torch.save(cnn_model.state_dict(), str(weights_path))
    print(f"  Ваги CNN збережено: {weights_path}")

    # Графік навчання
    history_plot = out_dir / "cnn_training_history.png"
    plot_training_history(history, save_path=str(history_plot), show=use_gui)
    print(f"  Графік навчання збережено: {history_plot}")

    # ── Фінальне порівняння ───────────────────────────────────────────────────
    print("\n  ── Підсумкова таблиця ──────────────────────────────")
    print(f"  {'Модель':<18} {'Accuracy':>10}")
    print(f"  {'-'*30}")
    for name, acc in accuracies.items():
        print(f"  {name:<18} {acc*100:>9.1f}%")
    print(f"  {'CNN (PyTorch)':<18} {cnn_acc*100:>9.1f}%")

    print("\nЛабораторна робота №5 виконана.")
    print(f"Результати збережено у: {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Лабораторна робота №5: класичне ML + CNN класифікація"
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=None,
        help="Директорія датасету (підпапки = класи). Без аргументів — синтетичний демо.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Не відкривати графічні вікна (лише збереження у файли)",
    )
    parser.add_argument(
        "--output-dir",
        default="lab5_output",
        help="Директорія для збереження результатів",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Кількість епох навчання CNN (за замовчуванням: 20)",
    )
    args = parser.parse_args()

    use_gui = not args.no_gui

    if args.data_dir and not Path(args.data_dir).is_dir():
        print(f"Директорію не знайдено: {args.data_dir}")
        sys.exit(1)

    run_lab5(
        data_dir=args.data_dir,
        use_gui=use_gui,
        output_dir=args.output_dir,
        cnn_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
