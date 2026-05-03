# Архітектура проєкту: Система розпізнавання літаючих об'єктів

> Єдиний наскрізний проєкт — кожна лабораторна додає нові модулі поверх попередніх.
> Усі лабораторні — окремі гілки у Git (`lab2`, `lab3`, …).

---

## Структура файлів

```
dvvs-comp-bachenia/
│
│  ── Документація ────────────────────────────────────────────────
├── Task.md                      завдання (всі 7 лабораторних)
├── ARCHITECTURE.md              цей файл
│
│  ── Лабораторна 1 ───────────────────────────────────────────────
├── image_loader.py              завантаження/відображення зображень
├── histogram_contrast.py        гістограма яскравості, CLAHE, нормалізація
├── lab1_main.py                 ► точка входу лаб1
├── README_LAB1.md               звіт лаб1
├── ОГЛЯД_БІБЛІОТЕК.md           огляд OpenCV / PIL / NumPy
├── lab1_output/
│   ├── histogram.png
│   ├── contrast_comparison.png
│   └── test_flying_object.png
│
│  ── Лабораторна 2 ───────────────────────────────────────────────
├── filtering.py                 Gaussian, Median, Bilateral; sharpening
├── segmentation.py              threshold, Otsu, Watershed, GrabCut
├── lab2_main.py                 ► точка входу лаб2
├── README_LAB2.md               звіт лаб2
├── lab2_output/
│   ├── demo_aircraft.png
│   ├── filters/
│   │   ├── gaussian_blur.png
│   │   ├── median_blur.png
│   │   ├── bilateral_filter.png
│   │   ├── sharpen_unsharp.png
│   │   ├── sharpen_laplacian.png
│   │   ├── noisy_gaussian.png
│   │   ├── noisy_salt_pepper.png
│   │   └── filters_comparison.png
│   └── segmentation/
│       ├── threshold_binary.png
│       ├── threshold_otsu.png
│       ├── watershed.png
│       ├── grabcut.png
│       └── segmentation_comparison.png
│
│  ── Лабораторна 3 (планується) ──────────────────────────────────
├── features.py                  контури, SIFT, ORB, HOG дескриптори
├── video_processing.py          Optical Flow, Background Subtraction
├── lab3_main.py                 ► точка входу лаб3
├── README_LAB3.md
└── lab3_output/
    ├── contours.png
    ├── descriptors.png
    └── video/
        ├── optical_flow_*.png
        └── bg_subtraction_*.png
│
│  ── Лабораторна 4 (планується) ──────────────────────────────────
├── geometry.py                  scaling, rotation, perspective
├── morphology.py                erosion, dilation, opening, closing
├── lab4_main.py                 ► точка входу лаб4
├── README_LAB4.md
└── lab4_output/
    ├── transformations_comparison.png
    └── morphology_comparison.png
│
│  ── Лабораторна 5 (планується) ──────────────────────────────────
├── classical_ml.py              SVM, Random Forest, KNN + HOG-ознаки
├── cnn_model.py                 CNN з нуля (PyTorch)
├── lab5_main.py                 ► точка входу лаб5
├── README_LAB5.md
└── lab5_output/
    ├── classifiers_accuracy.png
    ├── cnn_training_curve.png
    ├── confusion_matrix.png
    └── cnn_weights.pth
│
│  ── Лабораторна 6 (планується) ──────────────────────────────────
├── pretrained_models.py         ResNet50, MobileNetV3, YOLOv8
├── augmentation.py              rotation, brightness, noise, flip
├── lab6_main.py                 ► точка входу лаб6
├── README_LAB6.md
└── lab6_output/
    ├── yolo_detection.png
    ├── resnet_predictions.png
    └── augmentation_grid.png
│
│  ── Лабораторна 7 (планується) ──────────────────────────────────
├── realtime_detection.py        відео/webcam детекція, satellite tiling
├── lab7_main.py                 ► точка входу лаб7
├── README_LAB7.md
└── lab7_output/
    ├── video_detection.mp4
    └── satellite_detection.png
│
│  ── Датасет (не в git) ──────────────────────────────────────────
├── data/                        .gitignore — не завантажується в git
│   ├── raw/
│   │   ├── aircraft/            ~100+ фото літаків (з Kaggle)
│   │   ├── helicopter/          ~100+ фото гелікоптерів
│   │   └── bird/                ~100+ фото птахів
│   ├── augmented/               результат augmentation.py (лаб6)
│   └── videos/
│       └── sample.mp4           тестове відео (лаб3, лаб7)
│
│  ── Конфігурація ────────────────────────────────────────────────
├── requirements.txt             залежності Python (оновлюється по лабах)
└── .gitignore                   виключає data/, *.pth, __pycache__/
```

---

## Принцип повторного використання

Кожна нова лабораторна **імпортує** модулі попередніх — нічого не копіюється.

```
lab7_main.py
  ├── realtime_detection.py   [лаб7]
  │     └── pretrained_models.py  [лаб6]  ← YOLO/ResNet для детекції
  ├── augmentation.py         [лаб6]
  │     ├── geometry.py       [лаб4]  ← random_rotation
  │     └── filtering.py      [лаб2]  ← add_synthetic_noise
  ├── classical_ml.py         [лаб5]
  │     └── features.py       [лаб3]  ← compute_hog (витяг ознак)
  ├── segmentation.py         [лаб2]  ← threshold_otsu (маски)
  └── image_loader.py         [лаб1]  ← load_image (завантаження)
```

---

## Git-гілки

| Гілка | Вміст |
|-------|-------|
| `master` | Лабораторна 1 (вихідний стан) |
| `lab2` | + filtering.py, segmentation.py ✓ готово |
| `lab3` | + features.py, video_processing.py |
| `lab4` | + geometry.py, morphology.py |
| `lab5` | + classical_ml.py, cnn_model.py |
| `lab6` | + pretrained_models.py, augmentation.py |
| `lab7` | + realtime_detection.py (фінальна версія) |

**Здача:** кожна гілка здається окремо. Новіша гілка завжди містить увесь попередній код.

---

## Запуск кожної лабораторної

```bash
# Лаб1
python3 lab1_main.py --no-gui

# Лаб2
python3 lab2_main.py --no-gui

# Лаб3 (після реалізації)
python3 lab3_main.py --no-gui

# Лаб4
python3 lab4_main.py --no-gui

# Лаб5 (потрібен датасет у data/raw/)
python3 lab5_main.py --no-gui

# Лаб6
python3 lab6_main.py --no-gui

# Лаб7 — відеофайл
python3 lab7_main.py --video data/videos/sample.mp4 --no-gui

# Лаб7 — webcam
python3 lab7_main.py --webcam
```

## Залежності по лабах

| Лаб | Нові залежності |
|-----|-----------------|
| 1–4 | `opencv-python`, `Pillow`, `numpy`, `matplotlib` |
| 5 | + `scikit-learn`, `torch`, `torchvision` |
| 6 | + `ultralytics` (YOLOv8) |
| 7 | нічого нового |
