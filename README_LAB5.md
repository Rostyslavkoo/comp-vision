# Лабораторна робота №5 — Класифікація зображень: класичне ML та CNN

**Варіант 1:** Розробка системи розпізнавання літаючих об'єктів.

## Зміст виконання

### Тиждень 9: Класичне машинне навчання (`classical_ml.py`)

| Функція | Метод | Призначення |
|---------|-------|-------------|
| `extract_hog_features` | HOG з `features.py` (лаб3) | HOG-вектор ознак 128×128 зображення |
| `build_dataset` | sklearn `train_test_split` | Завантаження зображень з підпапок, 80/20 split |
| `build_synthetic_dataset` | OpenCV рисування | Синтетичний датасет для демо без реальних даних |
| `train_svm` | `SVC(kernel='rbf')` | Тренування SVM-класифікатора |
| `train_random_forest` | `RandomForestClassifier(100)` | Тренування Random Forest |
| `train_knn` | `KNeighborsClassifier(k=5)` | Тренування k-NN |
| `evaluate` | accuracy, F1, confusion matrix | Оцінка моделі на тестовій вибірці |
| `compare_classifiers` | matplotlib | Bar chart + 3 confusion matrices |

### Тиждень 10: CNN з нуля (`cnn_model.py`)

| Клас / Функція | Опис |
|----------------|------|
| `FlyingObjectCNN` | CNN: Conv(32)→ReLU→MaxPool → Conv(64)→ReLU→MaxPool → Conv(128)→ReLU→MaxPool → FC(256)→Dropout(0.4)→FC(N) |
| `FlyingObjectDataset` | PyTorch Dataset: зображення з підпапок → сіре 64×64, нормалізація [0,1] |
| `SyntheticFlyingDataset` | Синтетичний Dataset для демонстрації |
| `train_cnn` | Навчання: Adam + StepLR, CrossEntropyLoss; повертає модель та history |
| `evaluate_cnn` | Точність на test DataLoader |
| `plot_training_history` | Графіки loss та accuracy по епохах |

## Зв'язок з попередніми лабораторними

`lab5_main.py` імпортує:
- `load_image`, `print_characteristics` з `image_loader.py` — **лаб1**
- `compute_hog` з `features.py` — **лаб3** (HOG-ознаки для SVM/RF/k-NN)

Ключова інтеграція — `extract_hog_features`:
1. `compute_hog` (лаб3) → 1D вектор HOG-ознак
2. `SVM / RF / k-NN` (лаб5) → класифікація на основі цих ознак

## Структура датасету (для реального запуску)

```
data/
└── raw/
    ├── aircraft/      — зображення літаків (PNG/JPG)
    ├── helicopter/    — зображення гелікоптерів
    └── bird/          — зображення птахів
```

Мінімальна рекомендована кількість: **50+ зображень на клас**.

### Джерела датасетів (Kaggle)

```bash
# Потрібен акаунт Kaggle + ~/.kaggle/kaggle.json
pip install kaggle
kaggle datasets download -d saurabhshahane/military-aircraft-detection
kaggle datasets download -d gpiosenka/100-bird-species
```

або вручну завантажити з:
- https://www.kaggle.com/datasets/saurabhshahane/military-aircraft-detection
- https://www.kaggle.com/datasets/gpiosenka/100-bird-species

## Файли проєкту

| Файл | Опис |
|------|------|
| `classical_ml.py` | SVM, Random Forest, k-NN + витяг HOG-ознак |
| `cnn_model.py` | CNN архітектура, тренування, оцінка (PyTorch) |
| `lab5_main.py` | Головний скрипт лабораторної №5 |
| `data/README.md` | Інструкція зі збору датасету |

## Запуск

1. Встановити залежності:
   ```bash
   pip install -r requirements.txt
   ```

2. Синтетичне демо (без реальних даних):
   ```bash
   python lab5_main.py --no-gui
   ```

3. Запуск із реальним датасетом:
   ```bash
   python lab5_main.py data/raw --no-gui
   ```

4. Зі збільшеною кількістю епох:
   ```bash
   python lab5_main.py data/raw --no-gui --epochs 50
   ```

## Структура результатів

```
lab5_output/
├── classifiers_comparison.png   — bar chart точності + 3 confusion matrices
├── cnn_training_history.png     — графіки loss/accuracy по епохах
└── cnn_weights.pth              — збережені ваги CNN (PyTorch state_dict)
```

## Залежності

| Пакет | Версія | Призначення |
|-------|--------|-------------|
| `scikit-learn` | ≥ 1.3 | SVM, Random Forest, k-NN, метрики |
| `torch` | ≥ 2.0 | CNN, тренування, DataLoader |
| `torchvision` | ≥ 0.15 | (резерв для лаб6 — pretrained моделі) |

## Архітектура CNN

```
Вхід: (B, 1, 64, 64)
  └─ Conv2d(1→32, 3×3, pad=1) → ReLU → MaxPool(2×2)   # → (B, 32, 32, 32)
  └─ Conv2d(32→64, 3×3, pad=1) → ReLU → MaxPool(2×2)  # → (B, 64, 16, 16)
  └─ Conv2d(64→128, 3×3, pad=1) → ReLU → MaxPool(2×2) # → (B, 128, 8, 8)
  └─ Flatten → Linear(8192→256) → ReLU → Dropout(0.4)
  └─ Linear(256→num_classes)
Вихід: logits розміром (B, num_classes)
```

Оптимізатор: **Adam** (lr=0.001, weight_decay=1e-4)  
Scheduler: **StepLR** (step_size=8, gamma=0.5)  
Loss: **CrossEntropyLoss**
