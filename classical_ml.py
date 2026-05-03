"""
Класичне машинне навчання для класифікації зображень.
Лабораторна робота №5 — Тиждень 9.

Алгоритми: SVM, Random Forest, k-NN.
Ознаки: HOG (reuse з features.py лаб3).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Лаб3 — повторне використання HOG
from features import compute_hog

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ── Витяг HOG-ознак ───────────────────────────────────────────────────────────

TARGET_SIZE = (128, 128)


def extract_hog_features(image: np.ndarray, target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Витягує HOG-вектор ознак із зображення.
    Змінює розмір до target_size перед обчисленням.
    Повертає 1D float32 вектор.
    """
    resized = cv2.resize(image, target_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    descriptor, _ = compute_hog(gray)
    return descriptor.astype(np.float32)


# ── Завантаження датасету ─────────────────────────────────────────────────────

LABEL_MAP: Dict[str, int] = {}


def build_dataset(
    data_dir: str,
    target_size: Tuple[int, int] = TARGET_SIZE,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Завантажує зображення з підпапок data_dir (кожна підпапка = клас).
    Витягує HOG-ознаки та ділить на train/test (80/20).

    Повертає: (X_train, X_test, y_train, y_test, label_map).
    label_map: {'aircraft': 0, 'bird': 1, ...}
    """
    data_path = Path(data_dir)
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"Не знайдено підпапок класів у {data_dir}")

    label_map = {d.name: i for i, d in enumerate(class_dirs)}
    X, y = [], []

    for class_dir in class_dirs:
        label = label_map[class_dir.name]
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
        if not images:
            continue
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            features = extract_hog_features(img, target_size)
            X.append(features)
            y.append(label)

    if not X:
        raise ValueError(f"Не знайдено зображень у {data_dir}")

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr
    )
    return X_train, X_test, y_train, y_test, label_map


def build_synthetic_dataset(
    n_per_class: int = 80,
    target_size: Tuple[int, int] = TARGET_SIZE,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Генерує синтетичний датасет для демонстрації, коли реальних даних нема.
    Три класи: aircraft (горизонтальні лінії), helicopter (кола), bird (V-форма).
    """
    label_map = {"aircraft": 0, "bird": 1, "helicopter": 2}
    X, y = [], []
    rng = np.random.default_rng(random_state)

    for _ in range(n_per_class):
        img = _make_aircraft_image(target_size, rng)
        X.append(extract_hog_features(img, target_size))
        y.append(0)

    for _ in range(n_per_class):
        img = _make_bird_image(target_size, rng)
        X.append(extract_hog_features(img, target_size))
        y.append(1)

    for _ in range(n_per_class):
        img = _make_helicopter_image(target_size, rng)
        X.append(extract_hog_features(img, target_size))
        y.append(2)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr
    )
    return X_train, X_test, y_train, y_test, label_map


def _make_aircraft_image(size: Tuple[int, int], rng) -> np.ndarray:
    h, w = size
    img = np.ones((h, w, 3), dtype=np.uint8) * int(rng.integers(180, 220))
    cx, cy = w // 2 + int(rng.integers(-10, 10)), h // 2 + int(rng.integers(-10, 10))
    cv2.rectangle(img, (cx - 40, cy - 6), (cx + 40, cy + 6), (60, 60, 60), -1)
    wing_pts = np.array([[cx - 10, cy], [cx + 10, cy], [cx, cy - 30]], np.int32)
    cv2.fillPoly(img, [wing_pts], (80, 80, 80))
    noise = rng.integers(-15, 15, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _make_bird_image(size: Tuple[int, int], rng) -> np.ndarray:
    h, w = size
    img = np.ones((h, w, 3), dtype=np.uint8) * int(rng.integers(180, 220))
    cx, cy = w // 2 + int(rng.integers(-10, 10)), h // 2 + int(rng.integers(-5, 5))
    span = int(rng.integers(25, 45))
    cv2.line(img, (cx - span, cy - 10), (cx, cy), (50, 50, 50), 3)
    cv2.line(img, (cx, cy), (cx + span, cy - 10), (50, 50, 50), 3)
    cv2.circle(img, (cx, cy - 2), 4, (40, 40, 40), -1)
    noise = rng.integers(-15, 15, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _make_helicopter_image(size: Tuple[int, int], rng) -> np.ndarray:
    h, w = size
    img = np.ones((h, w, 3), dtype=np.uint8) * int(rng.integers(180, 220))
    cx, cy = w // 2 + int(rng.integers(-10, 10)), h // 2 + int(rng.integers(-5, 5))
    cv2.ellipse(img, (cx, cy), (25, 12), 0, 0, 360, (60, 60, 60), -1)
    rotor_len = int(rng.integers(30, 45))
    cv2.line(img, (cx - rotor_len, cy - 18), (cx + rotor_len, cy - 18), (50, 50, 50), 3)
    cv2.line(img, (cx, cy - 18), (cx, cy - 18), (50, 50, 50), 3)
    cv2.line(img, (cx + 20, cy + 12), (cx + 30, cy + 20), (55, 55, 55), 2)
    noise = rng.integers(-15, 15, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


# ── Моделі ───────────────────────────────────────────────────────────────────

def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
) -> SVC:
    """Тренує SVM-класифікатор (RBF-ядро за замовчуванням)."""
    model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
) -> RandomForestClassifier:
    """Тренує Random Forest класифікатор."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int = 5,
) -> KNeighborsClassifier:
    """Тренує k-NN класифікатор."""
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


# ── Оцінка ───────────────────────────────────────────────────────────────────

def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_map: Optional[Dict[str, int]] = None,
) -> Dict:
    """
    Обчислює accuracy, classification_report та confusion_matrix.
    Повертає dict з ключами: accuracy, report, confusion_matrix.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    target_names = None
    if label_map:
        idx_to_name = {v: k for k, v in label_map.items()}
        target_names = [idx_to_name[i] for i in sorted(idx_to_name)]

    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


# ── Порівняння класифікаторів ─────────────────────────────────────────────────

def compare_classifiers(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_map: Dict[str, int],
    save_path: Optional[str] = None,
    show: bool = False,
) -> Dict[str, float]:
    """
    Тренує SVM, RF, k-NN і порівнює їх точність.
    Малює bar chart + confusion matrices.
    Повертає dict {name: accuracy}.
    """
    models = {
        "SVM (RBF)": train_svm(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
        "k-NN (k=5)": train_knn(X_train, y_train),
    }

    results = {}
    cms = {}
    for name, model in models.items():
        metrics = evaluate(model, X_test, y_test, label_map)
        results[name] = metrics["accuracy"]
        cms[name] = metrics["confusion_matrix"]

    # Малюємо порівняльний графік
    idx_to_name = {v: k for k, v in label_map.items()}
    class_names = [idx_to_name[i] for i in sorted(idx_to_name)]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # Bar chart
    ax_bar = axes[0]
    names = list(results.keys())
    accs = [results[n] * 100 for n in names]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    bars = ax_bar.bar(names, accs, color=colors, edgecolor="white")
    ax_bar.set_ylim(0, 110)
    ax_bar.set_ylabel("Точність, %")
    ax_bar.set_title("Порівняння класифікаторів")
    for bar, acc in zip(bars, accs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{acc:.1f}%", ha="center", va="bottom", fontsize=9)
    ax_bar.tick_params(axis="x", rotation=15)

    # Confusion matrices
    for ax, (name, cm) in zip(axes[1:], cms.items()):
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"CM: {name}", fontsize=8)
        ax.set_xlabel("Передбачено", fontsize=7)
        ax.set_ylabel("Справжнє", fontsize=7)
        ticks = range(len(class_names))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(class_names, rotation=30, fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.suptitle("Лаб5 — Класичне ML: SVM / Random Forest / k-NN", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    return results
