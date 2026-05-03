"""
CNN-класифікатор з нуля на PyTorch.
Лабораторна робота №5 — Тиждень 10.

Архітектура: Conv32→ReLU→MaxPool → Conv64→ReLU→MaxPool → FC128→FC(num_classes).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split


# ── Модель ───────────────────────────────────────────────────────────────────

class FlyingObjectCNN(nn.Module):
    """
    Простий CNN для 3-класової класифікації літаючих об'єктів.
    Вхід: (B, 1, 64, 64) — сірі зображення.
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 64→32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 32→16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 16→8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── Dataset ───────────────────────────────────────────────────────────────────

IMG_SIZE = 64


class FlyingObjectDataset(Dataset):
    """
    Завантажує зображення з підпапок (кожна підпапка = клас).
    Перетворює на сіре 64×64, нормалізує до [0, 1].
    """

    def __init__(self, data_dir: str, img_size: int = IMG_SIZE) -> None:
        self.img_size = img_size
        self.samples: List[Tuple[str, int]] = []
        self.label_map: Dict[str, int] = {}

        data_path = Path(data_dir)
        class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
        self.label_map = {d.name: i for i, d in enumerate(class_dirs)}

        for class_dir in class_dirs:
            label = self.label_map[class_dir.name]
            for ext in ("*.jpg", "*.png", "*.jpeg"):
                for p in class_dir.glob(ext):
                    self.samples.append((str(p), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size))
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, label


class SyntheticFlyingDataset(Dataset):
    """
    Синтетичний датасет для демонстрації без реальних даних.
    Генерує прості зображення трьох класів: aircraft, bird, helicopter.
    """

    def __init__(self, n_per_class: int = 100, img_size: int = IMG_SIZE, seed: int = 42) -> None:
        self.img_size = img_size
        self.label_map = {"aircraft": 0, "bird": 1, "helicopter": 2}
        self.samples: List[Tuple[np.ndarray, int]] = []
        rng = np.random.default_rng(seed)

        generators = [
            (_gen_aircraft, 0),
            (_gen_bird, 1),
            (_gen_helicopter, 2),
        ]
        for gen_fn, label in generators:
            for _ in range(n_per_class):
                img = gen_fn(img_size, rng)
                self.samples.append((img, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.samples[idx]
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, label


def _gen_aircraft(size: int, rng) -> np.ndarray:
    img = np.full((size, size), rng.integers(180, 220), dtype=np.uint8)
    cx, cy = size // 2 + int(rng.integers(-8, 8)), size // 2 + int(rng.integers(-8, 8))
    cv2.rectangle(img, (cx - 20, cy - 3), (cx + 20, cy + 3), 50, -1)
    pts = np.array([[cx - 5, cy], [cx + 5, cy], [cx, cy - 15]], np.int32)
    cv2.fillPoly(img, [pts], 70)
    noise = rng.integers(-12, 12, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _gen_bird(size: int, rng) -> np.ndarray:
    img = np.full((size, size), rng.integers(180, 220), dtype=np.uint8)
    cx, cy = size // 2 + int(rng.integers(-8, 8)), size // 2 + int(rng.integers(-5, 5))
    span = int(rng.integers(12, 22))
    cv2.line(img, (cx - span, cy - 6), (cx, cy), 50, 2)
    cv2.line(img, (cx, cy), (cx + span, cy - 6), 50, 2)
    cv2.circle(img, (cx, cy - 1), 2, 40, -1)
    noise = rng.integers(-12, 12, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _gen_helicopter(size: int, rng) -> np.ndarray:
    img = np.full((size, size), rng.integers(180, 220), dtype=np.uint8)
    cx, cy = size // 2 + int(rng.integers(-8, 8)), size // 2 + int(rng.integers(-5, 5))
    cv2.ellipse(img, (cx, cy), (14, 6), 0, 0, 360, 55, -1)
    rotor = int(rng.integers(16, 24))
    cv2.line(img, (cx - rotor, cy - 10), (cx + rotor, cy - 10), 50, 2)
    cv2.line(img, (cx + 12, cy + 6), (cx + 18, cy + 12), 55, 1)
    noise = rng.integers(-12, 12, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ── Тренування ───────────────────────────────────────────────────────────────

def train_cnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int = 3,
    epochs: int = 20,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> Tuple["FlyingObjectCNN", Dict]:
    """
    Тренує FlyingObjectCNN.
    Повертає (trained_model, history).
    history: {'train_loss', 'val_loss', 'train_acc', 'val_acc'} — списки per epoch.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model = FlyingObjectCNN(num_classes=num_classes).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }

    for epoch in range(1, epochs + 1):
        # Тренування
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(dev), y_batch.to(dev)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(y_batch)
            t_correct += (logits.argmax(1) == y_batch).sum().item()
            t_total += len(y_batch)

        # Валідація
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(dev), y_batch.to(dev)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                v_loss += loss.item() * len(y_batch)
                v_correct += (logits.argmax(1) == y_batch).sum().item()
                v_total += len(y_batch)

        scheduler.step()

        history["train_loss"].append(t_loss / t_total)
        history["val_loss"].append(v_loss / v_total)
        history["train_acc"].append(t_correct / t_total)
        history["val_acc"].append(v_correct / v_total)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Епоха {epoch:3d}/{epochs} | "
                f"loss {history['train_loss'][-1]:.4f} / {history['val_loss'][-1]:.4f} | "
                f"acc {history['train_acc'][-1]:.3f} / {history['val_acc'][-1]:.3f}"
            )

    return model, history


# ── Оцінка CNN ───────────────────────────────────────────────────────────────

def evaluate_cnn(
    model: "FlyingObjectCNN",
    test_loader: DataLoader,
    device: Optional[str] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Оцінює модель на тестовому DataLoader.
    Повертає (accuracy, y_true, y_pred).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model.eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(dev)
            preds = model(X_batch).argmax(1).cpu().numpy()
            all_pred.extend(preds)
            all_true.extend(y_batch.numpy())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    acc = (y_true == y_pred).mean()
    return acc, y_true, y_pred


# ── Візуалізація ─────────────────────────────────────────────────────────────

def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Малює графіки loss та accuracy по епохах."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"], label="Val loss")
    axes[0].set_xlabel("Епоха")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Функція втрат")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], label="Train acc")
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], label="Val acc")
    axes[1].set_xlabel("Епоха")
    axes[1].set_ylabel("Точність, %")
    axes[1].set_title("Точність класифікації")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Лаб5 — Навчання CNN", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
