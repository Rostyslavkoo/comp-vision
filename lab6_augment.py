#!/usr/bin/env python3
"""
Лабораторна робота №6 (Тиждень 12):
Аугментація датасету (обертання, зміна яскравості, додавання шуму).

Скрипт:
- приймає вхідну папку зображень
- генерує N аугментованих версій кожного зображення
- зберігає результат у вихідну папку (та метадані JSON)
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass
class AugmentConfig:
    rotate_limit: int = 25
    brightness_limit: float = 0.25
    contrast_limit: float = 0.20
    noise_sigma: float = 12.0


def _list_images(dir_path: Path) -> List[Path]:
    if not dir_path.is_dir():
        return []
    out: List[Path] = []
    for p in sorted(dir_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            out.append(p)
    return out


def _read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не вдалося прочитати зображення: {path}")
    return img


def _rotate(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _brightness_contrast(image: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    """
    brightness in [-1..1] approx, contrast in [-1..1]
    Реалізація через alpha/beta: new = alpha*img + beta.
    """
    alpha = 1.0 + contrast
    beta = 255.0 * brightness
    out = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return out


def _add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
    out = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def augment_image(
    image: np.ndarray,
    cfg: AugmentConfig,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Зробити одну випадкову аугментацію та повернути (image, metadata).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    angle = random.uniform(-cfg.rotate_limit, cfg.rotate_limit)
    brightness = random.uniform(-cfg.brightness_limit, cfg.brightness_limit)
    contrast = random.uniform(-cfg.contrast_limit, cfg.contrast_limit)
    sigma = random.uniform(0.0, cfg.noise_sigma)

    out = _rotate(image, angle)
    out = _brightness_contrast(out, brightness=brightness, contrast=contrast)
    out = _add_gaussian_noise(out, sigma=sigma)

    meta = {
        "rotate_deg": round(angle, 3),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "noise_sigma": round(sigma, 3),
    }
    return out, meta


def run_augmentation(
    input_dir: str,
    output_dir: str = "lab6_augmented",
    copies_per_image: int = 3,
    cfg: Optional[AugmentConfig] = None,
) -> Path:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = cfg or AugmentConfig()
    images = _list_images(in_dir)

    manifest: List[dict] = []
    for img_path in images:
        img = _read_bgr(img_path)
        stem = img_path.stem
        for i in range(copies_per_image):
            aug_img, meta = augment_image(img, cfg=cfg)
            out_path = out_dir / f"{stem}_aug{i+1}{img_path.suffix.lower()}"
            cv2.imwrite(str(out_path), aug_img)
            manifest.append(
                {
                    "input": str(img_path),
                    "output": str(out_path),
                    "meta": meta,
                }
            )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab6: dataset augmentation (rotate/brightness/noise)")
    parser.add_argument("input_dir", help="Вхідна папка з зображеннями")
    parser.add_argument("--output-dir", default="lab6_augmented", help="Вихідна папка")
    parser.add_argument("--copies", type=int, default=3, help="Кількість аугментованих копій на 1 зображення")
    parser.add_argument("--rotate", type=int, default=25, help="Максимальний кут повороту (градуси)")
    parser.add_argument("--brightness", type=float, default=0.25, help="Межа зміни яскравості (0..1)")
    parser.add_argument("--contrast", type=float, default=0.20, help="Межа зміни контрасту (0..1)")
    parser.add_argument("--noise-sigma", type=float, default=12.0, help="Макс sigma для Gaussian noise")
    args = parser.parse_args()

    cfg = AugmentConfig(
        rotate_limit=args.rotate,
        brightness_limit=args.brightness,
        contrast_limit=args.contrast,
        noise_sigma=args.noise_sigma,
    )
    out_dir = run_augmentation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        copies_per_image=args.copies,
        cfg=cfg,
    )
    print(f"Готово. Аугментовані дані: {out_dir}")
    print(f"Маніфест: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

