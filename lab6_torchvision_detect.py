#!/usr/bin/env python3
"""
Лабораторна робота №6 (Тиждень 11) — альтернативний варіант:
pretrained детектор на базі ResNet з torchvision: Faster R-CNN (COCO).

Переваги:
- відповідає вимозі "ResNet" (backbone)
- не залежить від Ultralytics/YOLO (але все одно може потребувати завантаження ваг)

Скрипт:
- приймає одне зображення або папку зображень
- зберігає анотовані картинки та JSON з детекціями
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass
class Detection:
    label_id: int
    label_name: str
    score: float
    xyxy: Tuple[float, float, float, float]


def _list_images(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
        return [path]
    if not path.is_dir():
        return []
    out: List[Path] = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            out.append(p)
    return out


def _load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не вдалося прочитати: {path}")
    return img


def _draw_detections(img_bgr: np.ndarray, dets: List[Detection]) -> np.ndarray:
    out = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(d.xyxy[0]), int(d.xyxy[1]), int(d.xyxy[2]), int(d.xyxy[3]))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{d.label_name} {d.score:.2f}"
        cv2.putText(out, text, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def run_torchvision_detect(
    source: str,
    output_dir: str = "lab6_output_torchvision",
    score_thresh: float = 0.5,
    device: Optional[str] = None,
) -> Path:
    src = Path(source)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    preprocess = weights.transforms()
    categories = weights.meta.get("categories", None)

    images = _list_images(src)
    manifest: List[dict] = []

    with torch.no_grad():
        for img_path in images:
            bgr = _load_bgr(img_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            x = preprocess(torch.from_numpy(rgb).permute(2, 0, 1))  # CHW, float, normalized
            x = x.to(dev)

            preds = model([x])[0]
            boxes = preds["boxes"].detach().cpu().numpy()
            labels = preds["labels"].detach().cpu().numpy()
            scores = preds["scores"].detach().cpu().numpy()

            dets: List[Detection] = []
            for box, lab, score in zip(boxes, labels, scores):
                if float(score) < score_thresh:
                    continue
                name = str(lab)
                if categories and 0 <= int(lab) < len(categories):
                    name = categories[int(lab)]
                dets.append(
                    Detection(
                        label_id=int(lab),
                        label_name=name,
                        score=float(score),
                        xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    )
                )

            annotated = _draw_detections(bgr, dets)
            out_path = out_dir / f"{img_path.stem}_det{img_path.suffix.lower()}"
            cv2.imwrite(str(out_path), annotated)

            manifest.append(
                {
                    "input": str(img_path),
                    "output": str(out_path),
                    "detections": [asdict(d) for d in dets],
                }
            )

    (out_dir / "detections.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab6: torchvision Faster R-CNN (ResNet) detection")
    parser.add_argument("source", help="Шлях до зображення або папки з зображеннями")
    parser.add_argument("--output-dir", default="lab6_output_torchvision", help="Вихідна папка")
    parser.add_argument("--score", type=float, default=0.5, help="Поріг score для детекцій")
    parser.add_argument("--device", default=None, help="cpu або cuda (опційно)")
    args = parser.parse_args()

    out_dir = run_torchvision_detect(
        source=args.source,
        output_dir=args.output_dir,
        score_thresh=args.score,
        device=args.device,
    )
    print(f"Готово. Результати: {out_dir}")
    print(f"Лог: {out_dir / 'detections.json'}")


if __name__ == "__main__":
    main()

