#!/usr/bin/env python3
"""
Лабораторна робота №6 (Тиждень 11):
Використання pretrained моделі YOLO для детекції об'єктів.

Скрипт:
- приймає шлях до зображення/папки зображень/відео
- запускає inference на pretrained YOLO (Ultralytics)
- зберігає анотовані зображення/відео та текстовий лог детекцій
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
SUPPORTED_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    xyxy: Tuple[float, float, float, float]


def _iter_inputs(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        return
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in (SUPPORTED_IMAGE_EXTS + SUPPORTED_VIDEO_EXTS):
            yield p


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED_IMAGE_EXTS


def _is_video(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED_VIDEO_EXTS


def run_yolo_detect(
    source: str,
    output_dir: str = "lab6_output",
    model: str = "yolov8n.pt",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    device: Optional[str] = None,
    max_det: int = 300,
) -> Path:
    """
    Запустити YOLO на source і зберегти результати у output_dir.
    Повертає шлях до директорії з конкретним запуском.
    """
    from ultralytics import YOLO

    src = Path(source)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        yolo = YOLO(model)
    except Exception as e:
        # Найчастіше тут падає завантаження ваг, якщо середовище offline.
        raise RuntimeError(
            "Не вдалося ініціалізувати YOLO модель. "
            "Якщо середовище без доступу до інтернету — попередньо завантажте ваги "
            "(наприклад `yolov8n.pt`) і передайте локальний шлях через `--model`.\n"
            f"Оригінальна помилка: {e}"
        ) from e

    # Ultralytics сам створює структуру runs/detect/..., але ми хочемо зберігати у своєму output_dir.
    # Тому даємо project/name.
    run_name = "detect"
    results = yolo.predict(
        source=str(src),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        max_det=max_det,
        project=str(out_root),
        name=run_name,
        save=True,
        save_txt=False,
        verbose=False,
    )

    # Директорія, куди Ultralytics зберіг результати
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    detections_log: List[dict] = []

    for r in results:
        item = {"path": str(r.path), "detections": []}
        names = r.names or {}
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                cls_id = int(b.cls.item()) if b.cls is not None else -1
                cls_name = str(names.get(cls_id, cls_id))
                conf_v = float(b.conf.item()) if b.conf is not None else 0.0
                xyxy = tuple(float(x) for x in b.xyxy[0].tolist())
                det = Detection(cls_id=cls_id, cls_name=cls_name, conf=conf_v, xyxy=xyxy)
                item["detections"].append(asdict(det))
        detections_log.append(item)

    (run_dir / "detections.json").write_text(json.dumps(detections_log, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab6: YOLO pretrained detection (Ultralytics)")
    parser.add_argument("source", help="Шлях до зображення/папки/відео")
    parser.add_argument("--output-dir", default="lab6_output", help="Папка для збереження результатів")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model (наприклад yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default=None, help="Device: 'cpu', '0', '0,1' ... (опційно)")
    args = parser.parse_args()

    run_dir = run_yolo_detect(
        source=args.source,
        output_dir=args.output_dir,
        model=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )
    print(f"Готово. Результати: {run_dir}")
    print(f"- Анотовані файли: {run_dir}")
    print(f"- Лог детекцій: {run_dir / 'detections.json'}")


if __name__ == "__main__":
    main()

