## Лабораторна робота №6 — Pretrained Models & Data Preparation

**Варіант 1:** система розпізнавання літаючих об'єктів (літаки, гелікоптери, птахи).

Ця лабораторна складається з:
- **Тиждень 11:** використання pretrained моделі для детекції об'єктів (YOLO).
- **Тиждень 12:** аугментація датасету (обертання, зміна яскравості, додавання шуму).

---

## Встановлення залежностей

```bash
pip install -r requirements.txt
```

> Примітка: `ultralytics` при першому запуску може автоматично завантажувати ваги моделі (наприклад `yolov8n.pt`).

---

## Тиждень 11 — Детекція об'єктів (YOLO pretrained)

Скрипт: `lab6_yolo_detect.py`

### 1) Детекція на одному зображенні

```bash
python lab6_yolo_detect.py path/to/image.jpg
```

### 2) Детекція на папці з зображеннями/відео

```bash
python lab6_yolo_detect.py path/to/folder
```

### 3) Налаштування порогів та моделі

```bash
python lab6_yolo_detect.py path/to/image.jpg --model yolov8n.pt --conf 0.25 --iou 0.45 --imgsz 640
```

### Результати

За замовчуванням зберігаються у `lab6_output/detect/`:
- анотовані файли (bounding boxes + labels)
- `detections.json` — лог детекцій (класи, confidence, координати)

### Якщо середовище offline

Якщо `ultralytics` не може завантажити ваги `yolov8n.pt`, завантажте їх один раз вручну і передайте локальний шлях:

```bash
python lab6_yolo_detect.py path/to/image.jpg --model /full/path/to/yolov8n.pt
```

---

## Тиждень 11 — Альтернатива: ResNet (torchvision Faster R-CNN)

Скрипт: `lab6_torchvision_detect.py`

```bash
python lab6_torchvision_detect.py path/to/image.jpg --score 0.5
```

Результати зберігаються у `lab6_output_torchvision/` + файл `detections.json`.

---

## Тиждень 12 — Аугментація датасету

Скрипт: `lab6_augment.py`

### Приклад

```bash
python lab6_augment.py path/to/input_images --output-dir lab6_augmented --copies 3
```

### Параметри
- `--rotate` — максимальний кут повороту (градуси)
- `--brightness` — межа зміни яскравості (0..1)
- `--contrast` — межа зміни контрасту (0..1)
- `--noise-sigma` — максимальна sigma для Gaussian noise

### Результати

У вихідній папці зберігаються:
- аугментовані зображення з суфіксом `_augN`
- `manifest.json` — відповідність input→output та параметри аугментації

---

## Рекомендована структура даних (для наступних лабораторних)

Наприклад:
- `data/raw/` — оригінальні зображення
- `data/aug/` — аугментовані

Тоді:

```bash
python lab6_augment.py data/raw --output-dir data/aug --copies 5
python lab6_yolo_detect.py data/raw
```

