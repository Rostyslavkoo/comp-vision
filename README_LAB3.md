# Лабораторна робота №3 — Витяг ознак та обробка відео

**Варіант 1:** Розробка системи розпізнавання літаючих об'єктів.

## Зміст виконання

### Тиждень 5: Витяг ознак (`features.py`)

| Функція | Метод | Призначення |
|---------|-------|-------------|
| `find_contours` | `cv2.findContours` + Otsu | Знаходить контури об'єктів після бінаризації |
| `draw_contours` | `cv2.drawContours` | Візуалізує контури на копії зображення |
| `get_contour_stats` | `cv2.contourArea`, `arcLength` | Кількість, площі та периметри контурів |
| `detect_sift` | `cv2.SIFT_create` | SIFT-ключові точки та 128-мірні дескриптори |
| `draw_keypoints_sift` | `cv2.drawKeypoints` RICH | Ключові точки з масштабом та орієнтацією |
| `detect_orb` | `cv2.ORB_create` | ORB-ключові точки та бінарні дескриптори |
| `draw_keypoints_orb` | `cv2.drawKeypoints` | Ключові точки ORB |
| `compute_hog` | `cv2.HOGDescriptor` | HOG-вектор ознак + візуалізація градієнтів |
| `compare_descriptors` | matplotlib | Grid-порівняння: контури / SIFT / ORB / HOG |

> **Примітка щодо SURF:** алгоритм SURF відсутній у вільній версії `opencv-python`
> через патентні обмеження (US Patent 6,711,293 — термін дії закінчився у 2023,
> але підтримку в OpenCV не відновлено). Використовуються SIFT та ORB як альтернативи:
> SIFT — точніший, ORB — швидший і безкоштовний аналог.

### Тиждень 6: Обробка відео (`video_processing.py`)

| Функція | Метод | Призначення |
|---------|-------|-------------|
| `make_synthetic_video` | `cv2.VideoWriter` | Генерує MP4 із рухомим літаком (коли немає реального відео) |
| `optical_flow_lk` | `cv2.calcOpticalFlowPyrLK` | Lucas-Kanade sparse flow: відстеження Shi-Tomasi точок |
| `optical_flow_dense` | `cv2.calcOpticalFlowFarneback` | Щільний потік: вектор для кожного пікселя, HSV-візуалізація |
| `background_subtraction` | `MOG2` / `KNN` | Виявлення рухомих об'єктів відніманням фону |
| `demo_optical_flow_static` | matplotlib | Статична демонстрація Farneback на 2 кадрах |
| `demo_background_subtraction_static` | matplotlib | Статична демонстрація MOG2 на 4 кадрах |

## Зв'язок з попередніми лабораторними

`lab3_main.py` імпортує:
- `load_image`, `print_characteristics` з `image_loader.py` — **лаб1**
- `threshold_otsu` з `segmentation.py` — **лаб2** (використовується в `find_contours` для бінаризації)

## Файли проєкту

| Файл | Опис |
|------|------|
| `features.py` | Витяг ознак: контури, SIFT, ORB, HOG |
| `video_processing.py` | Оптичний потік, віднімання фону |
| `lab3_main.py` | Головний скрипт лабораторної №3 |

## Запуск

1. Встановити залежності:
   ```bash
   pip install -r requirements.txt
   ```

2. Запустити з власним зображенням:
   ```bash
   python lab3_main.py шлях/до/зображення.png
   ```

3. Запустити з власним зображенням і відео:
   ```bash
   python lab3_main.py шлях/до/зображення.png --video шлях/до/відео.mp4
   ```

4. Без графічних вікон (лише збереження у файли):
   ```bash
   python lab3_main.py шлях/до/зображення.png --no-gui
   ```

5. Без аргументів — автоматично створюється синтетичне демо:
   ```bash
   python lab3_main.py --no-gui
   ```

## Структура результатів

```
lab3_output/
├── demo_aircraft.png              — синтетичне зображення (якщо без аргументів)
├── contours.png                   — контури об'єктів
├── hog_visualization.png          — візуалізація HOG-градієнтів
├── descriptors.png                — порівняльний grid (контури/SIFT/ORB/HOG)
├── optical_flow_demo.png          — демо оптичного потоку (2 кадри + Farneback)
├── bg_subtraction_demo.png        — демо MOG2 (4 кадри + маски)
├── synthetic_video.mp4            — згенероване тестове відео
└── video/
    ├── lk_flow_*.png              — кадри Lucas-Kanade потоку
    ├── dense_flow_*.png           — кадри щільного Farneback потоку
    ├── bg_sub_mog2_*.png          — кадри MOG2 (оригінал + маска)
    └── bg_sub_knn_*.png           — кадри KNN (оригінал + маска)
```

## Залежності

Нових залежностей не додано — використовується OpenCV, NumPy і matplotlib з `requirements.txt`.
