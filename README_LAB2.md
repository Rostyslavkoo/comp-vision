# Лабораторна робота №2 — Фільтрація та сегментація

**Варіант 1:** Розробка системи розпізнавання літаючих об'єктів.

## Зміст виконання

### Тиждень 3: Фільтрація зображень (`filtering.py`)

| Функція | Метод | Призначення |
|---------|-------|-------------|
| `apply_gaussian` | `cv2.GaussianBlur` | Розмиття Гауса — зглажує високочастотний шум |
| `apply_median` | `cv2.medianBlur` | Медіанний фільтр — ефективний проти salt-pepper шуму |
| `apply_bilateral` | `cv2.bilateralFilter` | Зберігає краї, прибирає шум |
| `sharpen_unsharp_mask` | Unsharp mask | Підвищення різкості: `img + amount*(img - blurred)` |
| `sharpen_laplacian` | Лапласіан | Підсилення країв через другу похідну |
| `add_synthetic_noise` | NumPy random | Додає гаусівський або salt-pepper шум для тестів |
| `compare_filters` | matplotlib | Grid-порівняння всіх фільтрів на одному зображенні |

### Тиждень 4: Сегментація зображень (`segmentation.py`)

| Функція | Метод | Призначення |
|---------|-------|-------------|
| `threshold_binary` | `cv2.threshold` BINARY | Проста бінаризація за фіксованим порогом |
| `threshold_otsu` | `THRESH_OTSU` | Автоматичний поріг Otsu за гістограмою |
| `segment_watershed` | `cv2.watershed` | Watershed для розділення торкаючихся об'єктів |
| `segment_grabcut` | `cv2.grabCut` | GrabCut із прямокутником ROI — відокремлення переднього плану |
| `compare_segmentation` | matplotlib | Grid-порівняння всіх методів сегментації |

## Зв'язок з Лабораторною №1

`lab2_main.py` імпортує та використовує:
- `load_image` з `image_loader.py` — завантаження зображення
- `print_characteristics` з `image_loader.py` — виведення характеристик перед обробкою

## Файли проєкту

| Файл | Опис |
|------|------|
| `filtering.py` | Модуль фільтрації (шумозаглушення, підвищення різкості) |
| `segmentation.py` | Модуль сегментації (порогова, Otsu, Watershed, GrabCut) |
| `lab2_main.py` | Головний скрипт лабораторної №2 |

## Запуск

1. Встановити залежності (ті ж, що і для лаб1):
   ```bash
   pip install -r requirements.txt
   ```

2. Запустити з власним зображенням:
   ```bash
   python lab2_main.py шлях/до/зображення.png
   ```

3. Без графічних вікон (лише збереження у файли):
   ```bash
   python lab2_main.py шлях/до/зображення.png --no-gui
   ```

4. Без аргументів — автоматично створюється синтетичне демо-зображення літака:
   ```bash
   python lab2_main.py --no-gui
   ```

## Структура результатів

```
lab2_output/
├── demo_aircraft.png              — синтетичне зображення (якщо запуск без аргументів)
├── filters/
│   ├── gaussian_blur.png
│   ├── median_blur.png
│   ├── bilateral_filter.png
│   ├── sharpen_unsharp.png
│   ├── sharpen_laplacian.png
│   ├── noisy_gaussian.png
│   ├── noisy_salt_pepper.png
│   └── filters_comparison.png    — зведений grid порівняння
└── segmentation/
    ├── threshold_binary.png
    ├── threshold_otsu.png
    ├── watershed.png
    ├── grabcut.png
    └── segmentation_comparison.png — зведений grid порівняння
```

## Залежності

Нових залежностей не додано — використовується OpenCV, NumPy і matplotlib з `requirements.txt`.
