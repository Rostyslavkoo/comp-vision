# Лабораторна робота №4 — Геометричні перетворення та морфологія

**Варіант 1:** Розробка системи розпізнавання літаючих об'єктів.

## Зміст виконання

### Тиждень 7: Геометричні перетворення (`geometry.py`)

| Функція | Метод | Призначення |
|---------|-------|-------------|
| `scale_image` | `cv2.resize` (fx, fy) | Масштабування з коефіцієнтами по осях |
| `scale_to_size` | `cv2.resize` (w, h) | Масштабування до точного розміру |
| `rotate_image` | `cv2.getRotationMatrix2D` + `warpAffine` | Поворот навколо центру; expand=True розширює полотно |
| `perspective_transform` | `cv2.getPerspectiveTransform` + `warpPerspective` | Довільна перспектива (4 точки → 4 точки) |
| `perspective_transform_auto` | — | Автоматична трапецієвидна перспектива |
| `affine_transform` | `cv2.getAffineTransform` + `warpAffine` | Афінне перетворення (3 точки → 3 точки) |
| `compare_transformations` | matplotlib | Grid 2×4: всі перетворення разом |

### Тиждень 8: Морфологічні операції (`morphology.py`)

| Функція | Метод | Призначення |
|---------|-------|-------------|
| `erode` | `cv2.erode` | Ерозія — стискає світлі об'єкти, видаляє шум |
| `dilate` | `cv2.dilate` | Дилатація — розширює об'єкти, заповнює прогалини |
| `opening` | `cv2.MORPH_OPEN` | Відкриття = ерозія → дилатація |
| `closing` | `cv2.MORPH_CLOSE` | Закриття = дилатація → ерозія |
| `morphological_gradient` | `cv2.MORPH_GRADIENT` | Градієнт = дилатація − ерозія (контури) |
| `top_hat` | `cv2.MORPH_TOPHAT` | Виділяє дрібні яскраві деталі |
| `black_hat` | `cv2.MORPH_BLACKHAT` | Виділяє дрібні темні деталі |
| `improve_segmentation` | opening(3) + closing(5) | Очищення бінарних масок |
| `segment_and_improve` | threshold\_otsu + improve | Повний пайплайн: сегментація → морфологія |
| `compare_morphology` | matplotlib | Grid 2×4: всі операції на масці |
| `compare_segmentation_improvement` | matplotlib | До/після: маска лаб2 + покращена лаб4 |

## Зв'язок з попередніми лабораторними

`lab4_main.py` імпортує:
- `load_image`, `print_characteristics` з `image_loader.py` — **лаб1**
- `threshold_otsu` з `segmentation.py` — **лаб2** (бінарна маска для морфологічних операцій)

Ключова інтеграція — функція `segment_and_improve`:
1. `threshold_otsu` (лаб2) → сира бінарна маска
2. `opening(3) + closing(5)` (лаб4) → очищена маска без шуму і прогалин

## Файли проєкту

| Файл | Опис |
|------|------|
| `geometry.py` | Геометричні перетворення зображень |
| `morphology.py` | Морфологічні операції та покращення масок |
| `lab4_main.py` | Головний скрипт лабораторної №4 |

## Запуск

1. Встановити залежності:
   ```bash
   pip install -r requirements.txt
   ```

2. Запустити з власним зображенням:
   ```bash
   python lab4_main.py шлях/до/зображення.png
   ```

3. Без графічних вікон (лише збереження у файли):
   ```bash
   python lab4_main.py шлях/до/зображення.png --no-gui
   ```

4. Без аргументів — синтетичне демо:
   ```bash
   python lab4_main.py --no-gui
   ```

## Структура результатів

```
lab4_output/
├── demo_aircraft.png              — синтетичне зображення (якщо без аргументів)
├── scale_0.5x.png                 — зменшення вдвічі
├── scale_1.5x.png                 — збільшення в 1.5×
├── rotate_30.png                  — поворот 30°
├── rotate_90.png                  — поворот 90°
├── rotate_45_expand.png           — поворот 45° з розширенням полотна
├── perspective.png                — перспективне спотворення
├── affine.png                     — афінне перетворення
├── transformations_comparison.png — зведений grid (2×4)
├── erode_3x3.png
├── dilate_3x3.png
├── opening_3x3.png
├── closing_5x5.png
├── gradient_3x3.png
├── top_hat.png
├── black_hat.png
├── improved_mask.png
├── morphology_comparison.png      — зведений grid морфології (2×4)
└── segmentation_improvement.png   — інтеграція лаб2 + лаб4 (до/після)
```

## Залежності

Нових залежностей не додано — використовується OpenCV, NumPy і matplotlib з `requirements.txt`.
