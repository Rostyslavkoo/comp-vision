"""
Обробка відео: оптичний потік (Lucas-Kanade та Farneback),
віднімання фону (MOG2, KNN).
Лабораторна робота №3 — Тиждень 6.
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ── Генератор синтетичного відео ──────────────────────────────────────────────

def make_synthetic_video(output_path: str, n_frames: int = 60) -> str:
    """
    Створює MP4 відео: синтетичний літак (прямокутник) рухається зліва направо
    на статичному небесному фоні. Використовується коли реального відео немає.
    """
    h, w = 240, 320
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 20, (w, h))

    rng = np.random.default_rng(0)
    for i in range(n_frames):
        # Фон — градієнт неба
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            val = int(180 - y * 0.4)
            frame[y, :] = [max(val - 20, 30), max(val - 10, 40), min(val + 30, 220)]

        # Рухомий літак
        x = int(i * w / n_frames)
        y_pos = h // 2 - 15
        # тіло
        cv2.rectangle(frame, (x, y_pos), (x + 60, y_pos + 20), (80, 80, 80), -1)
        # крило
        pts = np.array([[x + 15, y_pos + 10], [x + 35, y_pos - 15], [x + 50, y_pos + 10]], np.int32)
        cv2.fillPoly(frame, [pts], (100, 100, 100))

        # Легкий шум (робить фон менш ідеальним для тестування)
        noise = rng.integers(-5, 5, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        writer.write(frame)

    writer.release()
    return output_path


# ── Lucas-Kanade оптичний потік ───────────────────────────────────────────────

def optical_flow_lk(
    video_path: str,
    save_frames_dir: Optional[str] = None,
    max_frames: int = 30,
) -> int:
    """
    Lucas-Kanade sparse optical flow.
    Відстежує кутові точки (Shi-Tomasi) між сусідніми кадрами.
    Повертає кількість збережених кадрів.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не вдалося відкрити відео: {video_path}")

    if save_frames_dir:
        Path(save_frames_dir).mkdir(parents=True, exist_ok=True)

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return 0

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    # Рандомні кольори для траєкторій
    rng = np.random.default_rng(42)
    color = rng.integers(0, 255, (100, 3)).tolist()
    mask = np.zeros_like(prev_frame)

    saved = 0
    frame_idx = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_pts is not None and len(prev_pts) > 0:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

            if new_pts is not None and status is not None:
                good_new = new_pts[status == 1]
                good_old = prev_pts[status == 1]

                for j, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    col = color[j % len(color)]
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), col, 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, col, -1)

                prev_pts = good_new.reshape(-1, 1, 2)

        output = cv2.add(frame, mask)

        if save_frames_dir and frame_idx % 5 == 0:
            path = str(Path(save_frames_dir) / f"lk_flow_{saved:03d}.png")
            cv2.imwrite(path, output)
            saved += 1

        prev_gray = gray

    cap.release()
    return saved


# ── Farneback щільний оптичний потік ─────────────────────────────────────────

def optical_flow_dense(
    video_path: str,
    save_frames_dir: Optional[str] = None,
    max_frames: int = 10,
) -> int:
    """
    Farneback dense optical flow — обчислює вектори для кожного пікселя.
    Візуалізується через HSV: hue = напрям, value = magnitude.
    Повертає кількість збережених кадрів.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не вдалося відкрити відео: {video_path}")

    if save_frames_dir:
        Path(save_frames_dir).mkdir(parents=True, exist_ok=True)

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return 0

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255

    saved = 0
    frame_idx = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if save_frames_dir and frame_idx % 5 == 0:
            combined = np.hstack([frame, bgr])
            path = str(Path(save_frames_dir) / f"dense_flow_{saved:03d}.png")
            cv2.imwrite(path, combined)
            saved += 1

        prev_gray = gray

    cap.release()
    return saved


# ── Віднімання фону ───────────────────────────────────────────────────────────

def background_subtraction(
    video_path: str,
    method: str = "MOG2",
    save_frames_dir: Optional[str] = None,
    max_frames: int = 10,
) -> int:
    """
    Відніматель фону для виявлення рухомих об'єктів.
    method — 'MOG2' (Gaussian Mixture) або 'KNN' (K-Nearest Neighbors).
    Повертає кількість збережених кадрів.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не вдалося відкрити відео: {video_path}")

    if save_frames_dir:
        Path(save_frames_dir).mkdir(parents=True, exist_ok=True)

    if method == "MOG2":
        subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    elif method == "KNN":
        subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    else:
        raise ValueError(f"Невідомий метод: {method}. Використовуйте 'MOG2' або 'KNN'.")

    saved = 0
    frame_idx = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        fg_mask = subtractor.apply(frame)

        # Морфологічне очищення маски (видаляємо шум)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        if save_frames_dir and frame_idx % 5 == 0:
            combined = np.hstack([frame, cv2.cvtColor(fg_clean, cv2.COLOR_GRAY2BGR)])
            path = str(Path(save_frames_dir) / f"bg_sub_{method.lower()}_{saved:03d}.png")
            cv2.imwrite(path, combined)
            saved += 1

    cap.release()
    return saved


# ── Статична демонстрація (без відео) ────────────────────────────────────────

def demo_optical_flow_static(save_path: Optional[str] = None, show: bool = True) -> None:
    """
    Демонстрація концепції оптичного потоку на двох синтетичних кадрах.
    Показує: кадр1, кадр2, вектори потоку Farneback.
    """
    h, w = 120, 160

    # Кадр 1: прямокутник зліва
    frame1 = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        val = int(150 - y * 0.3)
        frame1[y, :] = [max(val - 20, 20), max(val - 10, 30), min(val + 30, 200)]
    cv2.rectangle(frame1, (30, 50), (70, 80), (80, 80, 80), -1)

    # Кадр 2: той самий прямокутник зсунутий вправо
    frame2 = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        val = int(150 - y * 0.3)
        frame2[y, :] = [max(val - 20, 20), max(val - 10, 30), min(val + 30, 200)]
    cv2.rectangle(frame2, (60, 50), (100, 80), (80, 80, 80), -1)

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=2, winsize=10,
        iterations=3, poly_n=5, poly_sigma=1.1, flags=0,
    )

    # HSV-кодування потоку
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(
        axes,
        [frame1, frame2, flow_vis],
        ["Кадр 1", "Кадр 2 (об'єкт зсунутий)", "Farneback потік"],
    ):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.suptitle("Лаб3 — Оптичний потік (демо)", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def demo_background_subtraction_static(
    save_path: Optional[str] = None, show: bool = True
) -> None:
    """
    Статична демонстрація концепції віднімання фону на 4 синтетичних кадрах.
    """
    h, w = 120, 160
    frames = []
    for i in range(4):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            val = int(150 - y * 0.3)
            frame[y, :] = [max(val - 20, 20), max(val - 10, 30), min(val + 30, 200)]
        x = 20 + i * 30
        cv2.rectangle(frame, (x, 50), (x + 30, 75), (80, 80, 80), -1)
        frames.append(frame)

    # Емулюємо MOG2 через різницю з першим кадром
    bg = frames[0].astype(np.float32)
    subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    masks = []
    for f in frames:
        m = subtractor.apply(f)
        masks.append(m)

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for i, (fr, mask) in enumerate(zip(frames, masks)):
        axes[0, i].imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Кадр {i + 1}", fontsize=9)
        axes[0, i].axis("off")
        axes[1, i].imshow(mask, cmap="gray")
        axes[1, i].set_title(f"MOG2 маска {i + 1}", fontsize=9)
        axes[1, i].axis("off")

    plt.suptitle("Лаб3 — Віднімання фону MOG2 (демо)", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
