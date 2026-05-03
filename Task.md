# Task: Flying Object Recognition System

## Project Overview

Build a **flying object recognition system** capable of detecting and classifying aircraft, helicopters, and birds using computer vision, segmentation, object detection, and deep learning.

> **Important:** All labs are part of **one continuous project**. Each lab builds on top of previous work. Code from earlier labs must be reusable in later labs.

---

## Lab 1 — Foundations & Image Analysis

### Week 1: Introduction to Computer Vision
- Review **OpenCV**, **PIL**, and **NumPy** libraries; document their roles in image analysis.
- Build an **image loader/display module** for flying object images (reusable across all subsequent labs).

### Week 2: Digital Image Analysis
- Support loading images in **PNG, JPG, BMP** formats and print their properties (dimensions, channels, dtype, etc.).
- Implement **brightness histogram** plotting and **contrast enhancement** methods (histogram equalization, CLAHE, etc.).

---

## Lab 2 — Filtering & Segmentation

### Week 3: Image Filtering
- Apply **Gaussian**, **Median**, and **Bilateral** filters for noise removal.
- Implement **sharpening** methods (unsharp mask, Laplacian-based, etc.).

### Week 4: Image Segmentation
- Implement **threshold segmentation** and **Otsu's method** for object isolation.
- Implement **Watershed** and **GrabCut** algorithms for foreground/background separation.

---

## Lab 3 — Feature Extraction & Video Processing

### Week 5: Feature Extraction & Descriptors
- Detect **contours** of flying objects.
- Apply **SIFT**, **SURF**, **ORB**, and **HOG** descriptors for feature analysis.

### Week 6: Video Stream Processing
- Open a video stream and analyze moving objects using **Optical Flow**.
- Apply **Background Subtraction** to isolate objects from the background.

---

## Lab 4 — Geometric Transformations & Morphology

### Week 7: Geometric Transformations
- Implement **scaling**, **rotation**, and **perspective transformation** for image correction.

### Week 8: Morphological Operations
- Apply **erosion**, **dilation**, **opening**, and **closing** to improve segmentation quality.

---

## Lab 5 — Classical ML Classification

### Week 9: Object Recognition
- Use **SVM**, **Random Forest**, and **KNN** algorithms to classify flying objects.

### Week 10: Deep Learning Basics
- Build a **CNN** neural network from scratch to classify flying objects.

---

## Lab 6 — Pretrained Models & Data Preparation

### Week 11: Popular Neural Network Architectures
- Use pretrained models — **ResNet**, **MobileNet**, **YOLO**, or **UNet** — for object detection.

### Week 12: Data Augmentation & Preparation
- Perform **dataset augmentation** (rotation, brightness shift, noise injection) to improve model accuracy.

---

## Lab 7 — Real-Time Recognition & Practical Applications

### Week 13: Object Recognition with Deep Learning
- Test **deep learning recognition algorithms** on flying object datasets.

### Week 14: Practical Applications
- Implement **real-time automatic detection and classification** of flying objects.
- Demonstrate the model on **satellite imagery or aerial photographs** to detect flying objects.

---

## Expected Result

A functional system for **detecting and recognizing flying objects** that:
- Processes both **static images** and **live video streams**
- Uses a diverse range of **computer vision methods**
- Is built as a **single evolving codebase** where each lab extends the previous one