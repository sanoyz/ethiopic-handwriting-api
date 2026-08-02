# 🖋️ Ethiopic Handwriting Recognition API

[![Live Demo](https://img.shields.io/badge/Live_on-Render-46C2CB?style=for-the-badge&logo=render)](https://ethiopic-handwriting-recognition.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)

**Live API Endpoint:** [https://ethiopic-handwriting-recognition.onrender.com/](https://ethiopic-handwriting-recognition.onrender.com/)

---

## 📖 Overview

This API serves a research prototype for **Online Ethiopic Handwriting Recognition**. Unlike traditional OCR which processes static images, this system captures pen strokes as a **temporal signal** (x/y coordinates, timestamps, and inter-stroke dynamics).

This project introduces the first comprehensive study of online Ethiopic handwriting recognition. It is an **independent, unfunded research initiative** built to demonstrate the feasibility of using memory-augmented transformers and temporal features for low-resource script recognition.

> **⚠️ Important:** This project is **distinct** from my Master's thesis and uses a smaller, private dataset. The full thesis is currently under a formal university publication embargo.

---

## 🚀 Key Technical Contributions

### 1. Rich Feature Engineering
We extract a **156-dimensional stroke feature representation** per stroke, combining:
- **Resampled point trajectories** (x, y, and curvature).
- **Per-stroke statistics** (length, bounding box, total duration).
- **Inter-stroke gap features** encoding the spatial, temporal, and structural relationships between consecutive strokes.

### 2. Memory-Augmented Temporal Encoder
The core architecture is built around a **position-aware memory bank** of learnable stroke-pattern prototypes. This memory is fused with the input stroke sequence through **multi-head cross-attention**, allowing the model to dynamically retrieve relevant stroke patterns during inference.

### 3. Dual Ablation Methodology
We employ a rigorous dual ablation strategy:
- **Retraining from scratch** without certain components.
- **Lesioning** (removing components at inference time) from a converged checkpoint.

Key findings from this methodology (with the current dataset):
- The **memory bank is structurally necessary**. Removing it degrades performance significantly.
- **Inter-stroke gap features** are critically used at inference time, even if they are only moderately necessary during training.
- A **spatial (image-based) branch** contributes little beyond a mild regularization effect.

### 4. Convergence Guidelines
We observe that extending training from 80 to 120 epochs allows the optimizer to escape a local plateau, leading to notable performance improvements. This establishes a practical convergence guideline for multi-modal handwriting recognition systems.

---

## ⚠️ Performance & Dataset Disclaimer

**Please read this section carefully.**

The current model has been trained on a **small-scale, private dataset** (~5,000 sentences from 5 participants). While the preliminary results on a held-out test set are promising, **this dataset is not large or diverse enough to guarantee robust, generalizable performance**.

| Limitation | Explanation |
|------------|-------------|
| **Small Sample Size** | Only 5 participants were involved. Writing styles vary significantly across individuals. |
| **Limited Vocabulary** | The dataset covers a subset of Ethiopic characters and common words. |
| **No Formal Ethics Approval** | The dataset was collected privately and cannot be publicly released. |

**What this prototype demonstrates:**
- The architectural framework is sound and worth scaling.
- The temporal encoding and memory-augmented attention mechanisms work in practice.
- The dual ablation methodology provides deep insights into the model's inner workings.

**What this prototype does NOT yet prove:**
- Robust performance across diverse handwriting styles.
- State-of-the-art performance compared to other low-resource or offline benchmarks.

We release this API as a **reproducible baseline** and a **research demonstration**, encouraging the community to validate and extend this work with larger, more diverse datasets.

---

## 📡 API Endpoints

### Root Endpoint
- **URL:** `GET /`
- **Description:** Health check. Confirms the API is running.
- **Response:**
  ```json
  {
    "status": "Online",
    "message": "Ethiopic Handwriting Recognition API is ready"
  }
