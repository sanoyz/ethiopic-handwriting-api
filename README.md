# 🖋️ Ethiopic Handwriting Recognition API

[![Live Demo](https://img.shields.io/badge/Live_on-Render-46C2CB?style=for-the-badge&logo=render)](https://ethiopic-handwriting-api.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)

**Live API Endpoint:** [https://ethiopic-handwriting-api.onrender.com/](https://ethiopic-handwriting-api.onrender.com/)

---

## 📖 Overview

This API serves a **Memory-Augmented Transformer** model for online handwriting recognition. It is designed to process raw stroke data (pen coordinates and timestamps) and output recognized Ethiopic script.

This project is an **independent, unfunded research initiative**. It demonstrates the feasibility of using temporal dynamics (velocity, hesitation, stroke order) combined with memory-augmented neural architectures for low-resource script recognition.

> **⚠️ Important Note:** This project is **distinct** from my Master's thesis and uses a smaller, private dataset (~5,000 sentences, 5 participants). The full thesis is currently under a formal university publication embargo.

---

## 🚀 Features

- **Real-Time Inference:** Send stroke data and receive predictions instantly.
- **Temporal Dynamics:** Utilizes stroke order, velocity, and pause duration as primary signals.
- **Memory-Augmented Transformer:** Experiments with external memory mechanisms to handle long stroke sequences.
- **Containerized & Deployed:** Served via a REST API hosted on Render.

---

## 🛠️ Tech Stack

- **Framework:** FastAPI (or Flask - adjust based on your actual code)
- **Deep Learning:** PyTorch
- **Deployment:** Render (Linux-based)
- **Language:** Python 3.9+

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
