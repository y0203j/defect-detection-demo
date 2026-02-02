## Defect Detection

This project explores an end-to-end deep learning pipeline for detecting manufacturing defects in powder bed fusion (PBF) layer images.
The goal is to identify defective regions in individual print layers using image-based semantic segmentation.

**Data source:** [Kaggle: Process Monitoring in Additive Manufacturing](https://www.kaggle.com/datasets/programmer3/process-monitoring-in-additive-manufacturing/data)

---

## Introduction

This repository is a compact, hands-on project designed to showcase the skills expected of a junior ML engineer at Additive Assurance. It focuses on image-based semantic segmentation for defect detection in powder bed fusion (PBF) layer images and covers the full ML lifecycle: data preparation, model training, quantization, and deployment.

The project includes training and benchmarking pipelines, along with a minimal FastAPI + Docker deployment to demonstrate how trained models integrate into a production software stack.

---


## Key Features

- ⁠**Architecture:** U-Net with a *MobileNetV2* encoder (Pre-trained on ImageNet). This reflects a practical compromise between performance and deployability.
- **Synthetic Data Engineering:** Includes an automated pipeline to generate synthetic ground-truth masks from raw layer imagery using computer vision thresholding.
- ⁠**Hardware Acceleration:** Supports CUDA and Apple Metal (MPS) training. Achieving up to 96.69% recall on GPU-trained runs.
- ⁠**Quantization:** Implements Dynamic Quantization (Post-Training), benchmarking a 1.05x–1.10x reduction in inference latency on CPU.

---

## Tech Stack

- **Core:** Python, PyTorch
- **Data & Utilities:** NumPy, Pillow, Pandas
- **Augmentation:** Albumentations
- **Modeling & Tooling:** Segmentation-Models-Pytorch, Timm
- **Deployment:** FastAPI, Uvicorn, Docker (for containerization)

---

## Project Structure

```text
defect-detection-demo/
├── data/                      # Raw and processed images
├── dataset_loader.py          # PyTorch Dataset + Augmentation & Preprocessing
├── setup_data.py              # Filters raw files + Generates synthetic masks
├── train_and_quantize.py      # CPU-friendly training + Quantization flow
├── train_and_quantize_gpu.py  # GPU/MPS training + Benchmarking
├── defect_model_float32.pth   # Example trained artifact
├── defect_model_quantized.pth # Quantized Artifact
├── requirements.txt           # Project Dependencies
└── deploy/                    # FastAPI demo + Dockerfile for deployment
    ├── deployapp.py
    ├── static/
    └── Dockerfile
```

---

## Quick Start

1. Create & activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

2.Data Preparation
The public dataset contains noise (recoater images, powder bed images). Run the setup script to filter for "After Laser" melt pool images and generate synthetic training labels. 
```bash
python setup_data.py
```
4. Run a short training experiment (CPU)
```bash
python train_and_quantize.py
```
5. Run the demo service (from the `deploy/` folder)
```bash
cd deploy
python3 -m venv venv && source venv/bin/activate
pip install -r deployrequirements.txt
uvicorn deployapp:app --host 0.0.0.0 --port 8001
# open http://localhost:8001/
```

---

## Results & Benchmarks

### 1. Training Performance Comparison

| Metric | GPU Training (Nvidia 3090) | CPU Training |
| :--- | :--- | :--- |
| **Data Scale** | **Full Dataset** (~500 images) | **Subset** (30 images, Low-Memory Mode) |
| **Training Time** | **23.8 mins** | 64.8 mins |
| **Final Dice Loss** | **0.1308** | 0.1916 |
| **Final Recall** | **96.69%** | 89.18% |
| **Converged?** | Yes | No (Needs more data/time) |

**Key Insight:**
The GPU-accelerated run achieved a **Recall of 0.967**, meaning the model detects **96.7% of all defects**.

### 2. Quantization Impact (Inference)

**Results observed on local hardware (MacBook Air M4):**

| Metric | Original (Float32) | Quantized (Int8) | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Size** | ~26 MB | ~26 MB | No Change |
| **Inference Latency** | ~193.58 ms | ~184.54 ms | 1.05x Faster |

**Results observed on Nvidia 3090:**
| Metric | Original (Float32) | Quantized (Int8) | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Size** | ~26 MB | ~26 MB | No Change |
| **Inference Latency** | ~44.59 ms | ~40.52 ms | 1.10x Faster |

---

## End-to-End ML Lifecycle Coverage

This project demonstrates hands-on involvement across the full ML lifecycle:
- Raw industrial image filtering and preprocessing
- Weakly supervised label generation using classical computer vision
- Deep learning model training and evaluation for semantic segmentation
- Hardware-aware benchmarking on CPU, GPU, and MPS
- Edge-oriented optimisation via post-training quantization
- Model integration into a production-style FastAPI service

---

## Analysis & Future Work

1.This prototype currently implements Dynamic Quantization, which primarily targets `Linear` and `RNN` layers in PyTorch. Since the MobileNetV2 backbone is composed almost entirely of Convolutional (`Conv2d`) layers, the Dynamic Quantization engine bypasses the majority of the network, resulting in minimal compression.To achieve the compression goal, the pipeline requires upgrading to Post-training Static Quantization.

2.The recall (96.69%) is pretty good. For further improvement, precision-recall curve would be idea to dynamically tune the decision threshold (currently 0.5) for specific manufacturing floor requirements.

---

## Web Demo & Accessibility Considerations

To demonstrate real-world integration and usability, the project includes an accessible web-based demo built with FastAPI and Docker.

A single-page demo is included under `deploy/static/index.html`. It provides a keyboard-friendly, screen-reader-friendly interface that:

- Lets you upload an image and see a pass/fail result announced (ARIA live regions).
- Shows a visual overlay image with detected defect areas (red overlay) and a yellow bounding box for the detected region.
- Returns a short text summary (max confidence, defect area %, bounding box) to aid assistive technologies.

How to run the demo locally:

1. Install deploy dependencies and run the API from the `deploy/` folder:

```bash
cd deploy
python3 -m venv venv && source venv/bin/activate
pip install -r deployrequirements.txt
uvicorn deployapp:app --host 0.0.0.0 --port 8000
```

2. Open `http://localhost:8000/` in a browser. Use the file picker, hit `Analyze`, and the UI will display results and an overlay.

Docker (optional):

```bash
cd deploy
docker build -t defect-deploy .
docker run -p 8000:8000 defect-deploy
```

Notes:
- The API returns `mask_image` as a small PNG overlay embedded as a data URL, plus `bbox` coordinates where defects were found.
- For accessibility testing, run Lighthouse (Accessibility), Pa11y, or manual checks with VoiceOver/NVDA.

---

Author: JY

