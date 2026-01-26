## Defect Detection

This project explores an end-to-end deep learning pipeline for detecting manufacturing defects in powder bed fusion (PBF) layer images.
The goal is to identify defective regions in individual print layers using image-based semantic segmentation.

**Data source:** [Kaggle: Process Monitoring in Additive Manufacturing](https://www.kaggle.com/datasets/programmer3/process-monitoring-in-additive-manufacturing/data)

---

## Why This Project

This project is designed to mirror real-world industrial ML workflows, where labels are imperfect, data is noisy, and models must balance performance with deployability on edge hardware.

---


## Key Features

- ⁠**Architecture:** U-Net with a *MobileNetV2* encoder (Pre-trained on ImageNet). This reflects a pratical compromise between performance and deployability.
- **Synthetic Data Engineering:** Includes an automated pipeline to generate synthetic ground-truth masks from raw layer imagery using computer vision thresholding.
- ⁠**Hardware Acceleration:** Supports CUDA and Apple Metal (MPS) training. Achieving up to 96.69% recall on GPU-trained runs.
- ⁠**Quantization:** Implements Dynamic Quantization (Post-Training), benchmarking a 1.05x–1.10x reduction in inference latency on CPU.

---

## Tech Stack

- **Core:** Python, PyTorch
- **data Preprocesing:** Pillow
- ⁠**Augmentation:** Albumentations
- ⁠**Architecture:** Segmentation Models PyTorch
- ⁠**Deployment:** Quantization

---

## Project Structure

```text
defect-detection-demo/
├── data/
│   ├── train_images/          # Filtered input images (Layer "B" - After Laser)
│   └── train_masks/           # Auto-generated binary masks
├── dataset_loader.py          # Custom PyTorch Dataset with Robust Augmentation
├── train_and_quantize.py      # Main pipeline: Training -> Evaluation -> Quantization
├── train_and_quantize_gpu.py  # Main pipeline: Training -> Evaluation -> Quantization GPU version
├── setup_data.py              # Data Engineering script (Filtering & Mask Gen)
└── requirements.txt           # Dependencies
```

---

## Quick Start

1.Environment Setup
##### Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
##### Install dependencies
```bash
pip install -r requirements.txt
```

2.Data Preparation
The public dataset contains noise (recoater images, powder bed images). Run the setup script to filter for "After Laser" melt pool images and generate synthetic training labels. 
```bash
python setup_data.py
```
Output: Scans raw data, creates data/train_images and data/train_masks.

3.Training & Quantization
Run the main pipeline. This will:
1.Train the U-Net for 5 Epochs.
2.Save the full precision model (defect_model_float32.pth).
3.Quantize the model to Int8.
4.Benchmark the speedup (Latency test).
```bash
python train_and_quantize.py or
python train_and_quantize_gpu.py
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

## Analysis & Future Work
1.This prototype currently implements Dynamic Quantization, which primarily targets `Linear` and `RNN` layers in PyTorch. Since the MobileNetV2 backbone is composed almost entirely of Convolutional (`Conv2d`) layers, the Dynamic Quantization engine bypasses the majority of the network, resulting in minimal compression.To achieve the compression goal, the pipeline requires upgrading to Post-training Static Quantization.

2.The recall (96.69%) is pretty good. For further improvement, precision-recall curve is idea to dynamically tune the decision threshold (currently 0.5) for specific manufacture floor requirements.

---

**Author:** JY
