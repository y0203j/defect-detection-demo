# End-to-End Defect Detection with Multi-Angle Imaging

This project demonstrates an end-to-end deep learning pipeline for detecting manufacturing defects in powder bed fusion (PBF) layer images. It leverages a multi-angle photometric stereo technique, using three images of the same location under different lighting to robustly identify defect regions.

This repository is a compact, hands-on project designed to showcase the skills expected of a junior ML engineer. It covers the full ML lifecycle: data preparation, model training, quantization, and deployment.

---

## Key Features

-   **Architecture:** U-Net with a **MobileNetV2** encoder (Pre-trained on ImageNet), offering a practical balance between performance and deployability.
-   **Multi-Angle Data Fusion:** Stacks three single-channel images (captured under different lighting angles) into a 3-channel tensor, providing rich topographical information to the model.
-   **Synthetic Data Engineering:** Includes an automated pipeline to generate synthetic ground-truth masks from raw layer imagery using computer vision thresholding.
-   **Hardware Acceleration:** Supports CUDA and Apple Metal (MPS) for training.
-   **Quantization:** Implements Post-Training Dynamic Quantization to benchmark inference latency improvements.
-   **Deployment:** The trained model is packaged into a lightweight FastAPI service, with a Dockerfile for portability and validation.

---

## Tech Stack

-   **Core:** Python, PyTorch
-   **Data & Utilities:** NumPy, Pillow, OpenCV, Pandas
-   **Augmentation:** Albumentations
-   **Modeling & Tooling:** Segmentation-Models-Pytorch, Timm
-   **Deployment:** FastAPI, Uvicorn, Docker

---

## Project Structure

```text
detect-detection-deploy-multiangle/
├── data/                      # Raw and processed images
├── dataset_loader_ma.py       # PyTorch Dataset + Augmentation & Preprocessing
├── setup_data_ma.py           # Filters raw files + Generates synthetic masks
├── train_and_quantize_ma.py   # Training + Quantization flow
├── defect_model_float32.pth   # Example trained artifact
├── defect_model_quantized.pth # Quantized Artifact
├── requirements.txt           # Project Dependencies
└── deploy/                    # FastAPI demo + Dockerfile for deployment
    ├── deployapp_ma.py
    ├── static/
    └── Dockerfile
```

---

## Quick Start

1.  **Create & activate a virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Data Preparation**
    The script filters raw files and generates synthetic training labels from the multi-angle images.
    ```bash
    python setup_data_ma.py
    ```
4.  **Run a training experiment**
    ```bash
    python train_and_quantize_ma.py
    ```
5.  **Run the demo service** (from the `deploy/` folder)
    ```bash
    cd deploy
    python3 -m venv venv && source venv/bin/activate
    pip install -r deployrequirements.txt
    uvicorn deployapp_ma:app --host 0.0.0.0 --port 8001
    # open http://localhost:8001/
    ```

---

## Modeling Insights & Results

### 1. Handling Extreme Class Imbalance

LPBF defect detection is an extremely imbalanced problem—most pixels are background. To address this, I used a composite loss function:
> I handle this using **Dice plus BCE loss**. Dice forces the model to focus on defect overlap, while BCE stabilizes pixel-level learning.
> This gives strong recall without sacrificing mask quality.

To further combat class imbalance, the `pos_weight` argument in the BCE loss was critical. By increasing it from an initial value of 5.0 to **40.0**, the model was penalized more heavily for missing defects (false negatives).
> This adjustment alone boosted validation recall from **0.637 to 0.777**.

### 2. Evolving Mask Generation Heuristics

The quality of synthetic labels is paramount.
> When I upgraded the data representation to multi-angle stacking, I intentionally made the mask heuristic conservative. The initial **AND-based consensus** (where a pixel had to be a defect in all three images) turned out to be too strict, producing extremely sparse labels. This caused the model to converge to a trivial background-only solution.
> I corrected this by moving to a **majority-vote heuristic** and implementing class-weighted BCE, which restored label density and stabilized training.

### 3. Quantization Impact (Inference)

Post-training dynamic quantization was applied to benchmark performance.

**Results observed on local hardware (MacBook Air M1):**

| Metric             | Original (Float32) | Quantized (Int8) | Improvement  |
| :----------------- | :----------------- | :--------------- | :----------- |
| **Inference Latency** | ~193.58 ms         | ~184.54 ms       | 1.05x Faster |

---

## Analysis & Future Work

1.  **Improve Quantization:** This prototype uses Dynamic Quantization, which primarily targets `Linear` layers. Since MobileNetV2 is composed almost entirely of `Conv2d` layers, the engine bypasses most of the network, yielding minimal compression. To achieve a significant reduction in model size and latency, the pipeline should be upgraded to **Post-Training Static Quantization (PTSQ)**, which requires a calibration dataset.

2.  **Tune Decision Threshold:** The current recall is strong. For further optimization, a **precision-recall curve analysis** would be ideal to dynamically tune the decision threshold (currently 0.5) to meet specific manufacturing floor requirements (e.g., minimizing false negatives vs. false positives).

---

## Web Demo & Accessibility

A single-page demo is included under `deploy/static/index.html` to demonstrate real-world integration. It provides a keyboard-friendly, screen-reader-friendly interface that:

-   Lets you upload the three angle images and see a pass/fail result announced via ARIA live regions.
-   Shows a visual overlay with detected defect areas.
-   Returns a short text summary (max confidence, defect area %, bounding box) to aid assistive technologies.

You can also run the deployment environment using Docker:
```bash
cd deploy
docker build -t defect-deploy-ma .
docker run -p 8001:8000 defect-deploy-ma
```
**Note:** The Docker image was built locally, pushed to a private registry, and successfully run on a separate machine, validating its portability.
