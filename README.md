# 🛡️ CrowdShield AI — Crowd Detection & Prevention System

CrowdShield AI is a state-of-the-art hybrid AI/ML platform that leverages computer vision (YOLOv11m) and density regression neural networks (CSRNet) to perform real-time crowd density estimations, track spatial occupancy counts, and log safety incidents to prevent dangerous crowd build-ups.

This project is built using a **hybrid workflow**:
1. **Google Colab (GPU)** or **Local Machine (RTX 4060 GPU)** for training and fine-tuning deep learning models.
2. **Local Machine (CPU/GPU)** for running the real-time processing stream (webcam or synthetic fallback), spatial grid calculations, heatmaps, and the interactive FastAPI web dashboard.

---

## ⚡ Key Highlights & Architecture

### 🔄 Dual-Model Switching & Occlusion Handling
Unlike traditional systems that rely solely on bounding boxes, CrowdShield AI features an **intelligent dual-model switching pipeline**:
- **YOLOv11m (Detection-based)**: Used for low-density or sparse crowds. It draws precise bounding boxes around individual people and excels when visual occlusion is minimal.
- **CSRNet (Congested Scene Recognition Network - Density-based)**: Automatically activated for dense crowds. Since crowd congestion causes severe body overlap (occlusion) where bounding box detectors fail, CSRNet utilizes dilated convolutions (Configuration B) to map density distributions directly without border detection, estimating highly congested crowds accurately.
- **Intelligent Switching Heuristics**: The system dynamically switches from YOLO to CSRNet if:
  - The YOLO count exceeds a customizable capacity (default `15` people).
  - **OR** if a significant ratio of bounding boxes overlap heavily (IoU > `0.3`), specifically when `20%` or more of the crowd overlaps.

---

## 🛠️ Project Structure

Below is the directory map of CrowdShield AI:

```text
Crowd-Detection-And-Prevention/
│
├── main.py                          # Root entry point to start the development server
├── requirements.txt                 # Project dependencies (FastAPI, PyTorch, Ultralytics, etc.)
├── train.py                         # YOLOv11m crowd detector training script (with Pause/Resume & OOM safety)
├── finetune_csrnet_partA.py         # CSRNet fine-tuning script for ShanghaiTech Part A dataset
├── validate_csrnet_groundtruth.py  # CSRNet ground-truth validation & accuracy evaluation
├── train_crowd_detector.ipynb       # Jupyter notebook for training models on Google Colab GPU
│
├── src/                             # Core AI/ML modules
│   ├── detector.py                  # YOLOv11m object detection logic
│   ├── csrnet_model.py              # CSRNet model architecture definition (VGG-16 front-end + dilated backend)
│   ├── csrnet_inference.py          # Preprocessing and estimation utility functions for CSRNet
│   ├── density.py                   # Grid densities, visual heatmaps, switching heuristics, and risk assessment
│   └── alerts.py                    # Real-time state tracking (NORMAL/WARNING/CRITICAL) and historical logs
│
├── webapp/                          # Interactive dashboard application
│   ├── main.py                      # FastAPI server endpoints, configuration POST handler, MJPEG stream generator
│   ├── templates/
│   │   └── index.html               # Sleek glassmorphic dark-mode dashboard HTML design
│   └── static/
│       ├── css/
│       │   └── style.css            # Dark mode variables, glassmorphic layout styling, and dashboard transitions
│       └── js/
│           └── dashboard.js         # Real-time dashboard DOM updates, config sliders, chart initialization
│
└── models/                          # Storage for trained weights (ignored in git)
    ├── csrnet_shanghaitech.pth      # Pre-trained baseline CSRNet weights
    ├── csrnet_partA_finetuned_best.pth # Fine-tuned custom CSRNet weights
    └── yolo11m_best.pt              # Best fine-tuned YOLOv11m weights
```

---

## 🚀 Quick Start (Local Web Dashboard)

You can run the web dashboard immediately. It will search for custom weights, download pre-trained weights (`yolo11n.pt`) as a fallback, and automatically initialize a **synthetic crowd simulation stream** if no physical webcam is connected.

### 1. Install Dependencies
Create a virtual environment and install requirements:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Run the Dashboard
Start the local server using the main entry point:

```bash
python main.py
```

Open your browser and navigate to: **`http://127.0.0.1:8000`**

### 📺 Available Feed Views in Dashboard:
- **Raw Bboxes (YOLOv11m)**: Renders standard bounding boxes around detected individuals.
- **Spatial Grid Count**: Overlay grid dividing the camera viewport into a customizable `3x3` grid displaying localized counts per cell. Cells are highlighted as **Safe (Green)**, **Caution (Yellow)**, or **Danger (Red)** based on density limits.
- **CSRNet Heatmap**: High-fidelity smooth Gaussian density heatmap overlays showing hotspots and dense clusters.

---

## 🧠 Model Training Workflow

### 1. YOLOv11m Custom Detector Training
To fine-tune a model on custom crowd/person datasets locally:
1. Ensure you have a CUDA-enabled GPU and your virtual environment is active.
2. Run the standalone training script:
   ```bash
   python train.py
   ```
3. **Advanced Training Features**:
   - **Pause & Resume Support**: Safe training termination is built-in. If interrupted (Ctrl+C), relaunching the script detects the `last.pt` checkpoint and gives you the option to **Resume `[R]`** from the last completed epoch or **Start Fresh `[F]`**.
   - **OOM Safety**: High-resolution image size (`960px`) is used to resolve distant, small people in crowds, with a reduced batch size (`2`) configured to prevent Out-Of-Memory (OOM) errors on 8GB VRAM GPUs (like RTX 4060).
   - **Checkpoints**: Once training completes, the script automatically copies the best checkpoint weights into `models/yolo11m_best.pt`.

### 2. CSRNet Density Model Fine-Tuning
To fine-tune CSRNet on the highly congested ShanghaiTech Part A dataset:
1. Update `DATASET_DIR` in `finetune_csrnet_partA.py` with your dataset path.
2. Run the fine-tuning script:
   ```bash
   python finetune_csrnet_partA.py
   ```
3. **Training Details**:
   - Uses VGG-16 front-end layers pretrained on ImageNet for feature extraction.
   - Fine-tunes using a low learning rate (`1e-6`) and MSE loss on density map regressions to prevent destroying general features.
   - Saves checkpoint files (`models/csrnet_partA_finetuned_epoch{epoch}.pth`) and outputs the best performance model to `models/csrnet_partA_finetuned_best.pth`.

---

## 📊 Validation & Incident Alerts

### 🔍 Ground-truth Validation Script
You can validate the accuracy of your fine-tuned CSRNet models against the ShanghaiTech test dataset using the validation runner:
```bash
python validate_csrnet_groundtruth.py
```
This loads test images across varying crowd densities (low, medium, high) and compares the estimated count against true counts loaded from ground-truth `.mat` files, presenting an error analysis table.

### 📈 Metrics Logging
Both YOLO and CSRNet counts are logged in real-time alongside active model decisions to a local `crowd_comparison.csv` file for offline analytical validation and audit trailing.

### 🚨 Alerting & Safety Mitigation
The dashboard features an automated **Alert Manager** (`src/alerts.py`) that monitors:
- **Max Capacity**: Threshold warning if total occupancy exceeds limit.
- **High Local Density**: Triggers warning if any single grid cell count exceeds the local limit.
- **Sustained Warning Alerting**: If warnings persist beyond the trigger delay (default `3` seconds), the status escalates to **CRITICAL**, appending incident details directly into the dashboard's live logs.


1) Create python virtual environment
python -m venv .venv
2) Activate virtual environment
.venv\Scripts\activate
3) Install requirements
pip install -r requirements.txt
4) Run dashboard
python main.py
5) Open http://[IP_ADDRESS] in browser

