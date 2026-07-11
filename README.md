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
├── main.py                          # Root entry point to start the development server (local setup)
├── requirements.txt                 # Project dependencies (FastAPI, PyTorch, Ultralytics, etc.)
├── Dockerfile                       # Instructions to build the Docker image (PyTorch + CUDA 12.1)
├── docker-compose.yml               # Service definitions, volume mounts, GPU reservation, and env vars
├── .dockerignore                    # Excludes massive datasets and virtual env from Docker context
├── custom_bytetrack.yaml            # Configuration file for ByteTrack multi-object tracking
│
├── scripts/                         # Relocated training, testing, and validation scripts
│   ├── train.py                     # YOLOv11m crowd detector training script (with Pause/Resume & OOM safety)
│   ├── finetune_csrnet_partA.py     # CSRNet fine-tuning script for ShanghaiTech Part A dataset
│   ├── finetune_csrnet_jhu.py       # CSRNet fine-tuning script for JHU-CROWD++ dataset
│   ├── validate_csrnet_groundtruth.py # CSRNet ground-truth validation & accuracy evaluation
│   ├── compare_yolo_csrnet.py       # Compares count metrics between YOLOv11m and CSRNet
│   ├── test_zone_risk.py            # Test script for density grid zone risk assessments
│   └── ...                          # Other test and validation helpers
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

## 🚀 Quick Start (Running the Dashboard)

You can run the web dashboard using either **Docker (Recommended)** or a **local Python virtual environment**. 

Both methods search for custom weights, download pre-trained weights (`yolo11n.pt`) as a fallback, and automatically initialize your designated video file or a webcam.

### 🐳 Option A: Docker Compose (Recommended)
This runs the application inside an isolated Linux container with native GPU acceleration using your host's NVIDIA graphics card.

#### Prerequisites
1. **WSL 2** installed (`wsl --install` on Windows).
2. **NVIDIA GPU drivers** installed on the host Windows system.
3. **Docker Desktop** installed (with "Use the WSL 2 based engine" enabled).

#### Execution
1. To build and start the server:
   ```bash
   docker compose up --build
   ```
2. Open your web browser and navigate to: **`http://localhost:8000`**
3. To stop the container, run:
   ```bash
   docker compose down
   ```

*Note: You can configure the `VIDEO_SOURCE` environment variable or mount local test videos directly inside [docker-compose.yml](file:///e:/Crowd%20Detection%20&%20%20Prevention/docker-compose.yml).*

---

### 🐍 Option B: Local Virtual Environment
This runs the application directly on your host machine's Python environment.

#### Execution
1. **Create and Activate Virtual Environment**:
   ```bash
   # Create environment
   python -m venv .venv

   # Activate (Windows)
   .venv\Scripts\activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Dashboard**:
   ```bash
   python main.py
   ```
4. Open your web browser and navigate to: **`http://127.0.0.1:8000`**

---

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
   python scripts/train.py
   ```
3. **Advanced Training Features**:
   - **Automatic Subsampling & Merging**: Before training, the script automatically selects a subset of the night-vision dataset (customizable `TARGET_COUNT` at the top of `scripts/train.py`, defaulting to `800` images, randomized with a fixed seed `42` for reproducibility). It merges this subset with the daytime dataset, mapping class `5` ('person') to class `0` ('people') and discarding non-person bounding boxes to produce a clean, single-class dataset in `crowd_density_merged/`.
   - **Correct Fine-Tuning Execution**: When launching the training, choose **Start Fresh `[F]`**. The script will automatically load your best daytime model weights (`models/yolo11m_best.pt`) as the starting point and begin training from Epoch 1 on the merged dataset.
   - **Pause & Resume Support**: Safe training termination is built-in. If interrupted (Ctrl+C), relaunching the script detects the `last.pt` checkpoint and gives you the option to **Resume `[R]`** from the last completed epoch.
   - **OOM Safety**: High-resolution image size (`960px`) is used to resolve distant, small people in crowds, with a reduced batch size (`2`) configured to prevent Out-Of-Memory (OOM) errors on 8GB VRAM GPUs (like RTX 4060).
   - **Checkpoints**: Once training completes, the script automatically copies the best checkpoint weights into `models/yolo11m_best.pt`.

### 2. CSRNet Density Model Fine-Tuning
To fine-tune CSRNet on the highly congested ShanghaiTech Part A dataset:
1. Update `DATASET_DIR` in `scripts/finetune_csrnet_partA.py` with your dataset path.
2. Run the fine-tuning script:
   ```bash
   python scripts/finetune_csrnet_partA.py
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
python scripts/validate_csrnet_groundtruth.py
```
This loads test images across varying crowd densities (low, medium, high) and compares the estimated count against true counts loaded from ground-truth `.mat` files, presenting an error analysis table.

### 📈 Metrics Logging
Both YOLO and CSRNet counts are logged in real-time alongside active model decisions to a local `crowd_comparison.csv` file for offline analytical validation and audit trailing.

### 🚨 Alerting & Safety Mitigation
The dashboard features an automated **Alert Manager** (`src/alerts.py`) that monitors:
- **Max Capacity**: Threshold warning if total occupancy exceeds limit.
- **High Local Density**: Triggers warning if any single grid cell count exceeds the local limit.
- **Sustained Warning Alerting**: If warnings persist beyond the trigger delay (default `3` seconds), the status escalates to **CRITICAL**, appending incident details directly into the dashboard's live logs.

---

## 🚀 Production Safety & Security Upgrades

CrowdShield AI is updated with enterprise-grade features for security, reliability, and proactive threat prevention:

### 🔒 MongoDB Persistence & JWT Authentication
* **Admin Authentication**: A secure login modal protects safety configurations (such as Max Capacity, Caution Thresholds, and Model Selection). Non-authenticated users can view the dashboard in a safe **Read-Only / Operator Mode**.
* **JWT Access Tokens**: Access control is managed via encrypted JSON Web Tokens (JWT) stored in browser storage.
* **Persistent Settings**: Configurations are stored in a persistent **MongoDB** collection rather than volatile system memory. When the server restarts, configurations are reloaded automatically.
* **Incident Persistence**: Critical safety escalations are asynchronously logged directly to the MongoDB `incidents` database in addition to CSV files.

### 📈 Trend-Based Predictive Crowd Warnings
* **Rate of Change Tracking**: Maintains a sliding 60-second temporal history of count data to calculate crowd growth rates (`fill_rate` in people/minute) using a 3-sample sliding average window.
* **Time-to-Capacity Forecasts**: Estimates how many minutes remain before capacity thresholds are crossed: `time_to_capacity = (max_capacity - current_count) / fill_rate` (capped at 60.0 mins).
* **Predictive Alert Tiers**:
  * **Early Warning** (Amber): Triggered when crowd growth exceeds `2.0 people/min` and estimated time to capacity is `< 10.0 minutes`.
  * **Urgent Warning** (Amber): Triggered when crowd growth exceeds `5.0 people/min` and estimated time to capacity is `< 5.0 minutes`.
  * **Critical Prediction** (Red): Triggered when estimated time to capacity drops below `2.0 minutes` regardless of growth rate.
* **Sustained Alarm Delay**: Trend warnings must be sustained for `10` seconds before triggering, filtering out brief transient spikes.
* **UI Trend Indicators**: The dashboard displays live visual cues under the capacity progress bar (`↑ +{rate} people/min · ~{time} min` in Amber/Red, or `↓ Crowd dispersing` in Green).

### 📘 Interactive System Operator Guide
* A clickable **System Guide** modal overlay provides immediate onboarding for operators, explaining:
  * **Detection Modes**: Dynamic switching behaviors between YOLO and CSRNet models.
  * **View Modes**: Visual rendering differences across raw bounding boxes, grid cell highlights, motion flow vectors, and hot-spot density maps.
  * **Camera Environments**: Preconfigured spatial bounding coordinates for General, Venue, and Aerial drone feeds.



