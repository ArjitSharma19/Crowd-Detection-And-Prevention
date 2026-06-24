# CrowdShield AI — Crowd Detection & Prevention System

CrowdShield AI is a hybrid AI/ML platform that leverages computer vision (YOLOv8) to perform real-time crowd density estimations, track spatial occupancy counts, and log safety incidents to prevent dangerous crowd build-ups.

This project is built using a **hybrid workflow**:
1. **Google Colab (GPU)** for training/fine-tuning the deep learning detection model.
2. **Local Machine (CPU/GPU)** for processing camera streams, calculating heatmaps, and running the interactive web dashboard interface.

---

## 🚀 Quick Start (Local Web Dashboard)

You can run the web dashboard immediately. It will search for custom weights, download pre-trained weights (`yolo11n.pt`) as a fallback, and automatically initialize a **synthetic crowd simulation stream** if no physical webcam is connected.

### 1. Install Dependencies
Create a virtual environment (optional but recommended) and install requirements:

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

---

## 🧠 Model Training Workflow (Local or Colab)

To fine-tune a model on custom crowd/person datasets, you can train either **locally** (recommended if you have a GPU) or using **Google Colab**:

### Option A: Local GPU Training (Recommended)
If your machine has a dedicated NVIDIA GPU (like your RTX 4060):
1. Activate your virtual environment and ensure you have CUDA-enabled PyTorch installed.
2. Run the standalone training script:
   ```bash
   python train.py --model yolo11m.pt --epochs 100 --batch 8 --name train_yolo11m
   ```
3. Once training completes, copy the resulting weights file:
   ```bash
   copy runs\detect\train_yolo11m\weights\best.pt models\best.pt
   ```

### Option B: Google Colab Training
1. Go to [Google Colab](https://colab.research.google.com/) and upload `train_crowd_detector.ipynb`.
2. Select a **GPU Runtime** (Runtime > Change runtime type > GPU).
3. Follow the setup and training cells.
4. Download the resulting `best.pt` weights and save them to `models/best.pt` in this directory.

After copying `best.pt` to `models/best.pt`, restart the web dashboard (`python main.py`). The system will automatically detect and load your custom weights!

---

## 🛠️ Project Structure

- `src/detector.py`: YOLOv8 wrapper to run inference on camera frames and filter human bounding boxes.
- `src/density.py`: Grid density calculations and visual smooth Gaussian heatmap overlay generation.
- `src/alerts.py`: Threshold tracking and incident log management.
- `webapp/main.py`: FastAPI server serving templates, endpoints, configuration posts, and MJPEG video streaming.
- `webapp/templates/index.html`: Sleek dark-mode dashboard HTML design with glassmorphic cards.
- `webapp/static/`: Layout styles (`css/style.css`) and metrics updater scripts (`js/dashboard.js`).
