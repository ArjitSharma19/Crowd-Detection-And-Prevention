# CrowdShield AI — Crowd Detection & Prevention System

CrowdShield AI is a hybrid AI/ML platform that leverages computer vision (YOLOv8) to perform real-time crowd density estimations, track spatial occupancy counts, and log safety incidents to prevent dangerous crowd build-ups.

This project is built using a **hybrid workflow**:
1. **Google Colab (GPU)** for training/fine-tuning the deep learning detection model.
2. **Local Machine (CPU/GPU)** for processing camera streams, calculating heatmaps, and running the interactive web dashboard interface.

---

## 🚀 Quick Start (Local Web Dashboard)

You can run the web dashboard immediately. It will search for custom weights, download pre-trained weights (`yolov8n.pt`) as a fallback, and automatically initialize a **synthetic crowd simulation stream** if no physical webcam is connected.

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

## 🧠 Google Colab Training Workflow

To fine-tune a model on custom crowd/person datasets:

1. Locate the pre-configured training notebook in this folder: `train_crowd_detector.ipynb`.
2. Go to [Google Colab](https://colab.research.google.com/) and upload `train_crowd_detector.ipynb`.
3. Select a **GPU Runtime** (Runtime > Change runtime type > GPU).
4. Run the setup cells to connect your Google Drive and install the training libraries.
5. Provide your dataset links (Roboflow exports or zip files on Drive) and run the training cell.
6. Once training completes, download the resulting model weights file: `best.pt`.
7. Move `best.pt` to the `models/` directory in this workspace folder on your computer:
   ```text
   Crowd Detection & Prevention/
   └── models/
       └── best.pt
   ```
8. Restart your local dashboard. It will automatically detect and load your custom fine-tuned weights!

---

## 🛠️ Project Structure

- `src/detector.py`: YOLOv8 wrapper to run inference on camera frames and filter human bounding boxes.
- `src/density.py`: Grid density calculations and visual smooth Gaussian heatmap overlay generation.
- `src/alerts.py`: Threshold tracking and incident log management.
- `webapp/main.py`: FastAPI server serving templates, endpoints, configuration posts, and MJPEG video streaming.
- `webapp/templates/index.html`: Sleek dark-mode dashboard HTML design with glassmorphic cards.
- `webapp/static/`: Layout styles (`css/style.css`) and metrics updater scripts (`js/dashboard.js`).
