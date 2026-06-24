import os
import cv2
import numpy as np
import time
import asyncio
import csv
from datetime import datetime
import torch
from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import modules from src/
from src.detector import CrowdDetector
from src.alerts import AlertManager
from src.csrnet_model import CSRNet, load_csrnet_model
from src.csrnet_inference import estimate_density, get_zone_densities, density_map_to_heatmap
from src.density import get_yolo_zone_counts, should_use_csrnet, get_zone_risk_scores

app = FastAPI(title="CrowdShield AI Backend")

# Setup directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "webapp", "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "webapp", "templates")

# Mount Static Files and Templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize AI / CV Engines
detector = CrowdDetector(imgsz=960, confidence_threshold=0.25)
alert_manager = AlertManager(max_capacity=10, density_limit=5.0, trigger_delay_seconds=3.0)

# Initialize CSRNet Engine
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = os.path.join(BASE_DIR, "models", "csrnet_shanghaitech.pth")

if os.path.exists(weights_path):
    print(f"FastAPI: Loading pretrained CSRNet weights from {weights_path} on {device}")
    try:
        csrnet_model = load_csrnet_model(weights_path, device)
    except Exception as e:
        print(f"FastAPI: Error loading CSRNet weights: {e}. Falling back to default initialization.")
        csrnet_model = CSRNet(load_weights=False).to(device)
        csrnet_model.eval()
else:
    print("FastAPI: CSRNet weights not found at 'models/csrnet_shanghaitech.pth'. Initializing default/ImageNet weights.")
    csrnet_model = CSRNet(load_weights=False).to(device)
    csrnet_model.eval()

# CSV Logging configuration
CSV_LOG_PATH = os.path.join(BASE_DIR, "crowd_comparison.csv")

def log_counts_to_csv(yolo_count, csrnet_count, model_selected):
    """
    Logs comparison metrics to a CSV file.
    """
    file_exists = os.path.exists(CSV_LOG_PATH)
    try:
        with open(CSV_LOG_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Write header if file is new
                writer.writerow(['timestamp', 'yolo_count', 'csrnet_count', 'which_model_selected'])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            writer.writerow([timestamp, yolo_count, csrnet_count, model_selected])
    except Exception as e:
        print(f"Failed to write to CSV log: {e}")

def process_frame(frame):
    """
    Processes a single frame running both YOLO and CSRNet models.
    Applies heuristic switching logic to decide which model to trust,
    logs comparative counts to a CSV file, calculates per-zone risk status,
    and returns a structured dict of results.
    
    Args:
        frame (np.ndarray): BGR OpenCV frame.
        
    Returns:
        dict: {
            'count': float,           # Selected model's total crowd count
            'zone_grid': np.ndarray,   # Selected model's per-zone count grid (2D array)
            'risk_per_zone': np.ndarray, # 2D array of risk levels ("safe"/"caution"/"danger")
            'model_used': str,         # "YOLO" or "CSRNet"
            'heatmap_image': np.ndarray or None, # Heatmap overlay (only if CSRNet was used)
            'boxes': list or None      # List of YOLO bounding boxes (only if YOLO was used)
        }
    """
    if frame is None:
        return {
            'count': 0.0,
            'zone_grid': np.zeros((3, 3), dtype=np.float32),
            'risk_per_zone': np.full((3, 3), "safe", dtype=object),
            'model_used': "YOLO",
            'heatmap_image': None,
            'boxes': []
        }
        
    # 1. Run YOLO detection
    yolo_detections = detector.detect(frame)
    yolo_count = len(yolo_detections)
    yolo_boxes = [det['bbox'] for det in yolo_detections]
    
    # 2. Run CSRNet estimation
    density_map, csrnet_count = estimate_density(csrnet_model, frame, device)
    
    # 3. Determine which model output to trust
    # Default switching parameters: count threshold 15, overlap IoU threshold 0.3
    use_csrnet = should_use_csrnet(
        yolo_count, 
        yolo_boxes, 
        threshold=15,          # Tunable switching threshold
        overlap_threshold=0.3  # Tunable IoU threshold
    )
    
    model_selected = "CSRNet" if use_csrnet else "YOLO"
    
    # 4. Log both counts to CSV
    log_counts_to_csv(yolo_count, csrnet_count, model_selected)
    
    # 5. Build selected model's zone grid (3x3 grid)
    grid_rows, grid_cols = 3, 3
    if use_csrnet:
        zone_grid = get_zone_densities(density_map, grid_rows=grid_rows, grid_cols=grid_cols)
        selected_count = csrnet_count
        heatmap_image = density_map_to_heatmap(density_map, frame, alpha=0.5)
        boxes_out = None
    else:
        zone_grid = get_yolo_zone_counts(yolo_boxes, frame.shape, grid_rows=grid_rows, grid_cols=grid_cols)
        selected_count = float(yolo_count)
        heatmap_image = None
        boxes_out = yolo_boxes
        
    # 6. Calculate risk scores per zone
    # Configurable thresholds: caution level 3, danger level 6.0
    risk_thresholds = {
        'caution': 3.0,
        'danger': 6.0
    }
    risk_per_zone = get_zone_risk_scores(zone_grid, risk_thresholds)
    
    return {
        'count': selected_count,
        'zone_grid': zone_grid,
        'risk_per_zone': risk_per_zone,
        'model_used': model_selected,
        'heatmap_image': heatmap_image,
        'boxes': boxes_out,
        'raw_yolo_detections': yolo_detections,
        'raw_density_map': density_map
    }

def draw_grid_overlay(frame, grid):
    """
    Draws grid boundaries and count values on the frame.
    """
    annotated_frame = frame.copy()
    h, w = frame.shape[:2]
    rows, cols = grid.shape
    
    cell_h = h // rows
    cell_w = w // cols
    
    # Draw vertical lines
    for col in range(1, cols):
        x = col * cell_w
        cv2.line(annotated_frame, (x, 0), (x, h), (100, 100, 100), 1, cv2.LINE_AA)
        
    # Draw horizontal lines
    for row in range(1, rows):
        y = row * cell_h
        cv2.line(annotated_frame, (0, y), (w, y), (100, 100, 100), 1, cv2.LINE_AA)
        
    # Draw values in the center of each cell
    for r in range(rows):
        for c in range(cols):
            val = grid[r, c]
            text = f"{val:.1f}" if val % 1 != 0 else f"{int(val)}"
            
            text_x = int(c * cell_w + cell_w / 2 - 10)
            text_y = int(r * cell_h + cell_h / 2 + 5)
            
            # Color based on count thresholds (caution: 3, danger: 6)
            if val >= 6.0:
                color = (0, 0, 255) # Red for danger
            elif val >= 3.0:
                color = (0, 255, 255) # Yellow for caution
            else:
                color = (0, 255, 0) # Green for safe
                
            cv2.putText(annotated_frame, text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            
    return annotated_frame

# Global metrics cache
metrics_cache = {
    "current_count": 0,
    "peak_density": 0.0,
    "status": "NORMAL",
    "status_message": "System operating normally.",
    "model_used": "YOLO",
    "risk_per_zone": [],
    "alert_history": []
}

class ConfigPayload(BaseModel):
    max_capacity: int
    density_limit: float
    trigger_delay: float
    confidence_threshold: float
    imgsz: int
    model_type: str

@app.get("/")
async def get_dashboard(request: Request):
    """
    Renders the HTML Dashboard.
    """
    return templates.TemplateResponse(request, "index.html", {"request": request})

@app.get("/api/config")
async def get_config():
    """
    Retrieves current backend safety and detector configurations.
    """
    return {
        "max_capacity": alert_manager.max_capacity,
        "density_limit": alert_manager.density_limit,
        "trigger_delay": alert_manager.trigger_delay_seconds,
        "confidence_threshold": getattr(detector, 'confidence_threshold', 0.25),
        "imgsz": getattr(detector, 'imgsz', 960),
        "model_type": getattr(detector, 'model_type', 'general')
    }

@app.get("/api/metrics")
async def get_metrics():
    """
    Retrieves current crowd metrics and alert status.
    """
    # Fetch latest history log from manager
    metrics_cache["alert_history"] = alert_manager.get_history()
    return metrics_cache

@app.post("/api/config")
async def update_config(payload: ConfigPayload):
    """
    Updates warning limits dynamically.
    """
    alert_manager.max_capacity = payload.max_capacity
    alert_manager.density_limit = payload.density_limit
    alert_manager.trigger_delay_seconds = payload.trigger_delay
    detector.confidence_threshold = payload.confidence_threshold
    detector.imgsz = payload.imgsz
    detector.model_type = payload.model_type
    return {"status": "success", "config": payload}

# Simulated crowd coordinates for generator fallback
simulated_people = []
for i in range(12):
    simulated_people.append({
        "x": 100 + (i * 45) % 450,
        "y": 150 + (i * 35) % 250,
        "dx": 1.5 if i % 2 == 0 else -1.5,
        "dy": 1.0 if i % 3 == 0 else -1.0
    })

def generate_simulated_frame(w=640, h=480):
    """
    Generates a synthetic frame representing an indoor space with walking people.
    Used as an automated fallback if no physical webcam or video is available.
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Draw floor grid background
    cv2.rectangle(frame, (10, 10), (w-10, h-10), (20, 24, 33), -1)
    for i in range(0, w, 40):
        cv2.line(frame, (i, 10), (i, h-10), (32, 38, 50), 1)
    for j in range(0, h, 40):
        cv2.line(frame, (10, j), (w-10, j), (32, 38, 50), 1)
        
    # Draw simulated boundary walls
    cv2.rectangle(frame, (10, 10), (w-10, h-10), (67, 56, 202), 2)
    cv2.putText(frame, "SIMULATED VIEWPORT (No Camera Detected)", (20, h - 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (156, 163, 175), 1, cv2.LINE_AA)

    # Move simulated entities
    for idx, p in enumerate(simulated_people):
        p["x"] += p["dx"]
        p["y"] += p["dy"]
        
        # Wall bounce detection
        if p["x"] < 30 or p["x"] > w - 30:
            p["dx"] *= -1
        if p["y"] < 50 or p["y"] > h - 50:
            p["dy"] *= -1
            
        # Draw simulated human figure bounding box (so detector can find it)
        # Using a simple human shape color block
        box_w = 36
        box_h = 75
        x1 = int(p["x"] - box_w/2)
        y1 = int(p["y"] - box_h/2)
        x2 = int(p["x"] + box_w/2)
        y2 = int(p["y"] + box_h/2)
        
        # Draw "person" (head & body blocks)
        # Head
        cv2.circle(frame, (int(p["x"]), y1 + 15), 10, (220, 200, 180), -1)
        # Torso / Clothes (bright jackets)
        color = (180, 100, 50) if idx % 3 == 0 else ((50, 150, 100) if idx % 3 == 1 else (80, 80, 200))
        cv2.rectangle(frame, (x1, y1 + 25), (x2, y2), color, -1)
        
    return frame


def frame_generator(mode: str):
    """
    Video streaming generator. 
    Attempts to read from the camera (index 0). If not available, streams synthetic simulation.
    """
    # Open camera stream
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    use_simulation = not cap.isOpened()
    if use_simulation:
        print("FastAPI: Webcam not detected. Falling back to synthetic crowd simulation stream.")
    else:
        print("FastAPI: Webcam initialized successfully.")
        # Set 720p HD resolution to improve sharpness, details, and camera autofocus
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
    try:
        while True:
            start_time = time.time()
            
            if use_simulation:
                frame = generate_simulated_frame()
            else:
                ret, frame = cap.read()
                if not ret:
                    # In case of webcam error mid-stream, fallback
                    frame = generate_simulated_frame()
                    
            # Run the dual-model processing pipeline on the current frame
            result = process_frame(frame)
            
            current_count = result['count']
            # Peak density is defined as the maximum value inside any grid cell of the active model
            peak_density = float(np.max(result['zone_grid'])) if result['zone_grid'].size > 0 else 0.0
            
            # Apply Selected View Mode
            processed_frame = frame
            
            if mode == "heatmap":
                # Heatmap View: Always show the CSRNet density heatmap overlay
                if result['heatmap_image'] is not None:
                    processed_frame = result['heatmap_image']
                else:
                    # If YOLO was used for metrics but the user requested heatmap view, 
                    # we generate the heatmap from CSRNet's density map since both models ran.
                    processed_frame = density_map_to_heatmap(result['raw_density_map'], frame, alpha=0.5)
            elif mode == "grid":
                # Grid Cell View: Show the spatial grid overlay from the active model
                processed_frame = draw_grid_overlay(frame, result['zone_grid'])
            else:
                # Raw Box View: Draw standard YOLO bounding boxes on the frame
                processed_frame = detector.draw_detections(frame, result['raw_yolo_detections'])
                
            # Update alerting status (alert_manager runs on the active model's decision output)
            alert_status = alert_manager.update(current_count, peak_density)
            
            # Write metrics cache
            metrics_cache["current_count"] = int(current_count) if current_count % 1 == 0 else round(current_count, 1)
            metrics_cache["peak_density"] = round(peak_density, 2)
            metrics_cache["status"] = alert_status["status"]
            metrics_cache["status_message"] = alert_status["message"]
            metrics_cache["model_used"] = result['model_used']
            metrics_cache["risk_per_zone"] = result['risk_per_zone'].tolist()
            
            # Encode image to JPEG to transmit over HTTP
            ret, jpeg = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
                
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Control frame rate (target ~20-25 FPS)
            elapsed = time.time() - start_time
            sleep_time = max(0.01, 0.04 - elapsed)
            time.sleep(sleep_time)
            
    finally:
        if not use_simulation:
            cap.release()

@app.get("/video_feed")
def video_feed(mode: str = "heatmap"):
    """
    Returns the real-time processed camera feed as a Multipart HTTP stream.
    """
    return StreamingResponse(frame_generator(mode), 
                             media_type="multipart/x-mixed-replace; boundary=frame")
