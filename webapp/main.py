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
from src.density import (
    get_yolo_zone_counts,
    should_use_csrnet,
    get_zone_risk_scores,
    get_smoothed_model_decision,
    get_smoothed_count
)
from src.sahi_detector import run_sahi_detection, SLICE_CONFIG

app = FastAPI(title="CrowdShield AI Backend")

# Setup directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "webapp", "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "webapp", "templates")

# Mount Static Files and Templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# =====================================================================
# EDITABLE VIDEO SOURCE CONFIGURATION
# =====================================================================
# Set to 0 to use your webcam, or a string path to a video file.
# Example: VIDEO_SOURCE = r"C:\path\to\your\video.mp4"
VIDEO_SOURCE = r"E:\test_video.mp4"

# Auto-resolve directory to video file if needed
if isinstance(VIDEO_SOURCE, str) and os.path.isdir(VIDEO_SOURCE):
    files = os.listdir(VIDEO_SOURCE)
    video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    if video_files:
        dir_name = os.path.basename(os.path.normpath(VIDEO_SOURCE))
        best_match = None
        for vf in video_files:
            if os.path.splitext(vf)[0] == dir_name or vf == dir_name:
                best_match = vf
                break
        if not best_match:
            best_match = video_files[0]
        VIDEO_SOURCE = os.path.join(VIDEO_SOURCE, best_match)
        print(f"FastAPI: Auto-resolved VIDEO_SOURCE directory to file: {VIDEO_SOURCE}")
# =====================================================================

# Initialize AI / CV Engines
detector = CrowdDetector(imgsz=960, confidence_threshold=0.25)
alert_manager = AlertManager(max_capacity=1000, caution_at=70, density_limit=5.0, trigger_delay_seconds=20.0)

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

# Global configuration for frame skip/processing interval
FRAME_PROCESSING_INTERVAL = 1  # Process every Nth frame (1 = every frame)

# Global state tracking for manual override/model selection
current_detection_mode = "auto"  # "auto", "yolo", "sahi", "csrnet"

# Global state tracking for per-zone alerts
zone_states = {}  # key: (row, col), value: {'logged_tier': 'safe', 'pending_tier': None, 'start_time': None}

# Global rolling histories for temporal smoothing
model_decision_history = []
count_history = []

def process_frame(frame, detection_mode="auto"):
    """
    Processes a single frame running YOLO, SAHI, or CSRNet models.
    Applies heuristic switching logic to decide which model to trust,
    logs comparative counts to a CSV file, calculates per-zone risk status,
    and returns a structured dict of results.
    
    Args:
        frame (np.ndarray): BGR OpenCV frame.
        detection_mode (str): Force model type ("auto", "yolo", "csrnet").
        
    Returns:
        dict: {
            'count': float,           # Selected model's total crowd count
            'zone_grid': np.ndarray,   # Selected model's per-zone count grid (2D array)
            'risk_per_zone': np.ndarray, # 2D array of risk levels ("safe"/"caution"/"danger")
            'model_used': str,         # "YOLO", "YOLO + SAHI", or "CSRNet"
            'heatmap_image': np.ndarray or None, # Heatmap overlay (only if CSRNet was used)
            'boxes': list or None      # List of YOLO bounding boxes (only if YOLO/SAHI was used)
        }
    """
    global model_decision_history, count_history
    if frame is None:
        return {
            'count': 0.0,
            'zone_grid': np.zeros((3, 3), dtype=np.float32),
            'risk_per_zone': np.full((3, 3), "safe", dtype=object),
            'model_used': "YOLO",
            'heatmap_image': None,
            'boxes': []
        }
        
    # Get camera environment config
    env = getattr(detector, 'model_type', 'general')  # "general", "venue", or "aerial"
    slice_params = SLICE_CONFIG.get(env, None)
    
    yolo_count = 0
    yolo_boxes = []
    yolo_detections = []
    density_map = None
    csrnet_count = 0.0
    ran_sahi = False
    
    # 1. Run YOLO/SAHI first if Auto or forced YOLO is active.
    # Note: Running YOLO+SAHI (or plain YOLO if camera_environment is "general") first ensures we get
    # the best available YOLO-based count for this frame.
    # CRITICAL: This means YOLO/SAHI runs even on frames that end up using CSRNet's result. This adds
    # computational cost, which could be optimized later (e.g., a cheap quick density pre-check),
    # but correctness of the switching decision comes first.
    run_yolo = (detection_mode in ("yolo", "auto"))
    if run_yolo:
        if slice_params is not None:
            yolo_path = os.path.join(BASE_DIR, "models", "best.pt")
            if not os.path.exists(yolo_path):
                yolo_path = os.path.join(BASE_DIR, "models", "yolo11m_best.pt")
            if not os.path.exists(yolo_path):
                yolo_path = "yolo11m.pt"
            yolo_detections = run_sahi_detection(
                model_path=yolo_path,
                frame=frame,
                slice_height=slice_params["slice_height"],
                slice_width=slice_params["slice_width"],
                overlap_ratio=slice_params["overlap_ratio"]
            )
            ran_sahi = True
        else:
            yolo_detections = detector.detect(frame)

        yolo_count = len(yolo_detections)
        yolo_boxes = [det['bbox'] for det in yolo_detections]
        
    # 2. Determine which model output to trust.
    # CRITICAL: The switching decision must always use the most accurate available YOLO-based count
    # for the current camera environment, never a plain unsliced count when SAHI would normally apply,
    # otherwise dense scenes can be incorrectly judged as low-density and never hand off to CSRNet.
    if detection_mode == "csrnet":
        use_csrnet = True
    elif detection_mode == "yolo":
        use_csrnet = False
    else: # auto
        use_csrnet = get_smoothed_model_decision(
            yolo_count, 
            yolo_boxes, 
            model_decision_history, 
            window_size=10,
            detection_mode=detection_mode
        )
        
    if use_csrnet:
        model_selected = "CSRNet"
    else:
        model_selected = "YOLO + SAHI" if ran_sahi else "YOLO"
        
    # 3. Run CSRNet estimation only if CSRNet is chosen/forced
    run_csrnet = (detection_mode == "csrnet") or (detection_mode == "auto" and use_csrnet)
    if run_csrnet:
        density_map, csrnet_count = estimate_density(csrnet_model, frame, device)
    
    # 4. Log comparative metrics to CSV
    raw_use_csrnet = should_use_csrnet(
        yolo_count, 
        yolo_boxes, 
        threshold=50, 
        overlap_threshold=0.3,
        detection_mode=detection_mode
    )
    raw_model_selected = "CSRNet" if raw_use_csrnet else ("YOLO + SAHI" if ran_sahi else "YOLO")
    log_counts_to_csv(yolo_count, csrnet_count, raw_model_selected)
    
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
        
    # Smooth the count display
    smoothed_count = get_smoothed_count(selected_count, count_history, window_size=5)
        
    # 6. Calculate risk scores per zone based on local density limit
    caution_pct = float(alert_manager.caution_at) if hasattr(alert_manager, 'caution_at') else 70.0
    risk_thresholds = {
        'caution': (caution_pct / 100.0) * alert_manager.density_limit,
        'danger': alert_manager.density_limit
    }
    risk_per_zone = get_zone_risk_scores(zone_grid, risk_thresholds)
    return {
        'count': smoothed_count,
        'zone_grid': zone_grid,
        'risk_per_zone': risk_per_zone,
        'model_used': model_selected,
        'heatmap_image': heatmap_image,
        'boxes': boxes_out,
        'raw_yolo_detections': yolo_detections,
        'raw_density_map': density_map if density_map is not None else np.zeros((96, 128), dtype=np.float32)
    }

def draw_grid_overlay(frame, grid, risk_grid):
    """
    Draws grid boundaries and count values on the frame, styled according to the dynamic risk grid.
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
            risk = risk_grid[r, c]
            text = f"{val:.1f}" if val % 1 != 0 else f"{int(val)}"
            
            text_x = int(c * cell_w + cell_w / 2 - 10)
            text_y = int(r * cell_h + cell_h / 2 + 5)
            
            # Color based on dynamic risk grid
            if risk == 'danger':
                color = (0, 0, 255) # Red for danger
            elif risk == 'caution':
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
    caution_at: int
    trigger_delay: float
    confidence_threshold: float
    imgsz: int
    model_type: str
    detection_mode: str = "auto"

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
        "caution_at": getattr(alert_manager, 'caution_at', 70),
        "trigger_delay": alert_manager.trigger_delay_seconds,
        "confidence_threshold": getattr(detector, 'confidence_threshold', 0.25),
        "imgsz": getattr(detector, 'imgsz', 960),
        "model_type": getattr(detector, 'model_type', 'general'),
        "detection_mode": current_detection_mode
    }

@app.get("/api/metrics")
async def get_metrics():
    """
    Retrieves current crowd metrics and alert status.
    """
    # Fetch latest history log from manager
    metrics_cache["alert_history"] = alert_manager.get_history()
    metrics_cache["caution_at"] = getattr(alert_manager, 'caution_at', 70)
    return metrics_cache

@app.get("/api/current_status")
async def get_current_status():
    """
    Exposes process_frame()'s full current output dict in JSON format.
    """
    # Fetch alert history to sync incident logs as well
    alert_history = alert_manager.get_history()
    return {
        "count": metrics_cache.get("current_count", 0),
        "peak_density": metrics_cache.get("peak_density", 0.0),
        "status": metrics_cache.get("status", "NORMAL"),
        "status_message": metrics_cache.get("status_message", ""),
        "model_used": metrics_cache.get("model_used", "YOLO"),
        "risk_per_zone": metrics_cache.get("risk_per_zone", []),
        "boxes": metrics_cache.get("boxes", []),
        "avg_confidence": metrics_cache.get("avg_confidence", "N/A"),
        "zone_grid": metrics_cache.get("zone_grid", []),
        "alert_history": alert_history,
        "max_capacity": alert_manager.max_capacity,
        "caution_at": getattr(alert_manager, 'caution_at', 70)
    }

@app.post("/api/config")
async def update_config(payload: ConfigPayload):
    """
    Updates warning limits dynamically.
    """
    global current_detection_mode
    alert_manager.max_capacity = payload.max_capacity
    alert_manager.caution_at = payload.caution_at
    alert_manager.trigger_delay_seconds = payload.trigger_delay
    detector.confidence_threshold = payload.confidence_threshold
    detector.imgsz = payload.imgsz
    detector.model_type = payload.model_type
    current_detection_mode = payload.detection_mode
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
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    # Check if camera opened successfully
    use_simulation = not cap.isOpened()
    if use_simulation:
        source_name = "Video File" if isinstance(VIDEO_SOURCE, str) else "Webcam"
        print(f"FastAPI: {source_name} source not loaded. Falling back to synthetic crowd simulation stream.")
    else:
        source_name = f"Video File ({VIDEO_SOURCE})" if isinstance(VIDEO_SOURCE, str) else "Webcam"
        print(f"FastAPI: {source_name} source initialized successfully.")
        # If it's a webcam, set resolution
        if not isinstance(VIDEO_SOURCE, str):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
    frame_counter = 0
    last_result = {
        'count': 0.0,
        'zone_grid': np.zeros((3, 3), dtype=np.float32),
        'risk_per_zone': np.full((3, 3), "safe", dtype=object),
        'model_used': "YOLO",
        'heatmap_image': None,
        'boxes': [],
        'raw_yolo_detections': [],
        'raw_density_map': np.zeros((96, 128), dtype=np.float32)
    }
    
    try:
        while True:
            start_time = time.time()
            
            if use_simulation:
                frame = generate_simulated_frame()
            else:
                ret, frame = cap.read()
                if not ret:
                    # If it's a video file, loop it back to the beginning indefinitely
                    if isinstance(VIDEO_SOURCE, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    
                    # In case of webcam error mid-stream or empty video, fallback
                    if not ret:
                        frame = generate_simulated_frame()
                    
            # Run the dual-model processing pipeline on the current frame based on configurable interval
            if frame_counter % FRAME_PROCESSING_INTERVAL == 0:
                try:
                    result = process_frame(frame, detection_mode=current_detection_mode)
                    last_result = result
                except Exception as e:
                    print(f"FastAPI: Error during process_frame: {e}")
                    result = last_result
            else:
                result = last_result
                
            frame_counter += 1
            
            current_count = result['count']
            # Peak density is defined as the maximum value inside any grid cell of the active model
            peak_density = float(np.max(result['zone_grid'])) if result['zone_grid'].size > 0 else 0.0
            
            # Apply Selected View Mode
            processed_frame = frame.copy()
            model_used = result['model_used']
            
            if mode == "heatmap":
                # Heatmap View: Show the CSRNet density heatmap overlay if CSRNet is active
                if model_used == "CSRNet":
                    if result['heatmap_image'] is not None:
                        processed_frame = result['heatmap_image']
                    else:
                        processed_frame = density_map_to_heatmap(result['raw_density_map'], frame, alpha=0.5)
                else:
                    # If YOLO/SAHI is active, show the raw frame with a status label stating the active model
                    cv2.rectangle(processed_frame, (10, 10), (450, 45), (15, 23, 42), -1)
                    cv2.putText(processed_frame, f"Currently using {model_used}", (20, 32), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (56, 189, 248), 2, cv2.LINE_AA)
            elif mode == "grid":
                # Grid Cell View: Show the spatial grid overlay regardless of which model is active
                processed_frame = draw_grid_overlay(frame, result['zone_grid'], result['risk_per_zone'])
            else:
                # Raw Box View: Show the YOLO/SAHI boxes if active
                if model_used in ("YOLO", "SAHI", "YOLO + SAHI"):
                    processed_frame = detector.draw_detections(frame, result['raw_yolo_detections'])
                else:
                    # If CSRNet is active, show the raw frame with a status label stating "Currently using CSRNet (high density)"
                    cv2.rectangle(processed_frame, (10, 10), (450, 45), (15, 23, 42), -1)
                    cv2.putText(processed_frame, "Currently using CSRNet (high density)", (20, 32), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (56, 189, 248), 2, cv2.LINE_AA)
                
            # Update alerting status (alert_manager runs on the active model's decision output)
            alert_status = alert_manager.update(current_count, peak_density)
            
            # Write metrics cache
            metrics_cache["current_count"] = int(current_count) if current_count % 1 == 0 else round(current_count, 1)
            metrics_cache["peak_density"] = round(peak_density, 2)
            metrics_cache["status"] = alert_status["status"]
            metrics_cache["status_message"] = alert_status["message"]
            metrics_cache["model_used"] = model_used
            metrics_cache["detection_mode"] = current_detection_mode
            metrics_cache["risk_per_zone"] = result['risk_per_zone'].tolist()
            metrics_cache["zone_grid"] = result['zone_grid'].tolist()
            metrics_cache["boxes"] = result['boxes'] if result['boxes'] is not None else []
            metrics_cache["max_capacity"] = alert_manager.max_capacity
            metrics_cache["caution_at"] = getattr(alert_manager, 'caution_at', 70)
            
            # Compute real YOLO/SAHI detection average confidence
            yolo_det = result['raw_yolo_detections']
            if yolo_det:
                avg_conf = np.mean([d['confidence'] for d in yolo_det]) * 100.0
                metrics_cache["avg_confidence"] = f"{int(avg_conf)}%"
            else:
                metrics_cache["avg_confidence"] = "N/A"
            
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
