import os
import cv2
import numpy as np
import time
import asyncio
from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import modules from src/
from src.detector import CrowdDetector
from src.density import DensityEstimator
from src.alerts import AlertManager

app = FastAPI(title="CrowdShield AI Backend")

# Setup directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "webapp", "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "webapp", "templates")

# Mount Static Files and Templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize AI / CV Engines
detector = CrowdDetector()
density_estimator = DensityEstimator(grid_rows=8, grid_cols=8)
alert_manager = AlertManager(max_capacity=10, density_limit=5.0, trigger_delay_seconds=3.0)

# Global metrics cache
metrics_cache = {
    "current_count": 0,
    "peak_density": 0.0,
    "status": "NORMAL",
    "status_message": "System operating normally.",
    "alert_history": []
}

class ConfigPayload(BaseModel):
    max_capacity: int
    density_limit: float
    trigger_delay: float

@app.get("/")
async def get_dashboard(request: Request):
    """
    Renders the HTML Dashboard.
    """
    return templates.TemplateResponse(request, "index.html", {"request": request})

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
                    
            # Run YOLO Person detection
            # If using simulation, the YOLO detector will detect the colored rectangle blocks representing people
            detections = detector.detect(frame)
            
            # Apply Selected View Mode
            processed_frame = frame
            peak_density = 0.0
            
            if mode == "heatmap":
                # Generate smooth density heatmap
                processed_frame, peak_density = density_estimator.generate_heatmap(frame, detections)
            elif mode == "grid":
                # Generate density grid cell count lines
                grid = density_estimator.calculate_grid_density(frame.shape, detections)
                processed_frame = density_estimator.draw_grid_boundaries(frame, grid)
                # Calculate peak cell value
                peak_density = float(np.max(grid)) if len(grid) > 0 else 0.0
            else: # "normal"
                # Draw standard bounding boxes
                processed_frame = detector.draw_detections(frame, detections)
                peak_density = float(len(detections))
                
            # Update alerting status
            current_count = len(detections)
            alert_status = alert_manager.update(current_count, peak_density)
            
            # Write metrics cache
            metrics_cache["current_count"] = current_count
            metrics_cache["peak_density"] = peak_density
            metrics_cache["status"] = alert_status["status"]
            metrics_cache["status_message"] = alert_status["message"]
            
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
