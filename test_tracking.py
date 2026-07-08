"""
Test suite and visualization runner for crowd tracking, ByteTrack, and velocity statistics.
Loads the YOLO tracking model to perform real-time detection on a video stream or camera feed.
Includes a synthetic fallback generator to simulate crowd movements if no camera is connected.
"""

import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
from src.tracker import track_frame
from src.velocity import update_track_history, get_zone_velocity_stats

# =====================================================================
# EDITABLE INPUT PATH CONFIGURATION
# =====================================================================
# Set to 0 to use webcam, or path to a video file
INPUT_PATH = r"E:\test_video_3.mp4"
CONFIDENCE_THRESHOLD = 0.15          # Lower threshold (e.g. 0.10 - 0.20) to detect people in low light
ENHANCE_LOW_LIGHT = True             # Apply CLAHE to boost contrast in dark areas
# =====================================================================

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
    cv2.putText(frame, "SIMULATED VIEWPORT (No Camera/Video Detected)", (20, h - 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (156, 163, 175), 1, cv2.LINE_AA)

    # Move simulated entities
    for idx, p in enumerate(simulated_people):
        p["x"] += p["dx"]
        p["y"] += p["dy"]
        
        # Wall bounce detection
        if p["x"] < 30 or p["x"] > w - 30:
            p["dx"] *= -1
        if p["y"] < 50 or p["y"] > h - 50:
            p["dy"] *= -1
            
        # Draw "person" (head & body blocks)
        box_w = 36
        box_h = 75
        x1 = int(p["x"] - box_w/2)
        y1 = int(p["y"] - box_h/2)
        x2 = int(p["x"] + box_w/2)
        y2 = int(p["y"] + box_h/2)
        
        # Head
        cv2.circle(frame, (int(p["x"]), y1 + 15), 10, (220, 200, 180), -1)
        # Torso
        color = (180, 100, 50) if idx % 3 == 0 else ((50, 150, 100) if idx % 3 == 1 else (80, 80, 200))
        cv2.rectangle(frame, (x1, y1 + 25), (x2, y2), color, -1)
        
    return frame


def draw_overlays(frame, tracked_objects, history_dict, avg_speeds, dir_variances, grid_rows=3, grid_cols=3):
    """
    Draws tracking boxes, track IDs, motion trajectories, and spatial grid velocity stats.
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    
    # 1. Draw spatial grid lines
    for col in range(1, grid_cols):
        x = col * cell_w
        cv2.line(annotated, (x, 0), (x, h), (120, 120, 120), 1, cv2.LINE_AA)
    for row in range(1, grid_rows):
        y = row * cell_h
        cv2.line(annotated, (0, y), (w, y), (120, 120, 120), 1, cv2.LINE_AA)
        
    # 2. Draw trajectories (smooth trails) for each person
    for track_id, history in history_dict.items():
        if len(history) >= 2:
            points = [item['center'] for item in history]
            for idx in range(len(points) - 1):
                # Fade color based on age (older lines are darker/greener)
                alpha = int(255 * (idx / len(points)))
                color = (0, alpha, 255 - alpha)  # color transition
                cv2.line(annotated, points[idx], points[idx+1], color, 2, cv2.LINE_AA)
                
    # 3. Draw bounding boxes + track IDs
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj['bbox']
        track_id = obj['track_id']
        
        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 128, 0), 2, cv2.LINE_AA)
        
        # ID tag
        tag = f"ID: {track_id}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), (255, 128, 0), -1)
        cv2.putText(annotated, tag, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        
    # 4. Draw spatial zone velocity and direction variance statistics
    for r in range(grid_rows):
        for c in range(grid_cols):
            speed = avg_speeds[r][c]
            var = dir_variances[r][c]
            
            # Put text in the center of each cell
            cx = int(c * cell_w + cell_w / 2)
            cy = int(r * cell_h + cell_h / 2)
            
            # Speed text
            speed_str = f"Avg Spd: {speed:.1f} px/s"
            # Variance text
            var_str = f"Dir Var: {var:.2f}"
            
            # Semi-transparent backing rectangle for readability
            for text, offset_y in [(speed_str, -10), (var_str, 12)]:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (cx - tw//2 - 4, cy + offset_y - th - 2), 
                              (cx + tw//2 + 4, cy + offset_y + 2), (30, 30, 30), -1)
                cv2.putText(annotated, text, (cx - tw//2, cy + offset_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                            
    return annotated

def main():
    print(f"Loading YOLO tracking model...")
    # Initialize detector / tracker
    model_path = os.path.join("models", "yolo11m_best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join("models", "best.pt")
    if not os.path.exists(model_path):
        model_path = "yolo11m.pt"
    print(f"Loading YOLO tracking model from: {model_path} ...")
    model = YOLO(model_path)
    
    # Initialize track history dictionary
    history_dict = {}
    
    # Open camera / video stream or check for fallback
    use_simulation = False
    cap = None
    if isinstance(INPUT_PATH, str) and (not os.path.exists(INPUT_PATH) or os.path.getsize(INPUT_PATH) == 0):
        print(f"Warning: Input path '{INPUT_PATH}' is missing or is a 0-byte file.")
        print("Falling back to synthetic crowd simulation stream.")
        use_simulation = True
    else:
        cap = cv2.VideoCapture(INPUT_PATH)
        if not cap.isOpened():
            print(f"Warning: Could not open input source: {INPUT_PATH}. Falling back to synthetic crowd simulation stream.")
            use_simulation = True
        else:
            print(f"Successfully opened input source: {INPUT_PATH}")
            
    print("Press 'q' in the viewport window to quit.")
    cv2.namedWindow("Crowd Tracking & Velocity Metrics", cv2.WINDOW_NORMAL)
    
    while True:
        if use_simulation:
            frame = generate_simulated_frame()
        else:
            ret, frame = cap.read()
            if not ret:
                # If it's a video file, loop it back
                if isinstance(INPUT_PATH, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if not ret:
                    break
                    
        now = time.time()
        
        # Apply CLAHE contrast enhancement for low light
        if not use_simulation and ENHANCE_LOW_LIGHT:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            # ClipLimit sets contrast enhancement threshold; tileGridSize divides image into local zones
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            frame_to_process = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        else:
            frame_to_process = frame
            
        # 1. Run ByteTrack
        tracked_objects = track_frame(model, frame_to_process, imgsz=960, conf=CONFIDENCE_THRESHOLD)
        
        # 2. Update track history trajectories
        active_ids = set()
        for obj in tracked_objects:
            track_id = obj['track_id']
            center = obj['center_point']
            update_track_history(track_id, center, now, history_dict, max_history=15)
            active_ids.add(track_id)
            
        # Optional: Prune trajectories of lost tracks to save memory
        all_ids = list(history_dict.keys())
        for tid in all_ids:
            if tid not in active_ids and len(history_dict[tid]) > 0:
                # If they have been gone for more than 5 seconds, clear history
                if now - history_dict[tid][-1]['time'] > 5.0:
                    del history_dict[tid]
                    
        # 3. Compute spatial zone velocity metrics
        avg_speeds, dir_variances = get_zone_velocity_stats(
            tracked_objects, 
            history_dict, 
            frame.shape, 
            grid_rows=3, 
            grid_cols=3
        )
        
        # 4. Render visualizations on the enhanced frame (or raw frame if you prefer)
        output_frame = draw_overlays(
            frame_to_process, 
            tracked_objects, 
            history_dict, 
            avg_speeds, 
            dir_variances, 
            grid_rows=3, 
            grid_cols=3
        )
        
        # Show window
        cv2.imshow("Crowd Tracking & Velocity Metrics", output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Video window closed.")

if __name__ == "__main__":
    main()
