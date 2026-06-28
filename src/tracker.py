import os
from ultralytics import YOLO

def track_frame(model, frame, tracker_state=None, imgsz=960, conf=0.25):
    """
    Runs person tracking on a single frame using Ultralytics ByteTrack.
    
    Args:
        model: YOLO instance from ultralytics
        frame: opencv BGR image matrix
        tracker_state: dict, placeholder for custom tracker states (optional)
        imgsz: int, inference image size (default 960)
        conf: float, detection confidence threshold (default 0.25)
        
    Returns:
        list of dicts: [
            {
                'track_id': int,
                'bbox': [x1, y1, x2, y2],  # integers
                'center_point': (cx, cy)    # tuple of ints
            }, ...
        ]
    """
    if frame is None:
        return []
        
    # Run tracking with ByteTrack configuration
    # Note: ships with ultralytics under config name "bytetrack.yaml"
    # Filter class 0 (person)
    results = model.track(
        source=frame, 
        persist=True, 
        tracker="bytetrack.yaml", 
        imgsz=imgsz, 
        conf=conf, 
        classes=[0], 
        verbose=False
    )
    
    tracked_objects = []
    if len(results) > 0:
        result = results[0]
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy_list = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            for xyxy, track_id in zip(xyxy_list, ids):
                x1, y1, x2, y2 = map(int, xyxy)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                tracked_objects.append({
                    'track_id': int(track_id),
                    'bbox': [x1, y1, x2, y2],
                    'center_point': (cx, cy)
                })
                
    return tracked_objects
