import os
import cv2
import numpy as np
from ultralytics import YOLO

class CrowdDetector:
    def __init__(self, model_path=None):
        """
        Initializes the YOLOv8 object detector.
        If a custom model path is provided (e.g. 'models/best.pt') and exists, it will load it.
        Otherwise, it defaults to the pre-trained 'yolov8n.pt' (nano) model.
        """
        # Look for custom best.pt weights in the models directory first
        local_custom_path = os.path.join("models", "best.pt")
        
        if model_path and os.path.exists(model_path):
            self.model_path = model_path
        elif os.path.exists(local_custom_path):
            self.model_path = local_custom_path
            print(f"Loading custom fine-tuned weights from {local_custom_path}")
        else:
            self.model_path = "yolov8n.pt"  # Will automatically download via ultralytics if not found
            print("No custom weights found. Using default YOLOv8 Nano model (pretrained on COCO).")
        
        # Load the YOLO model
        self.model = YOLO(self.model_path)
        # Class index for 'person' in standard COCO dataset is 0
        self.person_class_id = 0

    def detect(self, frame, confidence_threshold=0.3):
        """
        Runs person detection on a single frame.
        
        Args:
            frame: opencv BGR image matrix
            confidence_threshold: float, confidence limit to consider a detection valid
            
        Returns:
            list of dicts: [
                {
                    'bbox': [x1, y1, x2, y2],  # integers
                    'confidence': float,
                    'class_id': int
                }, ...
            ]
        """
        if frame is None:
            return []
            
        # Run inference. verbose=False disables spamming console every frame
        results = self.model(frame, verbose=False)
        
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Filter for person detections only
                if cls_id == self.person_class_id and conf >= confidence_threshold:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls_id
                    })
                    
        return detections

    def draw_detections(self, frame, detections):
        """
        Helper to draw bounding boxes and count on the frame.
        """
        annotated_frame = frame.copy()
        count = len(detections)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box (Sleek light cyan / blue borders)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (235, 206, 135), 2)
            
            # Subtle tag
            label = f"P: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - h - 4), (x1 + w, y1), (235, 206, 135), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            
        # Draw counter banner
        cv2.rectangle(annotated_frame, (10, 10), (180, 45), (30, 30, 30), -1)
        cv2.putText(annotated_frame, f"Count: {count}", (20, 33), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
        return annotated_frame
