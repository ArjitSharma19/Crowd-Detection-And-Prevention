import os
import cv2
import numpy as np
from ultralytics import YOLO

class CrowdDetector:
    def __init__(self, model_path=None, imgsz=960, confidence_threshold=0.25):
        """
        Initializes the YOLO object detector.
        Supports switching between a General Detector (trained on COCO) and a Crowd Detector (custom fine-tuned).
        """
        self.imgsz = imgsz
        self.confidence_threshold = confidence_threshold
        self.model_type = "general"  # "general" or "crowd"

        # Load general detector (standard pretrained model on disk)
        print("Loading general COCO detector 'yolo11m.pt'...")
        self.model_general = YOLO("yolo11m.pt")

        # Load custom fine-tuned crowd model
        local_custom_path = os.path.join("models", "best.pt")
        if model_path and os.path.exists(model_path):
            print(f"Loading custom crowd weights from {model_path}")
            self.model_crowd = YOLO(model_path)
        elif os.path.exists(local_custom_path):
            print(f"Loading custom crowd weights from {local_custom_path}")
            self.model_crowd = YOLO(local_custom_path)
        else:
            print("Custom crowd weights not found. Defaulting Crowd Mode to standard model.")
            self.model_crowd = self.model_general

        # Class index for 'person' is 0 in both COCO and the custom dataset
        self.person_class_id = 0

    def detect(self, frame, confidence_threshold=None):
        """
        Runs person detection on a single frame.
        
        Args:
            frame: opencv BGR image matrix
            confidence_threshold: float, confidence limit to consider a detection valid (defaults to self.confidence_threshold)
            
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
            
        conf_val = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        
        # Select active model
        active_model = self.model_general if self.model_type == "general" else self.model_crowd
        
        # Run inference with configuration parameters.
        # iou=0.45 enforces Non-Maximum Suppression to prevent double overlapping boxes on the same person.
        results = active_model(frame, imgsz=self.imgsz, conf=conf_val, iou=0.45, verbose=False)
        
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Filter for person detections only
                if cls_id == self.person_class_id and conf >= conf_val:
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
