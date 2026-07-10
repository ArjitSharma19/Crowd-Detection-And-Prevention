# Copyright (c) 2026 MyCompany LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import numpy as np

# Camera Environment Slicing Configurations
# "general": None (uses standard plain YOLO)
# "venue": 640x640 slices with 25% overlap
# "aerial": 384x384 slices with 30% overlap
SLICE_CONFIG = {
    "general": None,
    "venue": {
        "slice_height": 640,
        "slice_width": 640,
        "overlap_ratio": 0.25
    },
    "aerial": {
        "slice_height": 384,
        "slice_width": 384,
        "overlap_ratio": 0.4,
        "imgsz": 512
    }
}

# Global cache for the SAHI AutoDetectionModel instances to avoid reloading weights
_sahi_model_cache = {}

def run_sahi_detection(model_path, frame, slice_height=512, slice_width=512, overlap_ratio=0.2, confidence_threshold=0.25):
    """
    Runs SAHI (Slicing Aided Hyper Inference) on a frame using a YOLO model.
    SAHI splits the image into slices, runs predictions on each slice, and merges them.
    This helps detect small/distant objects (like in aerial/dense crowd scenes).

    NOTE: SAHI is slower due to multiple inference passes per frame. It is intended
    for high-density frames or manual override, not real-time video feeds.
    """
    global _sahi_model_cache

    # Normalize model_path for caching lookup
    abs_model_path = os.path.abspath(model_path)

    if abs_model_path not in _sahi_model_cache:
        import torch
        from sahi import AutoDetectionModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"SAHI: Initializing AutoDetectionModel from {abs_model_path} on {device}...")
        
        _sahi_model_cache[abs_model_path] = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=abs_model_path,
            confidence_threshold=confidence_threshold,
            device=device
        )

    detection_model = _sahi_model_cache[abs_model_path]
    # Dynamically set threshold to match current dashboard slider setting
    detection_model.confidence_threshold = confidence_threshold

    # Run sliced prediction
    from sahi.predict import get_sliced_prediction

    result = get_sliced_prediction(
        frame,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        verbose=0
    )

    detections = []
    # Class ID for person is 0
    person_class_id = 0

    for pred in result.object_prediction_list:
        # Determine category ID in a robust way
        category_id = getattr(pred, 'category_id', None)
        if category_id is None and hasattr(pred, 'category'):
            category_id = getattr(pred.category, 'id', None)

        if category_id == person_class_id:
            # Get bounding box coordinates safely
            x1 = int(pred.bbox.minx)
            y1 = int(pred.bbox.miny)
            x2 = int(pred.bbox.maxx)
            y2 = int(pred.bbox.maxy)

            # Get confidence score safely
            score_val = getattr(pred, 'score', None)
            if score_val is not None:
                if hasattr(score_val, 'value'):
                    score = float(score_val.value)
                else:
                    score = float(score_val)
            else:
                score = 0.0

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': score,
                'class_id': person_class_id
            })

    return detections
