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

"""
YOLOv11m and CSRNet Side-by-Side Comparison & Crossover Analysis Script
----------------------------------------------------------------------
This script processes a set of validation/test images or video frames using
both YOLOv11m and CSRNet. It logs their respective counts, box overlap ratios,
and differences to a CSV file. After processing, it analyzes the data to suggest 
an empirical count-based threshold for switching between the two models in the hybrid pipeline.
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch
import cv2

# Set matplotlib backend to Agg to prevent graphical interface errors in headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
# EDITABLE CONFIGURATION (Feel free to edit these variables)
# =====================================================================
INPUT_PATH = r"crowd density.v1-v1.yolov8/valid/images"
INPUT_TYPE = "images"              # "images" or "video"
FRAME_SAMPLE_INTERVAL = 10         # Process every Nth frame in video mode
MAX_IMAGES_TO_PROCESS = 150        # Limit image count to keep analysis fast (None for all)
YOLO_MODEL_PATH = "models/yolo11m_best.pt"
CSRNET_MODEL_PATH = "models/csrnet_shanghaitech.pth"
# =====================================================================

# Append the project workspace root to the python module search path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.detector import CrowdDetector
from src.csrnet_model import load_csrnet_model
from src.csrnet_inference import estimate_density

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO and CSRNet comparison side-by-side.")
    parser.add_argument('--input', type=str, default=None, help='Override INPUT_PATH configuration.')
    parser.add_argument('--type', type=str, default=None, choices=['images', 'video'], help='Override INPUT_TYPE configuration.')
    parser.add_argument('--interval', type=int, default=None, help='Override FRAME_SAMPLE_INTERVAL.')
    parser.add_argument('--max-images', type=int, default=None, help='Override MAX_IMAGES_TO_PROCESS.')
    parser.add_argument('--yolo-model', type=str, default=None, help='Override YOLO_MODEL_PATH.')
    parser.add_argument('--csrnet-model', type=str, default=None, help='Override CSRNET_MODEL_PATH.')
    return parser.parse_args()

def calculate_overlap_ratio(yolo_boxes, overlap_threshold=0.3):
    """
    Computes the ratio of bounding boxes that overlap with at least one other box
    above the IoU threshold. This matches the logic used in should_use_csrnet().
    """
    n_boxes = len(yolo_boxes)
    if n_boxes <= 1:
        return 0.0

    overlapping_indices = set()
    for i in range(n_boxes):
        for j in range(i + 1, n_boxes):
            box1 = yolo_boxes[i]
            box2 = yolo_boxes[j]
            
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            # Intersection coordinates
            xi1 = max(x1_1, x1_2)
            yi1 = max(y1_1, y1_2)
            xi2 = min(x2_1, x2_2)
            yi2 = min(y2_1, y2_2)
            
            inter_w = max(0, xi2 - xi1)
            inter_h = max(0, yi2 - yi1)
            inter_area = inter_w * inter_h
            
            # Union area
            box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
            box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = box1_area + box2_area - inter_area
            
            if union_area > 0:
                iou = inter_area / union_area
                if iou > overlap_threshold:
                    overlapping_indices.add(i)
                    overlapping_indices.add(j)
                    
    return len(overlapping_indices) / n_boxes

def get_bucket_name(count):
    """
    Groups YOLO counts into analysis buckets.
    """
    if count < 10:
        return "0-10"
    elif count < 20:
        return "10-20"
    elif count < 30:
        return "20-30"
    elif count < 50:
        return "30-50"
    elif count < 100:
        return "50-100"
    else:
        return "100+"

def main():
    args = parse_args()
    
    # Resolve overrides
    input_path = args.input if args.input is not None else INPUT_PATH
    input_type = args.type if args.type is not None else INPUT_TYPE
    frame_interval = args.interval if args.interval is not None else FRAME_SAMPLE_INTERVAL
    max_images = args.max_images if args.max_images is not None else MAX_IMAGES_TO_PROCESS
    yolo_path = args.yolo_model if args.yolo_model is not None else YOLO_MODEL_PATH
    csrnet_path = args.csrnet_model if args.csrnet_model is not None else CSRNET_MODEL_PATH
    
    # 1. Validate paths and model checkpoints
    if not os.path.exists(yolo_path):
        # Fallback search if yolo_path doesn't exist
        fallback_yolos = ["models/best.pt", "runs/detect/runs/detect/train_yolo11m_960px/weights/best.pt"]
        for fallback in fallback_yolos:
            if os.path.exists(fallback):
                yolo_path = fallback
                break
                
    if not os.path.exists(yolo_path):
        print(f"ERROR: YOLO model checkpoint not found at {yolo_path}")
        sys.exit(1)
        
    if not os.path.exists(csrnet_path):
        # Fallback search for CSRNet
        fallback_csrnet = "models/csrnet_partA_finetuned_best.pth"
        if os.path.exists(fallback_csrnet):
            csrnet_path = fallback_csrnet
            
    if not os.path.exists(csrnet_path):
        print(f"ERROR: CSRNet model checkpoint not found at {csrnet_path}")
        sys.exit(1)
        
    if not os.path.exists(input_path):
        print(f"ERROR: Input path does not exist: {input_path}")
        sys.exit(1)

    print("=" * 60)
    print("      CROWD DETECTOR COMPARISON TOOL: YOLO vs CSRNET")
    print("=" * 60)
    print(f"Input Path:        {os.path.abspath(input_path)}")
    print(f"Input Type:        {input_type}")
    print(f"YOLO Model:        {os.path.abspath(yolo_path)}")
    print(f"CSRNet Model:      {os.path.abspath(csrnet_path)}")
    print("=" * 60 + "\n")

    # 2. Initialize Models
    print("Initializing models on target device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")
    
    # Initialize YOLO crowd detector wrapper
    detector = CrowdDetector(model_path=yolo_path)
    detector.model_type = "crowd"  # Force fine-tuned weights
    
    # Load CSRNet model
    csrnet = load_csrnet_model(csrnet_path, device)
    
    # 3. Collect and verify input sources
    image_files = []
    video_cap = None
    total_items = 0
    source_name = os.path.basename(input_path)

    if input_type == "images":
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        if os.path.isdir(input_path):
            image_files = [
                os.path.join(input_path, f) 
                for f in os.listdir(input_path) 
                if f.lower().endswith(valid_extensions)
            ]
            image_files.sort()
        else:
            if input_path.lower().endswith(valid_extensions):
                image_files = [input_path]
                
        if max_images and len(image_files) > max_images:
            print(f"Limiting evaluation to first {max_images} images out of {len(image_files)} found.")
            image_files = image_files[:max_images]
            
        total_items = len(image_files)
        if total_items == 0:
            print(f"ERROR: No valid images found in path: {input_path}")
            sys.exit(1)
        print(f"Loaded {total_items} images for comparison.")
        
    elif input_type == "video":
        video_cap = cv2.VideoCapture(input_path)
        if not video_cap.isOpened():
            print(f"ERROR: Could not open video file: {input_path}")
            sys.exit(1)
        
        # Calculate total frames that will be processed
        raw_total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Total items is the number of sampled frames
        total_items = (raw_total_frames + frame_interval - 1) // frame_interval
        print(f"Opened video file. Raw frame count: {raw_total_frames}. Sampling every {frame_interval}th frame (~{total_items} samples).")

    # 4. Prepare logging CSV
    csv_path = "crowd_comparison.csv"
    print(f"Initializing log file: {csv_path}")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["source", "timestamp_or_frame_number", "yolo_count", "csrnet_count", "overlap_ratio", "abs_difference", "pct_difference"])

    # 5. Process Loop
    processed_count = 0
    
    if input_type == "images":
        for idx, img_path in enumerate(image_files, start=1):
            filename = os.path.basename(img_path)
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Error reading image: {img_path}. Skipping.")
                continue
                
            try:
                # a. Run YOLO Detection
                detections = detector.detect(frame)
                yolo_count = len(detections)
                yolo_boxes = [d['bbox'] for d in detections]
                
                # b. Calculate Overlap Ratio
                overlap_ratio = calculate_overlap_ratio(yolo_boxes, overlap_threshold=0.3)
                
                # c. Run CSRNet Estimation
                density_map, csrnet_count = estimate_density(csrnet, frame, device)
                
                # d. Calculate metrics
                abs_diff = abs(yolo_count - csrnet_count)
                if yolo_count == 0:
                    pct_diff = 0.0 if csrnet_count == 0 else 100.0 * csrnet_count
                else:
                    pct_diff = (abs_diff / yolo_count) * 100.0
                    
                # Write to CSV immediately
                with open(csv_path, 'a', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([
                        source_name, 
                        filename, 
                        yolo_count, 
                        f"{csrnet_count:.2f}", 
                        f"{overlap_ratio:.3f}", 
                        f"{abs_diff:.2f}", 
                        f"{pct_diff:.2f}"
                    ])
                    
                processed_count += 1
                print(f"Processed image {processed_count}/{total_items}: YOLO={yolo_count}, CSRNet={csrnet_count:.1f}, diff={abs_diff:.1f}")
                
            except Exception as e:
                print(f"ERROR: Failed processing image {filename}: {e}. Skipping.")
                continue
                
    elif input_type == "video":
        frame_idx = 0
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                frame_label = f"frame_{frame_idx}"
                try:
                    # a. Run YOLO Detection
                    detections = detector.detect(frame)
                    yolo_count = len(detections)
                    yolo_boxes = [d['bbox'] for d in detections]
                    
                    # b. Calculate Overlap Ratio
                    overlap_ratio = calculate_overlap_ratio(yolo_boxes, overlap_threshold=0.3)
                    
                    # c. Run CSRNet Estimation
                    density_map, csrnet_count = estimate_density(csrnet, frame, device)
                    
                    # d. Calculate metrics
                    abs_diff = abs(yolo_count - csrnet_count)
                    if yolo_count == 0:
                        pct_diff = 0.0 if csrnet_count == 0 else 100.0 * csrnet_count
                    else:
                        pct_diff = (abs_diff / yolo_count) * 100.0
                        
                    # Write to CSV immediately
                    with open(csv_path, 'a', newline='', encoding='utf-8') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow([
                            source_name, 
                            frame_label, 
                            yolo_count, 
                            f"{csrnet_count:.2f}", 
                            f"{overlap_ratio:.3f}", 
                            f"{abs_diff:.2f}", 
                            f"{pct_diff:.2f}"
                        ])
                        
                    processed_count += 1
                    print(f"Processed frame {processed_count}/{total_items} (Frame Index {frame_idx}): YOLO={yolo_count}, CSRNet={csrnet_count:.1f}, diff={abs_diff:.1f}")
                    
                except Exception as e:
                    print(f"ERROR: Failed processing frame index {frame_idx}: {e}. Skipping.")
                    
            frame_idx += 1
            
        video_cap.release()

    print(f"\nProcessing finished. Saved results to '{csv_path}'.")

    # 6. Load CSV and analyze
    records = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    'yolo_count': int(row['yolo_count']),
                    'csrnet_count': float(row['csrnet_count']),
                    'overlap_ratio': float(row['overlap_ratio']),
                    'abs_difference': float(row['abs_difference']),
                    'pct_difference': float(row['pct_difference'])
                })

    if not records:
        print("ERROR: No data records collected. Analysis aborted.")
        sys.exit(1)

    # Bucket analysis
    bucket_order = ["0-10", "10-20", "20-30", "30-50", "50-100", "100+"]
    buckets = {b: [] for b in bucket_order}
    
    for r in records:
        b_name = get_bucket_name(r['yolo_count'])
        buckets[b_name].append(r)
        
    bucket_stats = []
    crossover_bucket = None
    
    print("\n" + "=" * 70)
    print("                     CROSSOVER POINT ANALYSIS")
    print("=" * 70)
    print(f"{'YOLO Count Range':<18} | {'Sample Size':<11} | {'Avg Abs Diff':<13} | {'Avg Pct Diff':<12}")
    print("-" * 70)
    
    for b_name in bucket_order:
        items = buckets[b_name]
        size = len(items)
        if size > 0:
            avg_abs = sum(x['abs_difference'] for x in items) / size
            avg_pct = sum(x['pct_difference'] for x in items) / size
        else:
            avg_abs = 0.0
            avg_pct = 0.0
            
        print(f"{b_name:<18} | {size:<11} | {avg_abs:<13.2f} | {avg_pct:<12.1f}%")
        
        bucket_stats.append({
            'range': b_name,
            'size': size,
            'avg_abs': avg_abs,
            'avg_pct': avg_pct
        })

    # Find where the disagreement starts climbing sharply
    # We define this as the first bucket where the percentage difference exceeds 25% (and sample size > 0)
    for stat in bucket_stats:
        if stat['size'] > 0 and stat['avg_pct'] >= 25.0:
            crossover_bucket = stat['range']
            break
            
    # Fallback to the one with the maximum difference if none crossed 25%
    if not crossover_bucket:
        valid_stats = [s for s in bucket_stats if s['size'] > 0]
        if valid_stats:
            max_stat = max(valid_stats, key=lambda s: s['avg_pct'])
            if max_stat['avg_pct'] > 15.0:
                crossover_bucket = max_stat['range']

    # Map crossover bucket to suggested X threshold
    threshold_mapping = {
        "0-10": 5,
        "10-20": 15,
        "20-30": 25,
        "30-50": 30,
        "50-100": 50,
        "100+": 100
    }
    recommended_threshold = threshold_mapping.get(crossover_bucket, 15)

    print("-" * 70)
    if crossover_bucket:
        print(f"Empirical Crossover Point Identified: YOLO Range '{crossover_bucket}'")
        print(f"  At this range, the average percentage difference between the individual box")
        print(f"  detection model (YOLO) and density estimator (CSRNet) rises to {buckets[crossover_bucket][0]['pct_difference'] if len(buckets[crossover_bucket])==1 else bucket_stats[bucket_order.index(crossover_bucket)]['avg_pct']:.1f}%.")
        print(f"  This divergence indicates that YOLO is beginning to severely undercount due to ")
        print(f"  occlusions or dense crowd clusters.")
    else:
        print("Empirical Crossover Point: NOT CLEARLY DETECTED (low divergence overall).")
        print(f"  Defaulting threshold to 15 based on typical occlusion curves.")

    print("\nRECOMMENDATION:")
    print(f"  Based on this data, a should_use_csrnet() threshold of approximately {recommended_threshold} people")
    print("  is supported by the divergence pattern observed.")
    print("=" * 70 + "\n")

    # 7. Generate and save plot
    print("Generating comparison visualization plot...")
    yolo_counts = [r['yolo_count'] for r in records]
    csrnet_counts = [r['csrnet_count'] for r in records]
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of data points
    plt.scatter(yolo_counts, csrnet_counts, color='#1f77b4', alpha=0.6, edgecolors='none', s=45, label='Data Frames')
    
    # Draw reference Line of Equality
    max_val = max(max(yolo_counts) if yolo_counts else 10, max(csrnet_counts) if csrnet_counts else 10)
    plt.plot([0, max_val], [0, max_val], color='#d62728', linestyle='--', linewidth=1.5, label='Line of Equality (y=x)')
    
    # Draw Suggested Threshold Line
    plt.axvline(x=recommended_threshold, color='#2ca02c', linestyle=':', linewidth=2, 
                label=f'Suggested Switch Threshold (X={recommended_threshold})')
    
    plt.title('Crowd Counting Model Comparison: YOLOv11m vs CSRNet', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('YOLOv11m Predicted Bounding Box Count', fontsize=12)
    plt.ylabel('CSRNet Density-based Count Estimate', fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Add annotation text on the plot
    plt.text(recommended_threshold + (max_val * 0.02), max_val * 0.1, 
             f"Switch to CSRNet\n(Count >= {recommended_threshold})", 
             color='#2ca02c', fontweight='bold', fontsize=10)
    
    plot_filename = "comparison_plot.png"
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"Successfully saved comparison plot to: {os.path.abspath(plot_filename)}")

if __name__ == "__main__":
    main()
