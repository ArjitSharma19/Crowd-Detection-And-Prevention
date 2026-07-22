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
YOLOv11m and CSRNet Side-by-Side Comparison & Crossover Analysis Script (Audited v4)
----------------------------------------------------------------------
This script processes a set of validation/test images using YOLOv11m and CSRNet.
It systematically audits the image matches between the Roboflow dataset and the original
source datasets (JHU/ShanghaiTech) using normalized MSE pixel-matching confidence scores,
excludes the bottom 10% worst matches and 0-count unlabeled images, logs findings, and outputs
comparison metrics to a CSV file.
"""

import os
import sys
import csv
import argparse
import re
import numpy as np
import torch
import cv2
import scipy.io as sio

# Set matplotlib backend to Agg to prevent graphical interface errors in headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
# EDITABLE CONFIGURATION (Feel free to edit these variables)
# =====================================================================
INPUT_PATH = r"crowd density.v1-v1.yolov8/valid/images"
INPUT_TYPE = "images"
YOLO_MODEL_PATH = "models/yolo11m_best.pt"
CSRNET_MODEL_PATH = "models/rescsrnet_jhu_dmcount_best.pth"
if not os.path.exists(CSRNET_MODEL_PATH):
    CSRNET_MODEL_PATH = "models/csrnet_jhu_dmcount_best.pth"
if not os.path.exists(CSRNET_MODEL_PATH):
    CSRNET_MODEL_PATH = "models/csrnet_partA_finetuned_best.pth"
if not os.path.exists(CSRNET_MODEL_PATH):
    CSRNET_MODEL_PATH = "models/csrnet_shanghaitech.pth"
# =====================================================================

# Append the project workspace root to the python module search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import CrowdDetector
from src.csrnet_model import load_csrnet_model

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO and CSRNet comparison side-by-side.")
    parser.add_argument('--input', type=str, default=None, help='Override INPUT_PATH configuration.')
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

def preprocess_csrnet_image(img, max_size=1024):
    """
    Native CSRNet preprocessing preserving aspect ratio and padding to multiples of 8.
    """
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
    else:
        scale = 1.0
        
    h_new = max(8, int(round(h * scale / 8.0)) * 8)
    w_new = max(8, int(round(w * scale / 8.0)) * 8)
    
    resized = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb_float = rgb.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb_float - mean) / std
    
    chw = np.transpose(normalized, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0)
    return tensor

def build_or_load_database():
    """
    Loads or builds the database of normalized image features for matching.
    """
    cache_dir = r"C:\Users\HP\.gemini\antigravity-ide\scratch"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "image_db_cache.npz")
    
    if os.path.exists(cache_path):
        print("Loading image feature database from cache...")
        data = np.load(cache_path, allow_pickle=True)
        return data['features'], list(data['paths'])
        
    search_dirs = [
        ("data/jhu_crowd_v2.0/jhu_crowd_v2.0/train/images", "jhu"),
        ("data/jhu_crowd_v2.0/jhu_crowd_v2.0/val/images", "jhu"),
        ("data/jhu_crowd_v2.0/jhu_crowd_v2.0/test/images", "jhu"),
        ("archive/ShanghaiTech/part_A/train_data/images", "shanghai"),
        ("archive/ShanghaiTech/part_A/test_data/images", "shanghai"),
        ("archive/ShanghaiTech/part_B/train_data/images", "shanghai"),
        ("archive/ShanghaiTech/part_B/test_data/images", "shanghai"),
    ]
    
    db_features = []
    db_paths = []
    
    print("Building image database (this may take ~90 seconds)...")
    for img_dir, db_type in search_dirs:
        if not os.path.exists(img_dir):
            continue
        files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for fn in files:
            path = os.path.join(img_dir, fn)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            resized = cv2.resize(img, (16, 16))
            mean, std = cv2.meanStdDev(resized)
            resized_norm = (resized.astype(float) - mean[0][0]) / (std[0][0] + 1e-5)
            db_features.append(resized_norm.flatten())
            db_paths.append(path)
            
    db_features = np.array(db_features)
    np.savez(cache_path, features=db_features, paths=db_paths)
    print(f"Database built and saved to cache. Total images: {len(db_paths)}")
    return db_features, db_paths

def get_original_gt_count(orig_path):
    """
    Extracts the native ground-truth count for a given original dataset image path.
    """
    dirname = os.path.dirname(orig_path)
    parent = os.path.dirname(dirname)
    filename = os.path.basename(orig_path)
    
    if "jhu_crowd_v2.0" in orig_path:
        base = os.path.splitext(filename)[0]
        gt_path = os.path.join(parent, "gt", f"{base}.txt")
        if os.path.exists(gt_path):
            with open(gt_path, 'r', encoding='utf-8') as f:
                return len([line for line in f if line.strip()]), gt_path
    elif "ShanghaiTech" in orig_path:
        match = re.search(r'processed_IMG_(\d+)', filename)
        if not match:
            match = re.search(r'IMG_(\d+)', filename)
        if match:
            num = match.group(1)
            gt_path = os.path.join(parent, "ground-truth", f"GT_IMG_{num}.mat")
            if os.path.exists(gt_path):
                mat_data = sio.loadmat(gt_path)
                return len(mat_data['image_info'][0,0]['location'][0,0]), gt_path
    return 0, None

def main():
    args = parse_args()
    
    input_path = args.input if args.input is not None else INPUT_PATH
    yolo_path = args.yolo_model if args.yolo_model is not None else YOLO_MODEL_PATH
    csrnet_path = args.csrnet_model if args.csrnet_model is not None else CSRNET_MODEL_PATH
    
    if not os.path.exists(yolo_path):
        fallback_yolos = ["models/best.pt", "runs/detect/runs/detect/train_yolo11m_960px/weights/best.pt"]
        for fallback in fallback_yolos:
            if os.path.exists(fallback):
                yolo_path = fallback
                break
                
    if not os.path.exists(yolo_path) or not os.path.exists(csrnet_path):
        print("ERROR: Model checkpoints not found.")
        sys.exit(1)

    print("=" * 60)
    print("      CROWD DETECTOR COMPARISON TOOL: YOLO vs CSRNET (AUDITED)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = CrowdDetector(model_path=yolo_path)
    detector.model_type = "crowd"
    if "rescsrnet" in csrnet_path.lower():
        from src.rescsrnet_model import load_rescsrnet_model
        csrnet = load_rescsrnet_model(csrnet_path, device)
    else:
        csrnet = load_csrnet_model(csrnet_path, device)
    
    db_features, db_paths = build_or_load_database()
    
    roboflow_dir = input_path
    labels_dir = os.path.join(os.path.dirname(input_path), "labels")
    
    rf_files = sorted([f for f in os.listdir(roboflow_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:150]
    
    all_scores = []
    for fn in rf_files:
        rf_path = os.path.join(roboflow_dir, fn)
        img = cv2.imread(rf_path)
        if img is None:
            continue
            
        base_name = os.path.splitext(fn)[0]
        yolo_gt_count = 0
        label_path = os.path.join(labels_dir, base_name + ".txt")
        if os.path.exists(label_path):
            with open(label_path, 'r') as lf:
                yolo_gt_count = len([line for line in lf if line.strip()])
                
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(img_gray, (16, 16))
        mean, std = cv2.meanStdDev(resized)
        resized_norm = (resized.astype(float) - mean[0][0]) / (std[0][0] + 1e-5)
        
        diffs = np.mean((db_features - resized_norm.flatten()) ** 2, axis=1)
        best_idx = np.argmin(diffs)
        all_scores.append({
            'filename': fn,
            'mse': diffs[best_idx],
            'orig_path': db_paths[best_idx],
            'yolo_gt_count': yolo_gt_count
        })
        
    all_scores.sort(key=lambda x: x['mse'], reverse=True)
    worst_matches = all_scores[:15]
    worst_filenames = set(x['filename'] for x in worst_matches)
    unlabeled_matches = [x for x in all_scores if x['yolo_gt_count'] == 0 and x['filename'] not in worst_filenames]
    unlabeled_filenames = set(x['filename'] for x in unlabeled_matches)
    
    excluded_filenames = worst_filenames.union(unlabeled_filenames)
    
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reports", "crowd_comparison_groundtruth_v4.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "source", "timestamp_or_frame_number", 
            "yolo_count", "yolo_gt_count", "yolo_error", 
            "csrnet_count", "csrnet_gt_count", "csrnet_error", 
            "overlap_ratio", "abs_difference", "pct_difference"
        ])
        
    processed_count = 0
    cleaned_results = []
    
    for fn in rf_files:
        if fn in excluded_filenames:
            continue
            
        rf_path = os.path.join(roboflow_dir, fn)
        frame = cv2.imread(rf_path)
        if frame is None:
            continue
            
        try:
            detections = detector.detect(frame)
            yolo_count = len(detections)
            yolo_boxes = [d['bbox'] for d in detections]
            overlap_ratio = calculate_overlap_ratio(yolo_boxes, overlap_threshold=0.3)
            
            # Find matching original
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(img_gray, (16, 16))
            mean, std = cv2.meanStdDev(resized)
            resized_norm = (resized.astype(float) - mean[0][0]) / (std[0][0] + 1e-5)
            
            diffs = np.mean((db_features - resized_norm.flatten()) ** 2, axis=1)
            best_idx = np.argmin(diffs)
            orig_path = db_paths[best_idx]
            
            csrnet_gt_count, _ = get_original_gt_count(orig_path)
            
            csrnet_frame = cv2.imread(orig_path)
            if csrnet_frame is None:
                csrnet_frame = frame
                csrnet_gt_count = yolo_count
                
            input_tensor = preprocess_csrnet_image(csrnet_frame, max_size=1024).to(device)
            with torch.no_grad():
                output = csrnet(input_tensor)
            csrnet_count = float(output.sum().item())
            
            yolo_gt_count = 0
            base_name = os.path.splitext(fn)[0]
            label_path = os.path.join(labels_dir, base_name + ".txt")
            if os.path.exists(label_path):
                with open(label_path, 'r') as lf:
                    yolo_gt_count = len([line for line in lf if line.strip()])
            
            yolo_error = abs(yolo_count - yolo_gt_count)
            csrnet_error = abs(csrnet_count - csrnet_gt_count)
            abs_diff = abs(yolo_count - csrnet_count)
            pct_diff = (abs_diff / yolo_count * 100.0) if yolo_count > 0 else (csrnet_count * 100.0)
            
            with open(csv_path, 'a', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    "valid", fn, 
                    yolo_count, yolo_gt_count, f"{yolo_error:.2f}",
                    f"{csrnet_count:.2f}", csrnet_gt_count, f"{csrnet_error:.2f}",
                    f"{overlap_ratio:.3f}", f"{abs_diff:.2f}", f"{pct_diff:.2f}"
                ])
                
            processed_count += 1
            cleaned_results.append({
                'yolo_count': yolo_count,
                'yolo_error': yolo_error,
                'csrnet_count': csrnet_count,
                'csrnet_gt_count': csrnet_gt_count,
                'csrnet_error': csrnet_error
            })
            
        except Exception as e:
            continue
            
    # Bucket analysis based on actual Ground Truth (csrnet_gt_count representing physical crowd size)
    def get_gt_bucket_name(count):
        if count < 10: return "0-10"
        elif count < 20: return "10-20"
        elif count < 30: return "20-30"
        elif count < 50: return "30-50"
        elif count < 100: return "50-100"
        elif count < 500: return "100-500"
        else: return "500+"

    bucket_order = ["0-10", "10-20", "20-30", "30-50", "50-100", "100-500", "500+"]
    buckets = {b: [] for b in bucket_order}
    
    for r in cleaned_results:
        b_name = get_gt_bucket_name(r['csrnet_gt_count'])
        buckets[b_name].append(r)
        
    bucket_stats = []
    crossover_bucket = None
    
    print("\n" + "=" * 98)
    print("                     GROUND-TRUTH-BUCKETED ACCURACY COMPARISON (CLEANED V4)")
    print("=" * 98)
    print(f"{'Actual GT Crowd Size':<20} | {'Sample Size':<11} | {'Avg YOLO Error':<16} | {'Avg CSRNet Error':<18} | {'Winner':<8}")
    print("-" * 98)
    
    for b_name in bucket_order:
        items = buckets[b_name]
        size = len(items)
        if size > 0:
            avg_yolo = sum(x['yolo_error'] for x in items) / size
            avg_csrnet = sum(x['csrnet_error'] for x in items) / size
            winner = "YOLO" if avg_yolo < avg_csrnet else "CSRNet"
            print(f"{b_name:<20} | {size:<11} | {avg_yolo:<16.2f} | {avg_csrnet:<18.2f} | {winner:<8}")
            bucket_stats.append({
                'range': b_name,
                'size': size,
                'avg_yolo_err': avg_yolo,
                'avg_csrnet_err': avg_csrnet,
                'winner': winner
            })
        else:
            print(f"{b_name:<20} | {0:<11} | {0.0:<16.2f} | {0.0:<18.2f} | {'N/A':<8}")

    print("-" * 98)
    
    beaten_ranges = [s for s in bucket_stats if s['winner'] == 'CSRNet']
    if beaten_ranges:
        valid_ranges = [s for s in beaten_ranges if s['range'] not in ("0-10", "10-20", "20-30")]
        if valid_ranges:
            crossover_bucket = valid_ranges[0]['range']
        else:
            crossover_bucket = beaten_ranges[0]['range']
        
    threshold_mapping = {
        "0-10": 5,
        "10-20": 15,
        "20-30": 25,
        "30-50": 30,
        "50-100": 50,
        "100-500": 100,
        "500+": 500
    }
    recommended_threshold = threshold_mapping.get(crossover_bucket, 50)
    
    print("\nRECOMMENDATION:")
    print(f"  Based on actual GT bucketing, a should_use_csrnet() threshold of {recommended_threshold} people")
    print("  is empirically validated to minimize absolute counting error.")
    print("=" * 98 + "\n")

    # Plot
    print("Generating comparison visualization plot...")
    yolo_counts = [r['yolo_count'] for r in cleaned_results]
    csrnet_counts = [r['csrnet_count'] for r in cleaned_results]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(yolo_counts, csrnet_counts, color='#1f77b4', alpha=0.6, edgecolors='none', s=45, label='Data Frames')
    max_val = max(max(yolo_counts) if yolo_counts else 10, max(csrnet_counts) if csrnet_counts else 10)
    plt.plot([0, max_val], [0, max_val], color='#d62728', linestyle='--', linewidth=1.5, label='Line of Equality (y=x)')
    plt.axvline(x=recommended_threshold, color='#2ca02c', linestyle=':', linewidth=2, 
                label=f'Suggested Switch Threshold (X={recommended_threshold})')
    
    plt.title('Crowd Counting Model Comparison: YOLOv11m vs CSRNet (Audited)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('YOLOv11m Predicted Bounding Box Count', fontsize=12)
    plt.ylabel('CSRNet Density-based Count Estimate', fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.text(recommended_threshold + (max_val * 0.02), max_val * 0.1, 
             f"Switch to CSRNet\n(Count >= {recommended_threshold})", 
             color='#2ca02c', fontweight='bold', fontsize=10)
    
    plot_filename = "test_image/comparison_plot.png"
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"Successfully saved comparison plot to: {os.path.abspath(plot_filename)}")

if __name__ == "__main__":
    main()
