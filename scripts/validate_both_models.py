import os
import sys
import cv2
import torch
import numpy as np

# Append project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import CrowdDetector
from src.csrnet_model import load_csrnet_model
from src.csrnet_inference import estimate_density, density_map_to_heatmap
from src.density import should_use_csrnet

def get_gt_count_jhu(gt_path):
    if not os.path.exists(gt_path):
        return 0
    with open(gt_path, 'r', encoding='utf-8') as f:
        return len([line for line in f if line.strip()])

def main():
    print("=" * 90)
    print("      DUAL-MODEL VALIDATION SUITE: YOLOv11m vs FINE-TUNED CSRNet (DM-Count)")
    print("=" * 90)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load YOLOv11m Detector
    yolo_weights = "models/yolo11m_best.pt"
    if not os.path.exists(yolo_weights):
        yolo_weights = "models/best.pt"
    if not os.path.exists(yolo_weights):
        yolo_weights = "yolo11m.pt"
    print(f"Loading YOLO model from: {yolo_weights}")
    yolo_detector = CrowdDetector(model_path=yolo_weights, imgsz=960, confidence_threshold=0.25)
    
    # 2. Load Fine-Tuned CSRNet Model
    csrnet_weights = "models/csrnet_jhu_dmcount_best.pth"
    print(f"Loading CSRNet model from: {csrnet_weights}")
    csrnet_model = load_csrnet_model(csrnet_weights, device)
    
    # 3. Resolve JHU Test Set Paths
    test_img_dir = "data/jhu_crowd_v2.0/jhu_crowd_v2.0/test/images"
    test_gt_dir = "data/jhu_crowd_v2.0/jhu_crowd_v2.0/test/gt"
    
    if not os.path.exists(test_img_dir):
        print(f"ERROR: Test images directory not found at {test_img_dir}")
        return
        
    img_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png'))])
    print(f"Total test images available: {len(img_files)}")
    
    # Sample 20 diverse images across indices
    sample_indices = np.linspace(0, len(img_files) - 1, 20, dtype=int)
    sampled_files = [img_files[i] for i in sample_indices]
    
    output_vis_dir = "data/reports/validation_samples"
    os.makedirs(output_vis_dir, exist_ok=True)
    
    results = []
    
    print("\nEvaluating 20 sampled dataset images...\n")
    print(f"{'Image':<12} | {'GT Count':<9} | {'YOLO Est':<9} | {'YOLO Err':<9} | {'CSRNet Est':<11} | {'CSRNet Err':<11} | {'Auto Choice':<11} | {'Winner':<8}")
    print("-" * 105)
    
    for fn in sampled_files:
        img_path = os.path.join(test_img_dir, fn)
        gt_filename = os.path.splitext(fn)[0] + ".txt"
        gt_path = os.path.join(test_gt_dir, gt_filename)
        
        gt_count = get_gt_count_jhu(gt_path)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        # A. YOLO Inference
        yolo_dets = yolo_detector.detect(frame)
        yolo_count = len(yolo_dets)
        yolo_boxes = [d['bbox'] for d in yolo_dets]
        yolo_err = abs(yolo_count - gt_count)
        
        # B. CSRNet Tiled Inference
        density_map, csrnet_count = estimate_density(csrnet_model, frame, device, use_tiled=True)
        csrnet_err = abs(csrnet_count - gt_count)
        
        # C. Auto Mode Decision
        use_csrnet = should_use_csrnet(yolo_count, yolo_boxes, threshold=50, overlap_threshold=0.3)
        auto_choice = "CSRNet" if use_csrnet else "YOLO"
        
        # D. Winner Determination
        winner = "YOLO" if yolo_err < csrnet_err else "CSRNet"
        
        print(f"{fn:<12} | {gt_count:<9} | {yolo_count:<9} | {yolo_err:<9.1f} | {csrnet_count:<11.1f} | {csrnet_err:<11.1f} | {auto_choice:<11} | {winner:<8}")
        
        results.append({
            'file': fn,
            'gt': gt_count,
            'yolo_count': yolo_count,
            'yolo_err': yolo_err,
            'csrnet_count': csrnet_count,
            'csrnet_err': csrnet_err,
            'auto_choice': auto_choice,
            'winner': winner
        })
        
        # Save Visual Overlay Sample
        yolo_vis = yolo_detector.draw_detections(frame, yolo_dets)
        heatmap_vis = density_map_to_heatmap(density_map, frame, alpha=0.5)
        
        # Side by Side
        h, w = frame.shape[:2]
        target_w = 400
        target_h = int(h * (target_w / w))
        
        yolo_resized = cv2.resize(yolo_vis, (target_w, target_h))
        heatmap_resized = cv2.resize(heatmap_vis, (target_w, target_h))
        
        # Annotate images
        cv2.putText(yolo_resized, f"YOLO: {yolo_count} (GT: {gt_count})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(heatmap_resized, f"CSRNet: {csrnet_count:.1f} (GT: {gt_count})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        combined = np.hstack((yolo_resized, heatmap_resized))
        cv2.imwrite(os.path.join(output_vis_dir, f"val_{fn}"), combined)
        
    print("-" * 105)
    
    # Calculate Overall MAE
    avg_yolo_mae = np.mean([r['yolo_err'] for r in results])
    avg_csrnet_mae = np.mean([r['csrnet_err'] for r in results])
    
    print(f"\nSummary of {len(results)} Validation Test Samples:")
    print(f"  - Average YOLO MAE:   {avg_yolo_mae:.2f}")
    print(f"  - Average CSRNet MAE: {avg_csrnet_mae:.2f}")
    print(f"  - Side-by-side visual overlays saved to: {os.path.abspath(output_vis_dir)}")
    print("=" * 90)

if __name__ == "__main__":
    main()
