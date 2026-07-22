"""
CSRNet Batch Folder Inference and Density Mapping Test Runner.
This script scans a specified directory containing test images, performs
density estimation on each image, generates heatmaps, and prints out a
summary table of estimated counts for comparison.
"""

import os
import cv2
import torch
import numpy as np
import sys

# Append the project workspace root to the python module search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom CSRNet modules
from src.csrnet_model import load_csrnet_model, CSRNet
from src.csrnet_inference import estimate_density, get_zone_densities, density_map_to_heatmap

# =====================================================================
# CONFIGURATION
# =====================================================================
# Local folder containing test images
IMAGE_FOLDER = "test_image"
WEIGHTS_PATH = "models/csrnet_jhu_dmcount_best.pth"
if not os.path.exists(WEIGHTS_PATH):
    WEIGHTS_PATH = "models/csrnet_partA_finetuned_best.pth"
if not os.path.exists(WEIGHTS_PATH):
    WEIGHTS_PATH = "models/csrnet_shanghaitech.pth"
# =====================================================================

def main():
    # 1. Setup Device (RTX 4060 GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 2. Load CSRNet Model
    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading custom weights from '{WEIGHTS_PATH}'...")
        try:
            model = load_csrnet_model(WEIGHTS_PATH, device)
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}. Falling back to default initialization.")
            model = CSRNet(load_weights=False).to(device)
            model.eval()
    else:
        print(f"Pretrained weights not found at '{WEIGHTS_PATH}'. Initializing default model.")
        model = CSRNet(load_weights=False).to(device)
        model.eval()

    # To store summary info for the final table
    summary_results = []

    # 3. Process each image in the directory
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    if not os.path.exists(IMAGE_FOLDER):
        print(f"ERROR: Image folder not found at '{IMAGE_FOLDER}'")
        return
        
    image_files = [os.path.join(IMAGE_FOLDER, f) for f in sorted(os.listdir(IMAGE_FOLDER)) 
                   if f.lower().endswith(valid_extensions) and not f.startswith("heatmap_") and f != "comparison_plot.png"]
    
    if not image_files:
        print(f"No valid images found in '{IMAGE_FOLDER}'")
        return

    for index, filepath in enumerate(image_files, start=1):
        filename = os.path.basename(filepath)
        output_filename = os.path.join(IMAGE_FOLDER, f"heatmap_{filename}")
        
        print("\n" + "="*70)
        print(f"--- Image {index}: {filename} ---")
        print("="*70)
        
        # Load with OpenCV
        frame = cv2.imread(filepath)
        if frame is None:
            print(f"ERROR: OpenCV failed to read '{filepath}'")
            summary_results.append((filename, "FAILED (Read error)"))
            continue
            
        print(f"Image Resolution: {frame.shape[1]}x{frame.shape[0]}")
        
        # Run inference
        print("Running CSRNet inference...")
        density_map, total_count = estimate_density(model, frame, device)
        
        # Split density map into a 3x3 grid
        grid_rows, grid_cols = 3, 3
        zone_densities = get_zone_densities(density_map, grid_rows=grid_rows, grid_cols=grid_cols)
        
        # Print image results
        print(f"\nEstimated Count: {total_count:.2f}")
        print("Zone grid (3x3):")
        for r in range(grid_rows):
            row_values = [f"{zone_densities[r, c]:.2f}" for c in range(grid_cols)]
            print(f"  [ " + " | ".join(row_values) + " ]")
            
        # Generate and save heatmap overlay
        print("Generating heatmap overlay...")
        heatmap_frame = density_map_to_heatmap(density_map, frame, alpha=0.5)
        cv2.imwrite(output_filename, heatmap_frame)
        print(f"Heatmap overlay saved as: {output_filename}")
        
        # Add to summary results list
        summary_results.append((filename, f"{total_count:.2f}"))

    # 4. Print Summary Table
    print("\n" + "="*60)
    print("                 SUMMARY TABLE")
    print("="*60)
    print(f"{'Filename':<30} | {'Estimated Count / Status':<25}")
    print("-"*60)
    for filename, count_status in summary_results:
        print(f"{filename:<30} | {count_status:<25}")
    print("="*60)

if __name__ == "__main__":
    main()
