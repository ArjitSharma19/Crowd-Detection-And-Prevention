"""
CSRNet Single Image Inference and Density Mapping Test Runner.
This script demonstrates how to load CSRNet model weights, execute feed-forward
inference on a single image, divide the resulting density map into a 3x3 grid,
and output a visual heatmap representation.
"""

import os
import cv2
import torch
import numpy as np
import sys

# Append the project workspace root to the python module search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from src.csrnet_model import CSRNet, load_csrnet_model
from src.csrnet_inference import estimate_density, get_zone_densities, density_map_to_heatmap

# =====================================================================
# EDITABLE VARIABLES
# =====================================================================
TEST_IMAGE_PATH = "test_image/test_image.jpg"                  # Path to the test image
WEIGHTS_PATH = "models/csrnet_partA_finetuned_best.pth"
if not os.path.exists(WEIGHTS_PATH):
    WEIGHTS_PATH = "models/csrnet_shanghaitech.pth"     # Path to CSRNet pretrained weights
OUTPUT_HEATMAP_PATH = "test_image/test_heatmap.jpg"            # Output heatmap image file
# =====================================================================

def create_synthetic_image(path):
    """
    Creates a synthetic crowd test image to allow verification if no real image is provided.
    Draws shapes resembling heads on a dark background.
    """
    print(f"Test Image not found at '{path}'. Generating a synthetic crowd image...")
    # Create dark gray canvas
    img = np.ones((480, 640, 3), dtype=np.uint8) * 30
    
    # Draw floor patterns
    for x in range(0, 640, 80):
        cv2.line(img, (x, 0), (x, 480), (45, 45, 45), 1)
    for y in range(0, 480, 80):
        cv2.line(img, (0, y), (640, y), (45, 45, 45), 1)
        
    # Draw simulated human bodies (capsules/circles)
    # Cluster 1 (Left - dense)
    centers = [(150, 200), (160, 210), (140, 230), (170, 240), (130, 210), (150, 250)]
    # Cluster 2 (Right - sparse)
    centers += [(450, 150), (480, 180), (510, 140)]
    
    for (cx, cy) in centers:
        # Draw "shoulders"
        cv2.ellipse(img, (cx, cy + 20), (25, 12), 0, 0, 360, (70, 70, 200), -1)
        # Draw "head"
        cv2.circle(img, (cx, cy), 12, (200, 200, 220), -1)
        
    cv2.imwrite(path, img)
    print(f"Synthetic image saved to: {path}")


def main():
    # 1. Choose Device (RTX 4060 / CUDA GPU if available, else CPU)
    # CUDA is checked automatically.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for testing: {device}")
    if device.type == "cuda":
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
        
    # 2. Check if the test image exists; if not, generate a synthetic one
    if not os.path.exists(TEST_IMAGE_PATH):
        create_synthetic_image(TEST_IMAGE_PATH)
        
    # 3. Load CSRNet Model
    # Since the weights file might not exist yet, we handle both cases gracefully
    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading custom weights from {WEIGHTS_PATH}...")
        try:
            model = load_csrnet_model(WEIGHTS_PATH, device)
            print("Successfully loaded model from checkpoint file.")
        except Exception as e:
            print(f"Error loading model from checkpoint: {e}")
            print("Falling back to standard initialization for testing...")
            model = CSRNet(load_weights=False).to(device)
            model.eval()
    else:
        print(f"Pretrained weights not found at '{WEIGHTS_PATH}'.")
        print("Falling back to VGG-16 initialized model (useful for verification of inference pipeline mechanics).")
        model = CSRNet(load_weights=False).to(device)
        model.eval()
        
    # 4. Load the input frame
    frame = cv2.imread(TEST_IMAGE_PATH)
    if frame is None:
        print(f"CRITICAL ERROR: Could not read image at '{TEST_IMAGE_PATH}'")
        return
        
    print(f"Loaded image '{TEST_IMAGE_PATH}' of resolution: {frame.shape[1]}x{frame.shape[0]}")
    
    # 5. Run inference to calculate density map
    print("Running CSRNet inference...")
    density_map, total_count = estimate_density(model, frame, device)
    
    # 6. Split density map into a 3x3 grid to inspect per-zone values
    grid_rows, grid_cols = 3, 3
    zone_densities = get_zone_densities(density_map, grid_rows=grid_rows, grid_cols=grid_cols)
    
    # 7. Print results
    print("\n" + "="*50)
    print("                CSRNET TEST RESULTS")
    print("="*50)
    print(f"Estimated Total Count: {total_count:.2f}")
    print(f"Density map output shape: {density_map.shape}")
    print("\nZone counts (3x3 grid matching top-left to bottom-right):")
    for r in range(grid_rows):
        row_str = " | ".join([f"{zone_densities[r, c]:.2f}" for c in range(grid_cols)])
        print(f"  [{row_str}]")
    print("="*50)
    
    # 8. Generate and save heatmap overlay
    print("Generating heatmap overlay image...")
    heatmap_frame = density_map_to_heatmap(density_map, frame, alpha=0.5)
    
    # Save the output file
    cv2.imwrite(OUTPUT_HEATMAP_PATH, heatmap_frame)
    print(f"Success! Heatmap overlay saved to: {OUTPUT_HEATMAP_PATH}")


if __name__ == "__main__":
    main()
