import os
import re
import cv2
import argparse
import torch
import numpy as np
import scipy.io as sio

# Import custom CSRNet model loader
from src.csrnet_model import load_csrnet_model, CSRNet

# Try to use requests library, fallback to built-in urllib if not installed
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    HAS_REQUESTS = False

# Hardcoded defaults for backward compatibility
DEFAULT_SHANGHAI_A_IMAGES_DIR = r"E:\Crowd Detection &  Prevention\archive\ShanghaiTech\part_A\test_data\images"
DEFAULT_SHANGHAI_A_GT_DIR = r"E:\Crowd Detection &  Prevention\archive\ShanghaiTech\part_A\test_data\ground-truth"
DEFAULT_SHANGHAI_A_WEIGHTS = "models/csrnet_partA_finetuned_best.pth"

DEFAULT_IMAGE_FILES = [
    "processed_IMG_113.jpg",  # Low: ~66 count
    "processed_IMG_123.jpg",  # Low: ~115 count
    "processed_IMG_120.jpg",  # Low: ~133 count
    "processed_IMG_102.jpg",  # Medium: ~223 count
    "processed_IMG_109.jpg",  # Medium: ~379 count
    "processed_IMG_126.jpg",  # Medium: ~440 count
    "processed_IMG_10.jpg",   # High: ~502 count
    "processed_IMG_13.jpg",   # High: ~584 count
    "processed_IMG_104.jpg",  # High: ~1175 count
    "processed_IMG_117.jpg"   # High: ~1603 count
]

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate CSRNet on JHU-CROWD++ or ShanghaiTech test datasets.")
    parser.add_argument('--dataset', type=str, default='shanghai_a', choices=['jhu', 'shanghai_a', 'shanghai_b'],
                        help="Dataset type to validate: 'jhu', 'shanghai_a', or 'shanghai_b'.")
    parser.add_argument('--images_dir', type=str, default=None, help="Path to test images directory.")
    parser.add_argument('--gt_dir', type=str, default=None, help="Path to test ground-truth directory.")
    parser.add_argument('--weights', type=str, default=None, help="Path to weights file.")
    parser.add_argument('--max_images', type=int, default=None, help="Maximum number of test images to evaluate.")
    parser.add_argument('--max_size', type=int, default=1024, help="Maximum image dimension during evaluation (default: 1024).")
    parser.add_argument('--test_selected', action='store_true',
                        help="Validate only the 10 hardcoded test images (for ShanghaiTech A backward compatibility).")
    parser.add_argument('--device', type=str, default=None, help="Device to use: 'cuda' or 'cpu'.")
    return parser.parse_args()

def preprocess_validation_image(img, max_size=1024):
    """
    Consistent preprocessing with image scaling (max_size) and aspect ratio preservation.
    """
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
    else:
        scale = 1.0
        
    h_new = max(8, int(round(h * scale / 8.0)) * 8)
    w_new = max(8, int(round(w * scale / 8.0)) * 8)
    
    resized = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    
    # RGB Conversion
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb_float = rgb.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb_float - mean) / std
    
    chw = np.transpose(normalized, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0)
    return tensor

def extract_image_number(filename):
    match = re.search(r'IMG_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def main():
    args = get_args()
    
    # 1. Resolve paths based on dataset if not specified
    if args.dataset == 'jhu':
        images_dir = args.images_dir if args.images_dir else r"E:\Crowd Detection &  Prevention\archive\JHU\test\images"
        gt_dir = args.gt_dir if args.gt_dir else r"E:\Crowd Detection &  Prevention\archive\JHU\test\gt"
        weights_path = args.weights if args.weights else "models/csrnet_jhu_best.pth"
    elif args.dataset == 'shanghai_a':
        images_dir = args.images_dir if args.images_dir else DEFAULT_SHANGHAI_A_IMAGES_DIR
        gt_dir = args.gt_dir if args.gt_dir else DEFAULT_SHANGHAI_A_GT_DIR
        weights_path = args.weights if args.weights else DEFAULT_SHANGHAI_A_WEIGHTS
    else:  # shanghai_b
        images_dir = args.images_dir if args.images_dir else r"E:\Crowd Detection &  Prevention\archive\ShanghaiTech\part_B\test_data\images"
        gt_dir = args.gt_dir if args.gt_dir else r"E:\Crowd Detection &  Prevention\archive\ShanghaiTech\part_B\test_data\ground-truth"
        weights_path = args.weights if args.weights else "models/csrnet_shanghai_b_best.pth"
        
    if not os.path.exists(weights_path) and args.dataset == 'shanghai_a' and weights_path == DEFAULT_SHANGHAI_A_WEIGHTS:
        # Fallback check
        fallback_path = "models/csrnet_shanghaitech.pth"
        if os.path.exists(fallback_path):
            weights_path = fallback_path

    resolved_gt_dir = os.path.abspath(gt_dir)
    resolved_img_dir = os.path.abspath(images_dir)
    path_exists = os.path.exists(gt_dir) and os.path.exists(images_dir)
    
    print("="*70)
    print("                 CSRNET DATASET CHECK")
    print("="*70)
    print(f"Dataset selected:    {args.dataset.upper()}")
    print(f"Weights used:        {weights_path}")
    print(f"Images path:         {resolved_img_dir}")
    print(f"Ground truth path:   {resolved_gt_dir}")
    print(f"Paths exist:         {path_exists}")
    print("="*70 + "\n")
    
    if not path_exists:
        print("ERROR: Test dataset directories not found. Please verify paths.")
        return

    # 2. Setup Device (RTX 4060 GPU / CUDA if available)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. Load CSRNet Model
    if os.path.exists(weights_path):
        print(f"Loading custom weights from '{weights_path}'...")
        try:
            model = load_csrnet_model(weights_path, device)
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}. Falling back to default initialization.")
            model = CSRNet(load_weights=False).to(device)
            model.eval()
    else:
        print(f"Pretrained weights not found at '{weights_path}'. Initializing default model.")
        model = CSRNet(load_weights=False).to(device)
        model.eval()

    # 4. Gather Image Files
    if args.test_selected and args.dataset == 'shanghai_a':
        image_files = DEFAULT_IMAGE_FILES
    else:
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)])
        
    if args.max_images:
        print(f"Limiting validation to first {args.max_images} images.")
        image_files = image_files[:args.max_images]
        
    if not image_files:
        print(f"No test images found in '{images_dir}'")
        return
        
    print(f"Found {len(image_files)} test images for evaluation.")

    # 5. Process Loop
    results_list = []
    mae_accum = 0.0
    mse_accum = 0.0
    processed_count = 0
    
    for index, filename in enumerate(image_files, start=1):
        input_filename = os.path.join(images_dir, filename)
        
        frame = cv2.imread(input_filename)
        if frame is None:
            print(f"ERROR: OpenCV failed to read '{input_filename}'")
            continue
            
        # Preprocess and estimate
        input_tensor = preprocess_validation_image(frame, max_size=args.max_size).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        est_count = float(output.sum().item())
        
        # Determine ground truth count
        true_count = None
        notes = ""
        
        if args.dataset == 'jhu':
            # Parse JHU txt format
            base_name, _ = os.path.splitext(filename)
            gt_file = f"{base_name}.txt"
            gt_path = os.path.join(gt_dir, gt_file)
            if os.path.exists(gt_path):
                try:
                    lines_count = 0
                    with open(gt_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                lines_count += 1
                    true_count = lines_count
                    notes = f"Parsed JHU ground truth from {gt_file}"
                except Exception as e:
                    notes = f"Error parsing JHU gt file: {e}"
            else:
                notes = f"Ground truth txt not found for {filename}"
        else: # shanghai_a or shanghai_b
            img_number = extract_image_number(filename)
            if img_number is not None:
                gt_filename = f"GT_IMG_{img_number}.mat"
                gt_path = os.path.join(gt_dir, gt_filename)
                if os.path.exists(gt_path):
                    try:
                        mat = sio.loadmat(gt_path)
                        points = mat['image_info'][0][0][0][0][0]
                        true_count = len(points)
                        notes = f"Parsed ShanghaiTech mat from {gt_filename}"
                    except Exception as e:
                        notes = f"Error parsing mat file: {e}"
                else:
                    notes = f"Ground truth mat not found for {filename}"
            else:
                notes = "Could not extract image number from filename."

        if true_count is not None:
            err = est_count - true_count
            mae_accum += abs(err)
            mse_accum += err ** 2
            processed_count += 1
            
            error_pct = (abs(err) / true_count) * 100 if true_count > 0 else 0.0
            error_pct_str = f"{error_pct:.2f}%"
            true_count_str = str(true_count)
        else:
            true_count_str = "N/A"
            error_pct_str = "N/A"
            
        results_list.append({
            "index": index,
            "name": filename,
            "true_count": true_count_str,
            "est_count": f"{est_count:.2f}",
            "error_pct": error_pct_str,
            "notes": notes
        })
        
        if len(image_files) <= 15 or index % 50 == 0 or index == len(image_files):
            print(f"Processed {index}/{len(image_files)}: Est={est_count:.1f}, True={true_count_str}")

    # 6. Print Summary Table
    print("\n" + "="*85)
    print("                      CSRNET GROUND-TRUTH VALIDATION TABLE")
    print("="*85)
    print(f"{'Image / Filename':<24} | {'True Count':<10} | {'Estimated Count':<15} | {'Error %':<10}")
    print("-"*85)
    # Print only first 15 and last 5 if total images is large to avoid cluttering terminal
    if len(results_list) > 20:
        for res in results_list[:15]:
            print(f"{res['name']:<24} | {res['true_count']:<10} | {res['est_count']:<15} | {res['error_pct']:<10}")
        print("...")
        for res in results_list[-5:]:
            print(f"{res['name']:<24} | {res['true_count']:<10} | {res['est_count']:<15} | {res['error_pct']:<10}")
    else:
        for res in results_list:
            print(f"{res['name']:<24} | {res['true_count']:<10} | {res['est_count']:<15} | {res['error_pct']:<10}")
            
    print("="*85)
    
    # 7. Print Final Metrics
    if processed_count > 0:
        final_mae = mae_accum / processed_count
        final_rmse = (mse_accum / processed_count) ** 0.5
        print("\n" + "="*50)
        print("                  FINAL TEST EVALUATION METRICS")
        print("="*50)
        print(f"Dataset:            {args.dataset.upper()}")
        print(f"Weights:            {weights_path}")
        print(f"Total Evaluated:    {processed_count} images")
        print(f"Mean Absolute Error (MAE):  {final_mae:.4f}")
        print(f"Root Mean Square Error (RMSE): {final_rmse:.4f}")
        print("="*50)
    else:
        print("\nERROR: No images were successfully matched with ground-truth coordinates.")

if __name__ == "__main__":
    main()
