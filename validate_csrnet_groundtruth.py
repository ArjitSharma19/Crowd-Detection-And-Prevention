import os
import re
import cv2
import torch
import numpy as np
import scipy.io as sio

# Import custom CSRNet modules
from src.csrnet_model import load_csrnet_model, CSRNet
from src.csrnet_inference import estimate_density, preprocess_frame

# Try to use requests library, fallback to built-in urllib if not installed
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    HAS_REQUESTS = False

# =====================================================================
# CONFIGURATION
# =====================================================================
# Directory containing ShanghaiTech Part A ground-truth .mat files
GROUND_TRUTH_DIR = r"E:\Crowd Detection &  Prevention\archive\ShanghaiTech\part_A\test_data\ground-truth"

# Directory containing matching ShanghaiTech Part A images locally
IMAGES_DIR = r"E:\Crowd Detection &  Prevention\archive\ShanghaiTech\part_A\test_data\images"

# Path to the pretrained CSRNet model weights
# SWITCH HERE:
# WEIGHTS_PATH = "models/csrnet_shanghaitech.pth"  # original, Part-B-only
WEIGHTS_PATH = "models/csrnet_partA_finetuned_best.pth"  # new, fine-tuned

# Selected 10 test images from Part A spanning a range of crowd sizes (low, medium, high)
IMAGE_FILES = [
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
# =====================================================================

def download_image(url, save_path):
    """
    Downloads an image from a URL and saves it locally.
    Supports requests with a fallback to urllib.
    """
    print(f"Downloading: {url[:80]}...")
    try:
        if HAS_REQUESTS:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(response.content)
        else:
            req = urllib.request.Request(
                url, 
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=20) as response:
                with open(save_path, "wb") as f:
                    f.write(response.read())
        print(f"-> Saved as: {save_path}")
        return True
    except Exception as e:
        print(f"-> ERROR downloading URL: {e}")
        return False

def extract_image_number(url):
    """
    Tries to extract the image number from the URL filename (e.g. 'processed_IMG_45.jpg' -> 45).
    
    Args:
        url (str): The image URL string.
        
    Returns:
        int or None: The extracted number, or None if no match is found.
    """
    # Look for 'IMG_{digits}' in the URL
    match = re.search(r'IMG_(\d+)', url)
    if match:
        return int(match.group(1))
    return None

def main():
    # 0. Startup dataset checks
    resolved_gt_dir = os.path.abspath(GROUND_TRUTH_DIR)
    path_exists = os.path.exists(GROUND_TRUTH_DIR)
    mat_count = 0
    if path_exists:
        try:
            mat_count = len([f for f in os.listdir(GROUND_TRUTH_DIR) if f.endswith('.mat')])
        except Exception as e:
            print(f"Error listing directory {GROUND_TRUTH_DIR}: {e}")
            
    print("="*70)
    print("                 CSRNET DATASET CHECK")
    print("="*70)
    print(f"Weights used:        {WEIGHTS_PATH}")
    print(f"Ground truth path:   {resolved_gt_dir}")
    print(f"Path exists:         {path_exists}")
    print(f"GT .mat files found: {mat_count}")
    print("="*70 + "\n")

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

    # List to store results for final comparison table
    results_list = []

    # 3. Process each image file in sequence
    for index, url in enumerate(IMAGE_FILES, start=1):
        input_filename = os.path.join(IMAGES_DIR, url)
        print("\n" + "="*70)
        print(f"Processing Image {index}: {url}...")
        print("="*70)

        # Load with OpenCV
        frame = cv2.imread(input_filename)
        if frame is None:
            print(f"ERROR: OpenCV failed to read '{input_filename}'")
            results_list.append({
                "index": index,
                "name": f"{url} (Read Fail)",
                "true_count": "N/A",
                "est_count": "N/A",
                "error_pct": "N/A",
                "notes": "Failed to read image with cv2."
            })
            continue

        # Run inference
        print("Running CSRNet inference...")
        density_map, est_count = estimate_density(model, frame, device)

        # Determine if it's a ShanghaiTech image and find ground truth
        img_number = extract_image_number(url)
        true_count = None
        notes = "No ground truth available — not a ShanghaiTech dataset image."

        if img_number is not None:
            gt_filename = f"GT_IMG_{img_number}.mat"
            gt_path = os.path.join(GROUND_TRUTH_DIR, gt_filename)
            
            print(f"Identified as ShanghaiTech image #{img_number}.")
            print(f"Checking for ground-truth at: {gt_path}")
            
            if os.path.exists(gt_path):
                try:
                    # Load .mat file
                    mat = sio.loadmat(gt_path)
                    
                    # Validate nested structure to extract point coordinates
                    # Standard structure: mat['image_info'][0][0][0][0][0]
                    # Let's perform validation checks on the nested arrays to avoid crashes
                    if 'image_info' in mat:
                        image_info = mat['image_info']
                        
                        try:
                            # Access the points array
                            points = image_info[0][0][0][0][0]
                            true_count = len(points)
                            notes = f"Ground truth successfully loaded from {gt_filename}"
                            print(f"Ground truth loaded. Real count: {true_count}")
                        except IndexError as ie:
                            print(f"WARNING: Mat file has unexpected structure within 'image_info'.")
                            print(f"Mat keys: {list(mat.keys())}")
                            print(f"image_info type/shape: {type(image_info)} / {getattr(image_info, 'shape', 'N/A')}")
                            print(f"Detailed structure warning: {ie}")
                            notes = "Mat file structure mismatch during nested field access."
                    else:
                        print(f"WARNING: Mat file is missing 'image_info' key. Keys found: {list(mat.keys())}")
                        notes = "Mat file is missing 'image_info' key."
                        
                except Exception as e:
                    print(f"ERROR loading ground-truth .mat file: {e}")
                    notes = f"Failed to load/parse mat file: {e}"
            else:
                print(f"Ground-truth file not found at: {gt_path}")
                notes = f"Ground-truth mat file not found (expecting {gt_filename})."

        # Calculate errors if ground-truth is available
        if true_count is not None:
            abs_err = abs(true_count - est_count)
            error_pct = (abs_err / true_count) * 100 if true_count > 0 else 0.0
            error_pct_str = f"{error_pct:.2f}%"
            true_count_str = str(true_count)
        else:
            true_count_str = "N/A"
            error_pct_str = "N/A"

        # Record results
        img_name = f"processed_IMG_{img_number}.jpg" if img_number else f"val_input_{index}.jpg"
        results_list.append({
            "index": index,
            "name": img_name,
            "true_count": true_count_str,
            "est_count": f"{est_count:.2f}",
            "error_pct": error_pct_str,
            "notes": notes
        })

    # 4. Print Comparison Table
    print("\n" + "="*85)
    print("                      CSRNET GROUND-TRUTH VALIDATION TABLE")
    print("="*85)
    print(f"{'Image / Filename':<24} | {'True Count':<10} | {'Estimated Count':<15} | {'Error %':<10}")
    print("-"*85)
    for res in results_list:
        print(f"{res['name']:<24} | {res['true_count']:<10} | {res['est_count']:<15} | {res['error_pct']:<10}")
    print("="*85)
    
    # Print notes for detail
    print("\nProcessing Notes:")
    for res in results_list:
        print(f"  - Image {res['index']} ({res['name']}): {res['notes']}")
    print("="*85)

if __name__ == "__main__":
    main()
