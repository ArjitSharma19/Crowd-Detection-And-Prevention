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
# Directory containing ShanghaiTech Part B ground-truth .mat files
GROUND_TRUTH_DIR = r"E:\Crowd Detection &  Prevention\archive\ShanghaiTech\part_B\test_data\ground-truth"

# Directory containing matching ShanghaiTech Part B images locally
IMAGES_DIR = r"E:\Crowd Detection &  Prevention\archive\ShanghaiTech\part_B\test_data\images"

# Path to the pretrained CSRNet model weights
WEIGHTS_PATH = "models/csrnet_shanghaitech.pth"

# List of image URLs to test (reused from test_csrnet_image.py)
IMAGE_URLS = [
    # 1. ShanghaiTech Part B Test Image (Kaggle dataset URL)
    "https://storage.googleapis.com/kagglesdsdata/datasets/5953823/9729357/ShanghaiTech/part_B/test_data/images/processed_IMG_1.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260624%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260624T095814Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=7388342f65a41913230b91353854fc4cc4a937e8ac113a027cee86df905f8ccd63cae44cf2a44112dbc7c0e67fb28c0188a1624a32c23ef9dc8bb3387e157ce60b0b327ded98c8b2f31e830ba0779a2de13e16c21891b1e080925eba7e7f834bda0d6f6df8500e90e2af9646670083dd2a1d387d2b76dbd759cb340ca538cfa8d6605b5e055dedc7a5b46eb051ee0d84e74116b16bd2b7c4fc5265fdacefcfcf7d423cb3bc1872631929a67e6f89581b1244fe7a2d65d0d644a4d93934047a016732cfa7ae258bf217010566e17d58c13d6a614959805bd8e37383aa14505f6d5ab9f8252f73452a592a69ae7d19e022d10dce1f2c6f7b894cdb1dcae8b0c464",
    # 2. Busy city crowd sample (Non-ShanghaiTech)
    "https://images.stockcake.com/public/6/9/9/699b12b1-02d5-409a-89f2-7e885b66a0f2/busy-city-crowd-stockcake.jpg",
    # 3. Packed concert crowd sample (Non-ShanghaiTech)
    "https://images.stockcake.com/public/8/8/6/8863f639-65b1-4f30-84cf-d54b8d7eb80a/packed-concert-crowd-stockcake.jpg"
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

    # 3. Process each image URL in sequence
    for index, url in enumerate(IMAGE_URLS, start=1):
        input_filename = f"val_input_{index}.jpg"
        print("\n" + "="*70)
        print(f"Processing Image {index}...")
        print("="*70)

        # Download image
        if not download_image(url, input_filename):
            print(f"Skipping Image {index} due to download failure.")
            results_list.append({
                "index": index,
                "name": f"Image {index} (Download Fail)",
                "true_count": "N/A",
                "est_count": "N/A",
                "error_pct": "N/A",
                "notes": "Failed to download image."
            })
            continue

        # Load with OpenCV
        frame = cv2.imread(input_filename)
        if frame is None:
            print(f"ERROR: OpenCV failed to read '{input_filename}'")
            results_list.append({
                "index": index,
                "name": f"Image {index} (Read Fail)",
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
