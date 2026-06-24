import os
import cv2
import torch
import numpy as np

# Import custom CSRNet modules
from src.csrnet_model import load_csrnet_model, CSRNet
from src.csrnet_inference import estimate_density, get_zone_densities, density_map_to_heatmap

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
# List of image URLs to test
IMAGE_URLS = [
    # 1. ShanghaiTech Part B Test Image (Kaggle dataset URL from user request)
    "https://storage.googleapis.com/kagglesdsdata/datasets/5953823/9729357/ShanghaiTech/part_B/test_data/images/processed_IMG_1.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260624%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260624T095814Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=7388342f65a41913230b91353854fc4cc4a937e8ac113a027cee86df905f8ccd63cae44cf2a44112dbc7c0e67fb28c0188a1624a32c23ef9dc8bb3387e157ce60b0b327ded98c8b2f31e830ba0779a2de13e16c21891b1e080925eba7e7f834bda0d6f6df8500e90e2af9646670083dd2a1d387d2b76dbd759cb340ca538cfa8d6605b5e055dedc7a5b46eb051ee0d84e74116b16bd2b7c4fc5265fdacefcfcf7d423cb3bc1872631929a67e6f89581b1244fe7a2d65d0d644a4d93934047a016732cfa7ae258bf217010566e17d58c13d6a614959805bd8e37383aa14505f6d5ab9f8252f73452a592a69ae7d19e022d10dce1f2c6f7b894cdb1dcae8b0c464",
    # 2. Busy city crowd sample
    "https://images.stockcake.com/public/6/9/9/699b12b1-02d5-409a-89f2-7e885b66a0f2/busy-city-crowd-stockcake.jpg",
    # 3. Packed concert crowd sample
    "https://images.stockcake.com/public/8/8/6/8863f639-65b1-4f30-84cf-d54b8d7eb80a_large/packed-concert-crowd-stockcake.jpg"
]

WEIGHTS_PATH = "models/csrnet_shanghaitech.pth"
# =====================================================================

def download_image(url, save_path):
    """
    Downloads an image from a URL and saves it locally.
    Supports requests with a fallback to urllib.
    
    Args:
        url (str): Image URL.
        save_path (str): Destination local file path.
        
    Returns:
        bool: True if download succeeded, False otherwise.
    """
    print(f"Downloading: {url[:100]}...")
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

    # 3. Process each image URL in sequence
    for index, url in enumerate(IMAGE_URLS, start=1):
        input_filename = f"test_input_{index}.jpg"
        output_filename = f"test_heatmap_{index}.jpg"
        
        print("\n" + "="*70)
        print(f"--- Image {index}: {url[:60]}... ---")
        print("="*70)
        
        # Download image
        if not download_image(url, input_filename):
            print(f"Skipping Image {index} due to download failure.")
            summary_results.append((index, "FAILED (Download error)"))
            continue
            
        # Load with OpenCV
        frame = cv2.imread(input_filename)
        if frame is None:
            print(f"ERROR: OpenCV failed to read '{input_filename}'")
            summary_results.append((index, "FAILED (Read error)"))
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
        summary_results.append((index, f"{total_count:.2f}"))

    # 4. Print Summary Table
    print("\n" + "="*50)
    print("                 SUMMARY TABLE")
    print("="*50)
    print(f"{'Image No.':<12} | {'Estimated Count / Status':<25}")
    print("-"*50)
    for idx, count_status in summary_results:
        print(f"Image {idx:<6} | {count_status:<25}")
    print("="*50)

if __name__ == "__main__":
    main()
