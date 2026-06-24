import os
import re
import cv2
import time
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader

# Import custom CSRNet modules
from src.csrnet_model import load_csrnet_model, CSRNet
from src.csrnet_inference import preprocess_frame

# =====================================================================
# CONFIGURATION & HYPERPARAMETERS
# =====================================================================
# Path to ShanghaiTech Part A training data
DATASET_DIR = r"E:\Crowd Detection &  Prevention\archive\ShanghaiTech\part_A\train_data"

# Starting baseline model weights (Part B weights) - DO NOT OVERWRITE
WEIGHTS_PATH = "models/csrnet_shanghaitech.pth"

# Training Parameters
# Batch size is kept small (2) to prevent out-of-memory (OOM) errors on the 8GB RTX 4060 GPU
BATCH_SIZE = 2
VAL_SPLIT = 0.2  # Hold out 20% of the dataset for validation
NUM_EPOCHS = 10  # Fine-tuning needs far fewer epochs than training from scratch

# LEARNING RATE TUNING TIP:
# We use a very low learning rate (1e-6) because we are adapting/fine-tuning existing weights.
# Setting the learning rate too high (e.g. 1e-3 or 1e-4) risks destroying the general crowd-counting
# features learned on Part B, leading to model degradation.
LEARNING_RATE = 1e-6
# =====================================================================

def generate_density_map(points, original_shape, target_shape=(96, 128), sigma=3.0):
    """
    Generates a ground-truth density map at the model's output resolution (target_shape).
    Applies scaling factors to map the original point coordinates to the resized target grid.
    Places a normalized Gaussian kernel at each mapped point to ensure the sum of the 
    density map matches the total headcount.
    
    Args:
        points (np.ndarray): Array of [x, y] coordinates from the .mat file.
        original_shape (tuple): Shape of the original image (height, width).
        target_shape (tuple): Target shape of the density map (height, width).
        sigma (float): Standard deviation of the Gaussian kernel on the target grid (default: 3.0).
        
    Returns:
        np.ndarray: 2D density map of shape target_shape.
    """
    h_orig, w_orig = original_shape[:2]
    h_target, w_target = target_shape
    
    density_map = np.zeros(target_shape, dtype=np.float32)
    
    # Calculate scale factors from original image to target density map grid
    scale_x = w_target / w_orig
    scale_y = h_target / h_orig
    
    for pt in points:
        x_orig, y_orig = pt
        
        # Mapped coordinates on the target density map grid (96x128)
        cx = x_orig * scale_x
        cy = y_orig * scale_y
        
        # Skip points that fall outside the frame boundaries
        if cx < 0 or cx >= w_target or cy < 0 or cy >= h_target:
            continue
            
        # Place a local Gaussian kernel around (cx, cy)
        # local window of size 4 * sigma is used for speed
        radius = int(4 * sigma)
        x_min = max(0, int(cx - radius))
        x_max = min(w_target, int(cx + radius + 1))
        y_min = max(0, int(cy - radius))
        y_max = min(h_target, int(cy + radius + 1))
        
        # Compute Gaussian values in this window
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                dist_sq = (x - cx)**2 + (y - cy)**2
                val = np.exp(-dist_sq / (2 * sigma**2))
                density_map[y, x] += val
                
    # Normalize the entire density map to match the total count of valid points
    # inside the frame boundary. This is robust against edge clipping.
    valid_points_count = sum(1 for pt in points if 0 <= pt[0]*scale_x < w_target and 0 <= pt[1]*scale_y < h_target)
    
    current_sum = density_map.sum()
    if current_sum > 0 and valid_points_count > 0:
        density_map = (density_map / current_sum) * valid_points_count
        
    return density_map


class ShanghaiTechPartADataset(Dataset):
    """
    Custom PyTorch Dataset for loading ShanghaiTech Part A images and generating
    matching density map ground-truth labels dynamically.
    """
    def __init__(self, image_gt_pairs):
        self.pairs = image_gt_pairs
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        img_path, gt_path = self.pairs[idx]
        
        # Load image with OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image at {img_path}")
        h, w = img.shape[:2]
        
        # Preprocess image to tensor of shape (3, 768, 1024)
        # preprocess_frame returns shape (1, 3, 768, 1024), we squeeze out the batch dim
        img_tensor = preprocess_frame(img).squeeze(0)
        
        # Load ground truth points from .mat file
        mat = sio.loadmat(gt_path)
        points = mat['image_info'][0][0][0][0][0]
        
        # Generate target density map of shape (96, 128)
        density_map = generate_density_map(points, (h, w), target_shape=(96, 128), sigma=3.0)
        density_tensor = torch.from_numpy(density_map).unsqueeze(0)  # Add channel dimension: (1, 96, 128)
        
        return img_tensor, density_tensor


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=30, fill='#'):
    """
    Prints a text-based progress bar to standard output to avoid tqdm dependency errors.
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()


def main():
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Setup Device (RTX 4060 GPU / CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 2. Align images and ground-truth .mat files
    img_dir = os.path.join(DATASET_DIR, "images")
    gt_dir = os.path.join(DATASET_DIR, "ground-truth")

    if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
        print(f"CRITICAL ERROR: Dataset folders do not exist inside '{DATASET_DIR}'.")
        print("Please verify your DATASET_DIR path in the configuration section.")
        return

    # Match files (e.g. processed_IMG_1.jpg -> GT_IMG_1.mat)
    all_imgs = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if f.endswith('.jpg')]
    valid_pairs = []
    for img_path in all_imgs:
        img_name = os.path.basename(img_path)
        match = re.search(r'IMG_\d+', img_name)
        if match:
            img_id = match.group(0)  # e.g., 'IMG_1'
            gt_filename = f"GT_{img_id}.mat"
            gt_path = os.path.join(gt_dir, gt_filename)
            if os.path.exists(gt_path):
                valid_pairs.append((img_path, gt_path))

    total_pairs = len(valid_pairs)
    print(f"Dataset summary: Found {total_pairs} valid image/GT pairs.")
    if total_pairs == 0:
        print("ERROR: No matching files found. Check filename patterns.")
        return

    # 3. Create Train / Validation Split (80% Train, 20% Val)
    val_size = int(VAL_SPLIT * total_pairs)
    train_size = total_pairs - val_size

    # Deterministic shuffle
    indices = list(range(total_pairs))
    np.random.shuffle(indices)

    train_pairs = [valid_pairs[i] for i in indices[:train_size]]
    val_pairs = [valid_pairs[i] for i in indices[train_size:]]

    print(f"Train split size: {len(train_pairs)}")
    print(f"Val split size:   {len(val_pairs)}")

    # Create PyTorch datasets and dataloaders
    train_dataset = ShanghaiTechPartADataset(train_pairs)
    val_dataset = ShanghaiTechPartADataset(val_pairs)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. Load Baseline CSRNet Model (Pretrained on Part B)
    if os.path.exists(WEIGHTS_PATH):
        print(f"\nLoading baseline pretrained CSRNet weights from '{WEIGHTS_PATH}'...")
        model = load_csrnet_model(WEIGHTS_PATH, device)
        print("Model loaded successfully. Preparing for fine-tuning...")
    else:
        print(f"\nERROR: Pretrained weights not found at '{WEIGHTS_PATH}'.")
        print("Fine-tuning requires a valid starting checkpoint. Please verify the file path.")
        return

    # Set model to training mode (enables gradients and updates BatchNorm status if applicable)
    model.train()

    # 5. Loss Criterion and Optimizer Setup
    # MSE loss is standard for density map regression in crowd counting.
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Track best validation performance
    best_val_loss = float('inf')
    best_epoch = -1
    best_checkpoint_path = ""
    
    # Ensure models directory exists to save checkpoints
    os.makedirs("models", exist_ok=True)

    print("\n" + "="*70)
    print("                     STARTING FINE-TUNING LOOP")
    print("="*70)
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size:    {BATCH_SIZE}")
    print(f"Total Epochs:  {NUM_EPOCHS}")
    print("*" * 70)
    
    # SAFETY NOTE FOR TRAINING MONITOR:
    # Initial fine-tuning epochs may show validation error temporarily INCREASE before improving,
    # since the model is adapting from Part B's sparse-crowd assumptions to Part A's dense-crowd
    # distribution. This is expected behavior and not a bug.
    print("SAFETY NOTICE: During the first 1-3 epochs, validation loss might temporarily spike")
    print("as the model adapts from sparse Part B crowds to dense Part A distributions. This is normal.")
    print("*" * 70 + "\n")

    # 6. Epoch Training Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_batches = len(train_loader)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass & Optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Progress print
            print_progress_bar(batch_idx, train_batches, 
                               prefix=f"Epoch {epoch}/{NUM_EPOCHS} [Train]", 
                               suffix=f"Loss: {loss.item():.6f}")

        avg_train_loss = train_loss / train_batches

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader, start=1):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                print_progress_bar(batch_idx, val_batches, 
                                   prefix=f"Epoch {epoch}/{NUM_EPOCHS} [Val]  ", 
                                   suffix=f"Loss: {loss.item():.6f}")
                
        avg_val_loss = val_loss / val_batches
        epoch_time = time.time() - epoch_start_time
        
        print(f"Summary -> Epoch {epoch:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Time: {epoch_time:.1f}s")
        
        # 7. Checkpoint Saving
        checkpoint_name = f"csrnet_partA_finetuned_epoch{epoch}.pth"
        checkpoint_path = os.path.join("models", checkpoint_name)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"-> Saved epoch checkpoint: {checkpoint_path}")
        
        # Track and save the best-performing model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_checkpoint_path = os.path.join("models", "csrnet_partA_finetuned_best.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"-> *** NEW BEST MODEL *** saved: {best_checkpoint_path}")
            
        print("-" * 70)

    # 8. Fine-tuning Summary
    print("\n" + "="*70)
    print("                    FINE-TUNING COMPLETED")
    print("="*70)
    print(f"Best Validation Epoch: {best_epoch}")
    print(f"Best Validation Loss:  {best_val_loss:.6f}")
    print(f"Best Model Saved to:   {best_checkpoint_path}")
    print("="*70)

if __name__ == "__main__":
    main()
