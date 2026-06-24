import cv2
import numpy as np
import torch
from src.csrnet_model import CSRNet

def preprocess_frame(frame):
    """
    Converts a BGR OpenCV frame (numpy array) to the format expected by CSRNet.
    Steps:
      1. Resizes the frame to exactly 1024x768 pixels.
      2. Converts BGR to RGB color space.
      3. Scales pixel values to [0.0, 1.0].
      4. Normalizes using ImageNet mean and standard deviation.
      5. Reorders dimensions from HWC to CHW.
      6. Adds a batch dimension (1, 3, H, W).
      
    Args:
        frame (np.ndarray): OpenCV BGR frame of shape (H, W, 3).
        
    Returns:
        torch.Tensor: Preprocessed frame tensor of shape (1, 3, 768, 1024).
    """
    # CSRNet's density output is scale-sensitive even though the architecture is fully
    # convolutional — feeding it images at the wrong resolution relative to training
    # data causes systematic count errors. We resize to the native resolution of 
    # ShanghaiTech Part B images (1024x768) to ensure correct counting.
    # Note: cv2.resize expects dsize in (width, height) format.
    resized_frame = cv2.resize(frame, (1024, 768), interpolation=cv2.INTER_LINEAR)
    
    # Convert from BGR to RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Scale to [0.0, 1.0]
    rgb_float = rgb_frame.astype(np.float32) / 255.0
    
    # Standard ImageNet normalization coefficients
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    normalized = (rgb_float - mean) / std
    
    # Transpose layout from Height-Width-Channel (HWC) to Channel-Height-Width (CHW)
    chw = np.transpose(normalized, (2, 0, 1))
    
    # Add batch dimension to return (1, 3, H, W)
    tensor = torch.from_numpy(chw).unsqueeze(0)
    
    return tensor



def estimate_density(model, frame, device):
    """
    Runs CSRNet inference on a single frame to predict its density map and total crowd count.
    
    Args:
        model (nn.Module): Loaded CSRNet model.
        frame (np.ndarray): Input OpenCV BGR frame.
        device (torch.device): Device to run inference on (e.g. cpu or cuda).
        
    Returns:
        tuple: (density_map, total_count)
            - density_map (np.ndarray): 2D float32 array representing local density.
            - total_count (float): Sum of all density values (predicted count of people).
    """
    # Preprocess BGR frame to PyTorch tensor
    input_tensor = preprocess_frame(frame).to(device)
    
    # Perform forward pass without tracking gradients (inference mode)
    with torch.no_grad():
        output = model(input_tensor)
        
    # CSRNet output has shape (batch_size=1, channels=1, H_out, W_out).
    # Squeeze dimensions to obtain a clean 2D density map array.
    density_map = output.squeeze().cpu().numpy()
    
    # The sum of all density values represents the total estimated number of people.
    total_count = float(density_map.sum())
    
    return density_map, total_count


def get_zone_densities(density_map, grid_rows=3, grid_cols=3):
    """
    Divides the 2D density map into a grid of cells and returns the estimated
    people counts (sum of density values) in each zone.
    
    Args:
        density_map (np.ndarray): 2D float32 density map predicted by CSRNet.
        grid_rows (int): Number of rows in the output grid (default: 3).
        grid_cols (int): Number of columns in the output grid (default: 3).
        
    Returns:
        np.ndarray: A 2D float32 array of shape (grid_rows, grid_cols) containing
                    estimated crowd count inside each grid zone.
    """
    h, w = density_map.shape
    grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    
    # Calculate cell height and width boundaries on the density map
    cell_h = h / grid_rows
    cell_w = w / grid_cols
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            # Compute starting and ending pixel coordinates for the current cell
            y_start = int(r * cell_h)
            y_end = int((r + 1) * cell_h) if r < grid_rows - 1 else h
            x_start = int(c * cell_w)
            x_end = int((c + 1) * cell_w) if c < grid_cols - 1 else w
            
            # Sum the density map within these cell boundaries
            grid[r, c] = float(density_map[y_start:y_end, x_start:x_end].sum())
            
    return grid


def density_map_to_heatmap(density_map, original_frame, alpha=0.5):
    """
    Generates a color heatmap overlay from the density map and blends it onto the original frame.
    
    Args:
        density_map (np.ndarray): 2D float32 density map from CSRNet.
        original_frame (np.ndarray): Original BGR frame.
        alpha (float): Blending weight for the overlay (default: 0.5).
        
    Returns:
        np.ndarray: BGR frame with the color heatmap overlay applied.
    """
    h, w = original_frame.shape[:2]
    
    # 1. Normalize density map values to [0, 255] range for colormapping
    max_val = density_map.max()
    if max_val > 0:
        normalized = (density_map / max_val) * 255.0
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(density_map, dtype=np.uint8)
        
    # 2. Resize normalized density map back to original frame size (using bicubic for smoothness)
    resized_map = cv2.resize(normalized, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # 3. Apply standard JET colormap (Blue -> Green -> Yellow -> Red)
    colormap = cv2.applyColorMap(resized_map, cv2.COLORMAP_JET)
    
    # 4. Mask out zero/very low density areas to keep them fully transparent (no blue tint on background)
    # 5 is a threshold value to suppress background noise.
    mask = resized_map > 5
    
    heatmap_overlay = np.zeros_like(original_frame)
    heatmap_overlay[mask] = colormap[mask]
    
    # 5. Blend original image and the color heatmap overlay
    blended_frame = cv2.addWeighted(original_frame, 1.0, heatmap_overlay, alpha, 0)
    
    return blended_frame
