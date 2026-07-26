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



def estimate_density_tiled(model, frame, device, tile_size=512, stride=384):
    """
    Evaluates input frames using sliding window overlapping tiles at native image resolution.
    Accumulates density maps across tiles and normalizes overlapping regions.
    
    Args:
        model (nn.Module): Loaded CSRNet model.
        frame (np.ndarray): Input OpenCV BGR frame.
        device (torch.device): Device to run inference on.
        tile_size (int): Tile dimensions in pixels (default: 512).
        stride (int): Sliding window stride in pixels (default: 384).
        
    Returns:
        tuple: (density_map, total_count)
    """
    h, w = frame.shape[:2]
    
    # Pad image so H and W are multiples of 8 and at least tile_size
    pad_h = max(tile_size, int(np.ceil(h / 8.0)) * 8)
    pad_w = max(tile_size, int(np.ceil(w / 8.0)) * 8)
    
    target_h, target_w = pad_h // 8, pad_w // 8
    
    full_density = np.zeros((target_h, target_w), dtype=np.float32)
    weight_map = np.zeros((target_h, target_w), dtype=np.float32)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_float = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb_float - mean) / std
    
    padded_img = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    padded_img[:h, :w, :] = normalized
    
    y_steps = list(range(0, pad_h - tile_size + 1, stride))
    if y_steps[-1] < pad_h - tile_size:
        y_steps.append(pad_h - tile_size)
        
    x_steps = list(range(0, pad_w - tile_size + 1, stride))
    if x_steps[-1] < pad_w - tile_size:
        x_steps.append(pad_w - tile_size)
        
    with torch.no_grad():
        for y in y_steps:
            for x in x_steps:
                tile = padded_img[y:y+tile_size, x:x+tile_size, :]
                chw = np.transpose(tile, (2, 0, 1))
                tensor = torch.from_numpy(chw).unsqueeze(0).to(device)
                
                out = model(tensor)
                tile_density = out.squeeze().cpu().numpy()
                
                y_grid, x_grid = y // 8, x // 8
                th_grid, tw_grid = tile_size // 8, tile_size // 8
                
                full_density[y_grid:y_grid+th_grid, x_grid:x_grid+tw_grid] += tile_density
                weight_map[y_grid:y_grid+th_grid, x_grid:x_grid+tw_grid] += 1.0
                
    orig_grid_h, orig_grid_w = h // 8, w // 8
    if orig_grid_h > 0 and orig_grid_w > 0:
        full_density = full_density[:orig_grid_h, :orig_grid_w]
        weight_map = weight_map[:orig_grid_h, :orig_grid_w]
        
    full_density = full_density / np.maximum(weight_map, 1e-5)
    total_count = float(full_density.sum())
    return full_density, total_count


def estimate_density(model, frame, device, use_tiled=True, use_tta=False, scales=(0.85, 1.0, 1.15)):
    """
    Runs CSRNet inference on a single frame to predict its density map and total crowd count.
    Supports full-resolution tiled inference (default) and Multi-Scale Test-Time Augmentation (TTA).
    
    Args:
        model (nn.Module): Loaded CSRNet model.
        frame (np.ndarray): Input OpenCV BGR frame.
        device (torch.device): Device to run inference on (e.g. cpu or cuda).
        use_tiled (bool): If True, performs full-resolution sliding window tiled inference (default: True).
        use_tta (bool): If True, computes multi-scale & horizontal flip TTA (default: False).
        scales (tuple): Scale factors to evaluate during TTA.
        
    Returns:
        tuple: (density_map, total_count)
            - density_map (np.ndarray): 2D float32 array representing local density.
            - total_count (float): Sum of all density values (predicted count of people).
    """
    if use_tiled:
        return estimate_density_tiled(model, frame, device, tile_size=512, stride=384)

    if not use_tta:
        input_tensor = preprocess_frame(frame).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        density_map = output.squeeze().cpu().numpy()
        total_count = float(density_map.sum())
        return density_map, total_count
        
    # Multi-Scale TTA Pipeline
    base_w, base_h = 1024, 768
    target_shape = (base_h // 8, base_w // 8) # (96, 128)
    
    accum_density_map = np.zeros(target_shape, dtype=np.float32)
    sample_count = 0
    
    with torch.no_grad():
        for s in scales:
            w_s = max(8, int(round(base_w * s / 8.0)) * 8)
            h_s = max(8, int(round(base_h * s / 8.0)) * 8)
            
            # 1. Standard Scale Pass
            resized_frame = cv2.resize(frame, (w_s, h_s), interpolation=cv2.INTER_LINEAR)
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            rgb_float = rgb_frame.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (rgb_float - mean) / std
            chw = np.transpose(normalized, (2, 0, 1))
            tensor = torch.from_numpy(chw).unsqueeze(0).to(device)
            
            output = model(tensor)
            map_s = output.squeeze().cpu().numpy()
            orig_sum = map_s.sum()
            
            # Resize map back to base target shape (128, 96) and re-normalize count
            resized_map = cv2.resize(map_s, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
            curr_sum = resized_map.sum()
            if curr_sum > 0:
                resized_map = resized_map * (orig_sum / curr_sum)
            accum_density_map += resized_map
            sample_count += 1
            
            # 2. Horizontally Flipped Pass
            flip_frame = cv2.flip(resized_frame, 1)
            rgb_flip = cv2.cvtColor(flip_frame, cv2.COLOR_BGR2RGB)
            rgb_flip_float = rgb_flip.astype(np.float32) / 255.0
            normalized_flip = (rgb_flip_float - mean) / std
            chw_flip = np.transpose(normalized_flip, (2, 0, 1))
            tensor_flip = torch.from_numpy(chw_flip).unsqueeze(0).to(device)
            
            output_flip = model(tensor_flip)
            map_flip = output_flip.squeeze().cpu().numpy()
            orig_flip_sum = map_flip.sum()
            
            # Unflip output map horizontally
            unflipped_map = np.fliplr(map_flip)
            resized_flip_map = cv2.resize(unflipped_map, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
            curr_flip_sum = resized_flip_map.sum()
            if curr_flip_sum > 0:
                resized_flip_map = resized_flip_map * (orig_flip_sum / curr_flip_sum)
            accum_density_map += resized_flip_map
            sample_count += 1

    final_density_map = accum_density_map / max(1, sample_count)
    total_count = float(final_density_map.sum())
    return final_density_map, total_count


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
