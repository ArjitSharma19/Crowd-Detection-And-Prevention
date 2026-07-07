import os
import re
import cv2
import torch
import numpy as np
import scipy.io as sio
import scipy.spatial
from torch.utils.data import Dataset

def generate_density_map_adaptive(points, original_shape, target_shape=(96, 128), k=3, beta=0.3, min_sigma=2.0, max_sigma=20.0):
    """
    Generates a ground-truth density map at the model's output resolution (target_shape)
    using geometry-adaptive Gaussian kernels via k-nearest neighbors.
    """
    h_orig, w_orig = original_shape[:2]
    h_target, w_target = target_shape
    
    density_map = np.zeros(target_shape, dtype=np.float32)
    
    if len(points) == 0:
        return density_map
        
    scale_x = w_target / w_orig
    scale_y = h_target / h_orig
    
    # Scale points to target shape coordinates
    scaled_points = np.zeros_like(points, dtype=np.float32)
    scaled_points[:, 0] = points[:, 0] * scale_x
    scaled_points[:, 1] = points[:, 1] * scale_y
    
    # Build KDTree to find nearest neighbors on the target grid
    if len(points) > 1:
        tree = scipy.spatial.KDTree(scaled_points)
        # k+1 because the query point itself is returned as the nearest neighbor (distance 0)
        query_k = min(k + 1, len(points))
        distances, _ = tree.query(scaled_points, k=query_k)
    else:
        distances = None

    for i, pt in enumerate(scaled_points):
        cx, cy = pt[0], pt[1]
        
        # Skip points that fall outside the target boundaries
        if cx < 0 or cx >= w_target or cy < 0 or cy >= h_target:
            continue
            
        # Determine adaptive sigma
        if distances is not None and len(distances[i]) > 1:
            # Average distance to k nearest neighbors (excluding self, which is index 0)
            mean_dist = np.mean(distances[i][1:])
            sigma = beta * mean_dist
            sigma = np.clip(sigma, min_sigma, max_sigma)
        else:
            # Fallback if only one point or KDTree query failed
            # Default sigma (e.g. 15 in original resolution, scaled down by average scale factor)
            avg_scale = (scale_x + scale_y) / 2.0
            sigma = 15.0 * avg_scale
            sigma = np.clip(sigma, min_sigma, max_sigma)
            
        # Place Gaussian kernel
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
    valid_points_count = sum(1 for pt in scaled_points if 0 <= pt[0] < w_target and 0 <= pt[1] < h_target)
    
    current_sum = density_map.sum()
    if current_sum > 0 and valid_points_count > 0:
        density_map = (density_map / current_sum) * valid_points_count
        
    return density_map


def augment_data(img, points, crop_size=(512, 512)):
    """
    Applies random scale jitter, horizontal flip, and random crop to the image and coordinates.
    """
    h, w = img.shape[:2]
    
    # 1. Scale Jitter (0.8x to 1.2x)
    scale_factor = np.random.uniform(0.8, 1.2)
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    # Ensure dimensions are at least crop_size
    new_h = max(new_h, crop_size[0])
    new_w = max(new_w, crop_size[1])
    
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if len(points) > 0:
        points = points * np.array([new_w / w, new_h / h], dtype=np.float32)
        
    # 2. Horizontal Flip
    if np.random.random() > 0.5:
        img = cv2.flip(img, 1)
        if len(points) > 0:
            points[:, 0] = new_w - 1 - points[:, 0]
            
    # 3. Random Crop
    h, w = img.shape[:2]
    y_start = np.random.randint(0, h - crop_size[0] + 1)
    x_start = np.random.randint(0, w - crop_size[1] + 1)
    
    img = img[y_start:y_start + crop_size[0], x_start:x_start + crop_size[1]]
    
    if len(points) > 0:
        mask = (points[:, 0] >= x_start) & (points[:, 0] < x_start + crop_size[1]) & \
               (points[:, 1] >= y_start) & (points[:, 1] < y_start + crop_size[0])
        points = points[mask]
        if len(points) > 0:
            points[:, 0] -= x_start
            points[:, 1] -= y_start
            
    return img, points


class JHUCrowdDataset(Dataset):
    """
    Custom PyTorch Dataset for JHU-CROWD++ dataset.
    Loads image, parses ground truth txt annotations, resizes/crops,
    and dynamically generates geometry-adaptive Gaussian density maps.
    """
    def __init__(self, images_dir, gt_dir, max_size=1024, filter_blur=False, is_train=False, crop_size=(512, 512)):
        self.images_dir = images_dir
        self.gt_dir = gt_dir
        self.max_size = max_size
        self.filter_blur = filter_blur
        self.is_train = is_train
        self.crop_size = crop_size
        
        self.pairs = []
        if not os.path.exists(images_dir) or not os.path.exists(gt_dir):
            print(f"Warning: JHU Dataset directory not found at: {images_dir} or {gt_dir}")
            return
            
        img_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for img_file in img_files:
            base_name, _ = os.path.splitext(img_file)
            gt_file = f"{base_name}.txt"
            gt_path = os.path.join(gt_dir, gt_file)
            if os.path.exists(gt_path):
                img_path = os.path.join(images_dir, img_file)
                
                # Check blur level filtering
                if self.filter_blur:
                    try:
                        blurs = []
                        with open(gt_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    blurs.append(int(parts[4]))
                        if blurs:
                            avg_blur = sum(blurs) / len(blurs)
                            # Exclude if average blur level is high (severe/medium blur average > 1.0)
                            if avg_blur > 1.0:
                                continue
                    except Exception as e:
                        print(f"Warning: Failed to parse blur level for {gt_path}: {e}")
                        
                self.pairs.append((img_path, gt_path))

    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        img_path, gt_path = self.pairs[idx]
        
        # Load image with OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image at {img_path}")
            
        # Parse ground truth head coordinates
        points = []
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    x, y = float(parts[0]), float(parts[1])
                    points.append([x, y])
                    
        points = np.array(points, dtype=np.float32).reshape(-1, 2) if points else np.empty((0, 2), dtype=np.float32)
        
        # Apply training augmentations if active
        if self.is_train:
            img, points = augment_data(img, points, crop_size=self.crop_size)
            
        h, w = img.shape[:2]
        
        # Resize to fixed max dimension if not cropping
        if not self.is_train:
            if max(h, w) > self.max_size:
                scale = self.max_size / max(h, w)
            else:
                scale = 1.0
                
            h_new = max(8, int(round(h * scale / 8.0)) * 8)
            w_new = max(8, int(round(w * scale / 8.0)) * 8)
            
            img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
            if len(points) > 0:
                scale_x = w_new / w
                scale_y = h_new / h
                points[:, 0] *= scale_x
                points[:, 1] *= scale_y
                
            h, w = h_new, w_new
            
        # Standard ImageNet pre-processing (RGB, scaled to [0,1], normalized)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_float = rgb_img.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb_float - mean) / std
        
        chw = np.transpose(normalized, (2, 0, 1))
        img_tensor = torch.from_numpy(chw)
        
        # Density map grid size is 1/8 of image size
        h_target, w_target = h // 8, w // 8
        
        # Generate geometry-adaptive density map on the target grid
        density_map = generate_density_map_adaptive(points, (h, w), target_shape=(h_target, w_target))
        density_tensor = torch.from_numpy(density_map).unsqueeze(0)  # (1, h_target, w_target)
        
        return img_tensor, density_tensor


class ShanghaiTechDataset(Dataset):
    """
    Custom PyTorch Dataset for ShanghaiTech Part A and Part B.
    """
    def __init__(self, pairs, max_size=1024, is_train=False, crop_size=(512, 512)):
        self.pairs = pairs
        self.max_size = max_size
        self.is_train = is_train
        self.crop_size = crop_size
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        img_path, gt_path = self.pairs[idx]
        
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image at {img_path}")
            
        mat = sio.loadmat(gt_path)
        points = mat['image_info'][0][0][0][0][0]
        points = np.array(points, dtype=np.float32).reshape(-1, 2) if len(points) > 0 else np.empty((0, 2), dtype=np.float32)
        
        if self.is_train:
            img, points = augment_data(img, points, crop_size=self.crop_size)
            
        h, w = img.shape[:2]
        
        if not self.is_train:
            if max(h, w) > self.max_size:
                scale = self.max_size / max(h, w)
            else:
                scale = 1.0
                
            h_new = max(8, int(round(h * scale / 8.0)) * 8)
            w_new = max(8, int(round(w * scale / 8.0)) * 8)
            
            img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
            if len(points) > 0:
                scale_x = w_new / w
                scale_y = h_new / h
                points[:, 0] *= scale_x
                points[:, 1] *= scale_y
                
            h, w = h_new, w_new
            
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_float = rgb_img.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb_float - mean) / std
        
        chw = np.transpose(normalized, (2, 0, 1))
        img_tensor = torch.from_numpy(chw)
        
        h_target, w_target = h // 8, w // 8
        density_map = generate_density_map_adaptive(points, (h, w), target_shape=(h_target, w_target))
        density_tensor = torch.from_numpy(density_map).unsqueeze(0)
        
        return img_tensor, density_tensor
