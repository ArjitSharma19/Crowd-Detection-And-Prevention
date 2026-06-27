import cv2
import numpy as np

class DensityEstimator:
    def __init__(self, grid_rows=8, grid_cols=8):
        """
        Initializes the density estimator.
        Divides the camera view into a grid of cells to calculate local density distributions.
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    def calculate_grid_density(self, frame_shape, detections):
        """
        Calculates density values across a grid.
        
        Args:
            frame_shape: tuple of (height, width, channels)
            detections: list of detections from detector.py
            
        Returns:
            np.ndarray: grid of shape (grid_rows, grid_cols) containing person counts per grid cell.
        """
        h, w = frame_shape[:2]
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        cell_w = w / self.grid_cols
        cell_h = h / self.grid_rows
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # Find the bottom-center of the bounding box (represents the person's standing position)
            cx = int((x1 + x2) / 2)
            cy = int(y2)
            
            # Map center point to grid coordinate
            col = int(cx // cell_w)
            row = int(cy // cell_h)
            
            # Clip bounds to avoid indexing errors
            col = max(0, min(col, self.grid_cols - 1))
            row = max(0, min(row, self.grid_rows - 1))
            
            grid[row, col] += 1.0
            
        return grid

    def generate_heatmap(self, frame, detections, alpha=0.5, kernel_size=151):
        """
        Generates a visually striking smooth density heatmap overlay on the frame.
        
        Args:
            frame: opencv BGR image matrix
            detections: list of detections
            alpha: float, transparency level of the heatmap overlay
            kernel_size: odd int, size of Gaussian blur to apply for smoothing the hotspots
            
        Returns:
            heatmap_frame: opencv BGR image with heatmap overlay
            max_density_score: float, maximum single point density value in the current frame
        """
        h, w = frame.shape[:2]
        # Create a single-channel float matrix for accumulating density
        density_matrix = np.zeros((h, w), dtype=np.float32)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx = int((x1 + x2) / 2)
            cy = int(y2)  # Base/feet position
            
            # Draw a circle on the density matrix at the person's feet.
            # Scale the radius depending on bounding box size to simulate distance perspective
            box_width = x2 - x1
            radius = max(20, int(box_width * 0.8))
            
            # Create a localized source of density
            temp_mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(temp_mask, (cx, cy), radius, 1.0, -1)
            
            # Smooth the individual detection circle to create a gradient blob
            blurred_mask = cv2.GaussianBlur(temp_mask, (kernel_size, kernel_size), 0)
            
            # Accumulate
            density_matrix += blurred_mask
            
        # Normalize the density matrix to [0, 255] for colormap application
        max_val = np.max(density_matrix)
        if max_val > 0:
            normalized_matrix = np.uint8(np.minimum((density_matrix / max_val) * 255, 255))
        else:
            normalized_matrix = np.zeros((h, w), dtype=np.uint8)
            
        # Apply JET colormap (Blue -> Green -> Yellow -> Red)
        colormap = cv2.applyColorMap(normalized_matrix, cv2.COLORMAP_JET)
        
        # Suppress the colormap in areas with 0 density (which are colored deep blue in standard JET)
        # We only apply colormap to pixels where normalized_matrix is non-zero
        mask = normalized_matrix > 5
        heatmap_overlay = np.zeros_like(frame)
        heatmap_overlay[mask] = colormap[mask]
        
        # Blend the heatmap overlay with the original frame
        heatmap_frame = cv2.addWeighted(frame, 1.0, heatmap_overlay, alpha, 0)
        
        return heatmap_frame, float(max_val)

    def draw_grid_boundaries(self, frame, grid):
        """
        Optionally draws grid lines on the image with density counts.
        """
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        cell_w = int(w / self.grid_cols)
        cell_h = int(h / self.grid_rows)
        
        # Draw columns
        for col in range(1, self.grid_cols):
            x = col * cell_w
            cv2.line(annotated_frame, (x, 0), (x, h), (100, 100, 100), 1, cv2.LINE_AA)
            
        # Draw rows
        for row in range(1, self.grid_rows):
            y = row * cell_h
            cv2.line(annotated_frame, (0, y), (w, y), (100, 100, 100), 1, cv2.LINE_AA)
            
        # Write density counts inside cells
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                val = int(grid[r, c])
                if val > 0:
                    text_x = int(c * cell_w + cell_w / 2 - 5)
                    text_y = int(r * cell_h + cell_h / 2 + 5)
                    # Color goes green to red based on count
                    color = (0, 255, 0) if val <= 2 else ((0, 255, 255) if val <= 4 else (0, 0, 255))
                    cv2.putText(annotated_frame, str(val), (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                    
        return annotated_frame

# =====================================================================
# DUAL-MODEL SWITCHING & ZONE RISK HEURISTICS
# =====================================================================

def get_yolo_zone_counts(yolo_boxes, frame_shape, grid_rows=3, grid_cols=3):
    """
    Takes YOLO bounding boxes and assigns each detected person to a grid zone
    based on the center point of their bounding box.
    
    This function outputs a 2D grid of per-zone counts in the EXACT same format/shape
    as CSRNet's get_zone_densities() output. This allows the rest of the risk
    assessment pipeline to treat both models' outputs interchangeably.
    
    Args:
        yolo_boxes (list): A list of bounding boxes. Supports:
                            - [{'bbox': [x1, y1, x2, y2]}] (output from detector.detect)
                            - [[x1, y1, x2, y2]] (raw coordinates)
        frame_shape (tuple): Shape of the camera frame (height, width, channels).
        grid_rows (int): Number of rows in the spatial grid (default: 3).
        grid_cols (int): Number of columns in the spatial grid (default: 3).
        
    Returns:
        np.ndarray: A 2D numpy array of shape (grid_rows, grid_cols) containing
                    the count of detected persons whose box centers fall into each cell.
    """
    h, w = frame_shape[:2]
    grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    
    # Grid cell dimensions
    cell_h = h / grid_rows
    cell_w = w / grid_cols
    
    for item in yolo_boxes:
        # Check if the box is a dictionary (from detector.py) or a raw list/tuple
        if isinstance(item, dict) and 'bbox' in item:
            box = item['bbox']
        else:
            box = item
            
        x1, y1, x2, y2 = box
        
        # Calculate center point of the bounding box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        
        # Determine which cell the center point falls into
        col = int(cx // cell_w)
        row = int(cy // cell_h)
        
        # Clip coordinates to prevent index-out-of-bounds due to floating point precision
        col = max(0, min(col, grid_cols - 1))
        row = max(0, min(row, grid_rows - 1))
        
        grid[row, col] += 1.0
        
    return grid


def should_use_csrnet(yolo_count, yolo_boxes, threshold=50, overlap_threshold=0.3):
    """
    Tunable heuristic to decide whether the system should switch from YOLO to CSRNet.
    
    Individual detection (YOLO) works best for low-density/sparse crowds. 
    However, when crowds become dense, people block (occlude) each other, and YOLO's 
    accuracy drops. Density estimation (CSRNet) is fully convolutional and estimates 
    counts by scanning crowd patterns without needing to detect individual borders.
    
    We switch to CSRNet if:
      - YOLO count is >= a critical threshold (default: 50 people, based on empirical 
        YOLO-vs-CSRNet divergence analysis).
      - OR, if a significant ratio of YOLO boxes overlap heavily (high IoU), which indicates
        severe crowd congestion and potential detection failures due to occlusion.
        
    Args:
        yolo_count (int): Total count of persons currently detected by YOLO.
        yolo_boxes (list): Bounding boxes detected by YOLO.
        threshold (int): Count limit above which we automatically switch to CSRNet (default: 50).
                         This is determined by empirical YOLO-vs-CSRNet divergence analysis,
                         where disagreement was lowest in the 30-50 range and climbed
                         consistently above it (e.g. 58.3% on 50-100 range with 32 samples,
                         and 67.5% on 100+ range with 50 samples). The 10-20 bucket showed 
                         high disagreement (55.7%) but had only 6 samples, so it was excluded.
                         Note: This threshold may need further tuning once more real-world 
                         footage is available.
        overlap_threshold (float): IoU threshold (0.0 to 1.0) above which two boxes are 
                                   considered to be overlapping/occluded (default: 0.3).
                                   
    Returns:
        bool: True if CSRNet should be used, False if YOLO should be used.
    """
    # 1. Simple capacity check: if YOLO sees 50+ people (or current threshold limit),
    # switch to density estimation. This threshold is validated by side-by-side empirical testing.
    if yolo_count >= threshold:
        return True
        
    # Extract coordinates to standard format [[x1, y1, x2, y2], ...]
    boxes = []
    for item in yolo_boxes:
        if isinstance(item, dict) and 'bbox' in item:
            boxes.append(item['bbox'])
        else:
            boxes.append(item)
            
    n_boxes = len(boxes)
    if n_boxes <= 1:
        return False  # Overlaps are impossible with 0 or 1 person
        
    # Track the indices of boxes that overlap with at least one other box
    overlapping_indices = set()
    
    # Compare every box with every other box
    for i in range(n_boxes):
        for j in range(i + 1, n_boxes):
            box1 = boxes[i]
            box2 = boxes[j]
            
            # Unpack coordinates
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            # Compute intersection box coordinates
            xi1 = max(x1_1, x1_2)
            yi1 = max(y1_1, y1_2)
            xi2 = min(x2_1, x2_2)
            yi2 = min(y2_1, y2_2)
            
            # Intersection area
            inter_w = max(0, xi2 - xi1)
            inter_h = max(0, yi2 - yi1)
            inter_area = inter_w * inter_h
            
            # Union area
            box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
            box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = box1_area + box2_area - inter_area
            
            if union_area > 0:
                iou = inter_area / union_area
                # If IoU is high, these two detections are overlapping heavily
                if iou > overlap_threshold:
                    overlapping_indices.add(i)
                    overlapping_indices.add(j)
                    
    # Calculate what percentage of our detections are overlapping
    # TUNING TIP: If 20% or more of the crowd is overlapping, we assume detection is breaking down
    overlap_ratio_threshold = 0.20
    overlap_ratio = len(overlapping_indices) / n_boxes
    
    if overlap_ratio >= overlap_ratio_threshold:
        return True
        
    return False


def get_smoothed_model_decision(yolo_count, yolo_boxes, decision_history, window_size=10):
    """
    Computes a smoothed model selection decision (YOLO vs CSRNet) using a majority vote
    over a rolling window of recent raw decisions.
    
    Safety Design Trade-off Note:
    - window_size=10 is a starting point and may need tuning.
    - A larger window size improves stability by ignoring transient spikes, but slows down
      response times to genuinely fast-developing dangerous crowd density changes (critical safety risk).
    - A smaller window size enables faster response to danger but increases visible switching flicker.
    
    Args:
        yolo_count (int): Bounding box count from YOLO.
        yolo_boxes (list): Bounding box coordinates.
        decision_history (list): Rolling queue/history of past boolean decisions (True = CSRNet).
        window_size (int): Size of the rolling window.
        
    Returns:
        bool: Smoothed decision (True to use CSRNet, False to use YOLO).
    """
    # 1. Get raw per-frame decision
    raw_decision = should_use_csrnet(yolo_count, yolo_boxes)
    
    # 2. Append to rolling history
    decision_history.append(raw_decision)
    while len(decision_history) > window_size:
        decision_history.pop(0)
        
    # 3. Compute majority vote (return True if >= 50% of the window voted True)
    true_votes = sum(decision_history)
    return true_votes >= len(decision_history) / 2.0


def get_smoothed_count(current_count, count_history, window_size=5):
    """
    Computes a simple moving average of crowd counts to prevent visual jitter on the dashboard.
    
    Args:
        current_count (float): The current frame's estimated count.
        count_history (list): Rolling queue of past counts.
        window_size (int): Size of the moving average window.
        
    Returns:
        float: Smoothed crowd count.
    """
    count_history.append(current_count)
    while len(count_history) > window_size:
        count_history.pop(0)
        
    return float(np.mean(count_history))


def get_zone_risk_scores(zone_grid, thresholds):
    """
    Evaluates a per-zone grid of crowd counts/densities (from either YOLO or CSRNet)
    and maps each cell to a risk score: "safe", "caution", or "danger".
    
    Args:
        zone_grid (np.ndarray): 2D array of counts/densities per cell.
        thresholds (dict): Threshold values for classifying risk levels. Expected format:
                           {
                               'caution': float, # Threshold count/density for caution level
                               'danger': float   # Threshold count/density for danger level
                           }
                           
    Returns:
        np.ndarray: A 2D numpy array of strings containing "safe", "caution", or "danger"
                    corresponding to the risk level of each grid zone.
    """
    rows, cols = zone_grid.shape
    risk_grid = np.empty((rows, cols), dtype=object)
    
    # Fetch threshold values with safe fallback defaults
    caution_limit = thresholds.get('caution', 3.0)
    danger_limit = thresholds.get('danger', 6.0)
    
    for r in range(rows):
        for c in range(cols):
            val = zone_grid[r, c]
            if val >= danger_limit:
                risk_grid[r, c] = "danger"
            elif val >= caution_limit:
                risk_grid[r, c] = "caution"
            else:
                risk_grid[r, c] = "safe"
                
    return risk_grid

