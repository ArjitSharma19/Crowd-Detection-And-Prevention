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
