import math
import numpy as np

def update_track_history(track_id, center_point, timestamp, history_dict, max_history=10):
    """
    Appends the center point and timestamp for a tracked person.
    Keep history length bounded to max_history.
    
    Args:
        track_id (int): Unique tracking identifier.
        center_point (tuple): (x, y) coordinates of the detection center.
        timestamp (float): Current system time in seconds.
        history_dict (dict): Shared dictionary mapping track_id -> list of trajectory entries.
        max_history (int): Maximum length of trajectory memory per person.
    """
    if track_id not in history_dict:
        history_dict[track_id] = []
        
    history_dict[track_id].append({
        'center': center_point,
        'time': timestamp
    })
    
    # Prune history to limit size
    if len(history_dict[track_id]) > max_history:
        history_dict[track_id].pop(0)

def calculate_velocity(track_id, history_dict):
    """
    Computes velocity (speed and angle of direction) based on historical trajectory.
    Uses oldest and newest point in trajectory history to smooth short-term jitters.
    
    Args:
        track_id (int): Unique tracking identifier.
        history_dict (dict): Dictionary mapping track_id -> list of trajectory entries.
        
    Returns:
        tuple: (speed, direction_angle_degrees) or None if history is insufficient (< 2 points) or dt <= 0.
               Angle is in range [0, 360) where 0 is right, 90 is down (standard image coordinates).
    """
    # Require at least 5 frames of history to avoid noisy estimates from short tracks
    if track_id not in history_dict or len(history_dict[track_id]) < 5:
        return None
        
    history = history_dict[track_id]
    p_old = history[0]
    p_new = history[-1]
    
    dt = p_new['time'] - p_old['time']
    if dt <= 0:
        return None
        
    dx = p_new['center'][0] - p_old['center'][0]
    dy = p_new['center'][1] - p_old['center'][1]
    
    # Calculate pixel distance
    distance = math.sqrt(dx**2 + dy**2)
    speed = distance / dt
    
    # Calculate direction angle in degrees [0, 360)
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad) % 360
    
    return speed, angle_deg

def get_zone_velocity_stats(tracked_objects, history_dict, frame_shape, grid_rows=3, grid_cols=3):
    """
    Aggregates velocity data into spatial zones (3x3 grid by default).
    Computes per-zone average speed and circular direction variance.
    
    Circular variance represents direction coherence:
      - 0.0: Perfect cohesion (everyone moving in the exact same direction)
      - 1.0: Random distribution (movement directions are chaotic/canceling out)
      
    Args:
        tracked_objects (list): List of currently tracked object dicts.
        history_dict (dict): History mapping track_id -> trajectory lists.
        frame_shape (tuple): Frame shape (height, width, channels).
        grid_rows (int): Grid rows.
        grid_cols (int): Grid columns.
        
    Returns:
        tuple: (avg_speeds, dir_variances)
               avg_speeds: np.ndarray (grid_rows, grid_cols)
               dir_variances: np.ndarray (grid_rows, grid_cols)
    """
    h, w = frame_shape[:2]
    cell_h = h / grid_rows
    cell_w = w / grid_cols
    
    # Temporary lists for speed and angles per cell
    speeds_grid = [[[] for _ in range(grid_cols)] for _ in range(grid_rows)]
    angles_grid = [[[] for _ in range(grid_cols)] for _ in range(grid_rows)]
    
    for obj in tracked_objects:
        track_id = obj['track_id']
        cx, cy = obj['center_point']
        
        # Determine cell coordinates
        row = int(cy // cell_h)
        col = int(cx // cell_w)
        
        # Clamp to prevent boundary issues
        row = max(0, min(row, grid_rows - 1))
        col = max(0, min(col, grid_cols - 1))
        
        velocity = calculate_velocity(track_id, history_dict)
        if velocity is not None:
            speed, angle_deg = velocity
            speeds_grid[row][col].append(speed)
            angles_grid[row][col].append(math.radians(angle_deg))
            
    # Output metrics arrays
    avg_speeds = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    dir_variances = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            cell_speeds = speeds_grid[r][c]
            cell_angles = angles_grid[r][c]
            
            if cell_speeds:
                avg_speeds[r][c] = float(np.mean(cell_speeds))
            else:
                avg_speeds[r][c] = -1.0  # -1.0 signals insufficient history / empty cell
                
            if cell_angles:
                # Compute circular mean vector length (R)
                sum_cos = sum(math.cos(a) for a in cell_angles)
                sum_sin = sum(math.sin(a) for a in cell_angles)
                N = len(cell_angles)
                R = math.sqrt(sum_cos**2 + sum_sin**2) / N
                
                # Circular variance is 1 - R
                dir_variances[r][c] = float(1.0 - R)
            else:
                dir_variances[r][c] = -1.0  # -1.0 signals insufficient history / empty cell
                
    return avg_speeds, dir_variances
