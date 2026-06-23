import time
from datetime import datetime

class AlertManager:
    def __init__(self, max_capacity=15, density_limit=5.0, trigger_delay_seconds=3.0):
        """
        Initializes the Alert Manager.
        
        Args:
            max_capacity: int, max allowed total count of people in frame.
            density_limit: float, threshold for peak point density.
            trigger_delay_seconds: float, duration threshold before warning turns into critical alert.
        """
        self.max_capacity = max_capacity
        self.density_limit = density_limit
        self.trigger_delay_seconds = trigger_delay_seconds
        
        # State tracking
        self.current_status = "NORMAL"  # "NORMAL", "WARNING", "CRITICAL"
        self.capacity_exceeded_since = None
        self.density_exceeded_since = None
        
        # Log of recent alerts
        self.alert_history = []
        self.max_history_len = 50

    def update(self, current_count, peak_density):
        """
        Updates status based on new readings.
        
        Returns:
            dict: {
                'status': str ("NORMAL", "WARNING", "CRITICAL"),
                'message': str,
                'new_alert': bool
            }
        """
        now = time.time()
        exceeded_capacity = current_count > self.max_capacity
        exceeded_density = peak_density > self.density_limit
        
        # 1. Capacity state tracking
        if exceeded_capacity:
            if self.capacity_exceeded_since is None:
                self.capacity_exceeded_since = now
        else:
            self.capacity_exceeded_since = None
            
        # 2. Density state tracking
        if exceeded_density:
            if self.density_exceeded_since is None:
                self.density_exceeded_since = now
        else:
            self.density_exceeded_since = None
            
        # 3. Determine status
        new_status = "NORMAL"
        msg = "Crowd levels within safe parameters."
        
        # Calculate how long thresholds have been breached
        cap_duration = (now - self.capacity_exceeded_since) if self.capacity_exceeded_since else 0.0
        dens_duration = (now - self.density_exceeded_since) if self.density_exceeded_since else 0.0
        
        max_duration = max(cap_duration, dens_duration)
        
        if exceeded_capacity or exceeded_density:
            if max_duration >= self.trigger_delay_seconds:
                new_status = "CRITICAL"
                reasons = []
                if exceeded_capacity:
                    reasons.append(f"Capacity exceeded ({current_count}/{self.max_capacity})")
                if exceeded_density:
                    reasons.append(f"High local density detected ({peak_density:.1f}/{self.density_limit})")
                msg = f"CRITICAL: " + " & ".join(reasons) + " for over " + f"{int(max_duration)}s!"
            else:
                new_status = "WARNING"
                msg = f"WARNING: Crowd limits approached. Verifying sustained state..."
                
        # 4. Check for state change
        new_alert = False
        if new_status != self.current_status:
            self.current_status = new_status
            if new_status != "NORMAL":
                new_alert = True
                self._add_to_history(new_status, msg)
                
        return {
            'status': self.current_status,
            'message': msg,
            'new_alert': new_alert,
            'durations': {
                'capacity_seconds': cap_duration,
                'density_seconds': dens_duration
            }
        }

    def _add_to_history(self, status, message):
        alert_item = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'status': status,
            'message': message
        }
        self.alert_history.insert(0, alert_item)
        if len(self.alert_history) > self.max_history_len:
            self.alert_history.pop()

    def get_history(self):
        return self.alert_history
        
    def reset(self):
        self.current_status = "NORMAL"
        self.capacity_exceeded_since = None
        self.density_exceeded_since = None
        self.alert_history = []
