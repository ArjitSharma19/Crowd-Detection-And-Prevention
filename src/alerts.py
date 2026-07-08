import time
from datetime import datetime

class AlertManager:
    def __init__(self, max_capacity=15, caution_at=70, density_limit=5.0, trigger_delay_seconds=3.0, speed_limit=80.0, turbulence_limit=0.75):
        """
        Initializes the Alert Manager.
        
        Args:
            max_capacity: int, max allowed total count of people in frame.
            caution_at: int, capacity percentage that triggers caution level (default 70).
            density_limit: float, threshold for peak point density.
            trigger_delay_seconds: float, duration threshold before warning turns into critical alert.
            speed_limit: float, threshold for average speed in a cell to detect panic/running.
            turbulence_limit: float, threshold for circular direction variance to detect chaos.
        """
        self.max_capacity = max_capacity
        self.caution_at = caution_at
        self.density_limit = density_limit
        self.trigger_delay_seconds = trigger_delay_seconds
        self.speed_limit = speed_limit
        self.turbulence_limit = turbulence_limit
        
        # State tracking
        self.current_status = "NORMAL"  # "NORMAL", "WARNING", "CRITICAL"
        self.pending_status = None
        self.status_transition_since = None
        
        # Log of recent alerts
        self.alert_history = []
        self.max_history_len = 50

    def get_risk_tier(self, capacity_percentage, peak_density=0.0):
        """
        Determines the risk tier ("danger", "caution", "safe") for a given capacity percentage.
        Ensure any capacity_percentage >= 100 always returns "danger", with no upper bound.
        """
        if capacity_percentage >= 100:
            return "danger"
        elif capacity_percentage >= self.caution_at:
            return "caution"
        else:
            return "safe"

    def update(self, current_count, peak_density, avg_speeds=None, dir_variances=None, zone_grid=None):
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
        max_cap = float(self.max_capacity) if self.max_capacity > 0 else 1.0
        capacity_percentage = (current_count / max_cap) * 100
        
        # 1. Determine raw measured tier using helper
        raw_tier_name = self.get_risk_tier(capacity_percentage, peak_density)
        if raw_tier_name == "danger":
            raw_tier = "CRITICAL"
        elif raw_tier_name == "caution":
            raw_tier = "WARNING"
        else:
            raw_tier = "NORMAL"
            
        # Check motion anomalies
        velocity_alert = False
        turbulence_alert = False
        anomaly_details = ""
        
        if avg_speeds is not None and zone_grid is not None:
            import numpy as np
            speeds_arr = np.array(avg_speeds)
            grid_arr = np.array(zone_grid)
            rows, cols = speeds_arr.shape
            for r in range(rows):
                for c in range(cols):
                    # Speed anomaly: if there's someone in the cell and speed exceeds limit
                    if grid_arr[r, c] >= 1.0 and speeds_arr[r, c] > self.speed_limit:
                        velocity_alert = True
                        anomaly_details = f"Sudden crowd rush (avg speed {speeds_arr[r,c]:.1f} px/s in cell [{r+1},{c+1}])"
                        break
                if velocity_alert:
                    break
                    
        if not velocity_alert and dir_variances is not None and zone_grid is not None:
            import numpy as np
            vars_arr = np.array(dir_variances)
            grid_arr = np.array(zone_grid)
            rows, cols = vars_arr.shape
            for r in range(rows):
                for c in range(cols):
                    # Turbulence anomaly: if there are >= 3 people and variance exceeds limit
                    if grid_arr[r, c] >= 3.0 and vars_arr[r, c] > self.turbulence_limit:
                        turbulence_alert = True
                        anomaly_details = f"Chaotic crowd flow (turbulence {vars_arr[r,c]:.2f} in cell [{r+1},{c+1}])"
                        break
                if turbulence_alert:
                    break
                    
        # Escalate raw tier based on motion anomalies
        if velocity_alert:
            raw_tier = "CRITICAL"
        elif turbulence_alert and raw_tier == "NORMAL":
            raw_tier = "WARNING"
            
        # 2. Track state transition with trigger_delay_seconds
        if raw_tier != self.current_status:
            if self.trigger_delay_seconds <= 0.0:
                # Transition immediately!
                old_status = self.current_status
                self.current_status = raw_tier
                self.pending_status = None
                self.status_transition_since = None
                
                # Formulate log message for history
                if self.current_status == "WARNING":
                    if turbulence_alert:
                        log_msg = f"Warning: {anomaly_details}"
                    else:
                        log_msg = f"Warning: Crowd limits approached (capacity reached {int(capacity_percentage)}%)"
                elif self.current_status == "CRITICAL":
                    if velocity_alert:
                        log_msg = f"CRITICAL: {anomaly_details}"
                    else:
                        log_msg = f"CRITICAL: capacity exceeded ({round(current_count)}/{self.max_capacity})"
                else:
                    log_msg = "Crowd levels returned to normal."
                    
                self._add_to_history(self.current_status, log_msg)
            # If we are already tracking this transition
            elif self.pending_status == raw_tier:
                elapsed = now - self.status_transition_since
                if elapsed >= self.trigger_delay_seconds:
                    # Transition sustained! Log it and update status
                    old_status = self.current_status
                    self.current_status = raw_tier
                    self.pending_status = None
                    self.status_transition_since = None
                    
                    # Formulate log message for history
                    if self.current_status == "WARNING":
                        if turbulence_alert:
                            log_msg = f"Warning: {anomaly_details}"
                        else:
                            log_msg = f"Warning: Crowd limits approached (capacity reached {int(capacity_percentage)}%)"
                    elif self.current_status == "CRITICAL":
                        if velocity_alert:
                            log_msg = f"CRITICAL: {anomaly_details}"
                        else:
                            log_msg = f"CRITICAL: capacity exceeded ({round(current_count)}/{self.max_capacity})"
                    else:
                        log_msg = "Crowd levels returned to normal."
                        
                    self._add_to_history(self.current_status, log_msg)
            else:
                # Start tracking new transition
                self.pending_status = raw_tier
                self.status_transition_since = now
        else:
            # Measured tier matches current status; cancel pending transition
            self.pending_status = None
            self.status_transition_since = None

        # 3. Formulate status message based on current status (which is delayed/sustained)
        if self.current_status == "NORMAL":
            msg = "Crowd levels within safe parameters."
        elif self.current_status == "WARNING":
            if turbulence_alert or (avg_speeds is not None and turbulence_alert): # Check dynamically if sustained
                msg = f"Warning — {anomaly_details}"
            else:
                msg = f"Warning — capacity at {int(capacity_percentage)}%"
        else:  # CRITICAL
            if velocity_alert or (avg_speeds is not None and velocity_alert): # Check dynamically if sustained
                msg = f"CRITICAL — {anomaly_details}"
            else:
                msg = f"CRITICAL: Capacity exceeded ({round(current_count)}/{self.max_capacity})"
            
        return {
            'status': self.current_status,
            'message': msg,
            'new_alert': False,
            'durations': {
                'capacity_seconds': 0.0,
                'density_seconds': 0.0
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
        self.pending_status = None
        self.status_transition_since = None
        self.alert_history = []
