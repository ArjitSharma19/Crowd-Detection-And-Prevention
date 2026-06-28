import time
from datetime import datetime

class AlertManager:
    def __init__(self, max_capacity=15, caution_at=70, density_limit=5.0, trigger_delay_seconds=3.0):
        """
        Initializes the Alert Manager.
        
        Args:
            max_capacity: int, max allowed total count of people in frame.
            caution_at: int, capacity percentage that triggers caution level (default 70).
            density_limit: float, threshold for peak point density.
            trigger_delay_seconds: float, duration threshold before warning turns into critical alert.
        """
        self.max_capacity = max_capacity
        self.caution_at = caution_at
        self.density_limit = density_limit
        self.trigger_delay_seconds = trigger_delay_seconds
        
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
                    log_msg = f"Warning: Crowd limits approached (capacity reached {int(capacity_percentage)}%)"
                elif self.current_status == "CRITICAL":
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
                        log_msg = f"Warning: Crowd limits approached (capacity reached {int(capacity_percentage)}%)"
                    elif self.current_status == "CRITICAL":
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
            msg = f"Warning — capacity at {int(capacity_percentage)}%"
        else:  # CRITICAL
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
