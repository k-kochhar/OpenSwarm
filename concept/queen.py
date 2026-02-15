"""
Queen central intelligence - logging and metrics.
"""

class Queen:
    """Central intelligence logging and metrics system."""

    def __init__(self):
        self.logs = []
        self.max_logs = 12
        self.current_mode = "INITIALIZING"
        self.pushing_robots = 0
        self.push_speed_percent = 0
        self.fire_status = "NONE"

    def log(self, message, sim_time):
        """Add a timestamped log message."""
        timestamp = f"T+{sim_time:06.2f}s"
        full_message = f"[{timestamp}] {message}"
        self.logs.append(full_message)

        if len(self.logs) > self.max_logs:
            self.logs.pop(0)

    def update_metrics(self, mode, pushing_count, speed_percent, fire_status):
        """Update display metrics."""
        self.current_mode = mode
        self.pushing_robots = pushing_count
        self.push_speed_percent = speed_percent
        self.fire_status = fire_status
