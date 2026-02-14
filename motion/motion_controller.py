import math


def parse_command(command_string):
    """
    Parse a command string and return left and right wheel speeds.
    
    Supported formats:
    - Simple: "F", "L", "R", "S" (Forward, Left, Right, Stop)
    - Detailed: "L:50;R:40" (Left wheel at 50, Right wheel at 40)
    
    Returns:
        tuple: (left_speed, right_speed) in range [-100, 100]
    """
    command_string = command_string.strip().upper()
    
    # Check for detailed format (L:50;R:40)
    if ':' in command_string and ';' in command_string:
        parts = command_string.split(';')
        left_speed = 0
        right_speed = 0
        
        for part in parts:
            if ':' in part:
                key, value = part.split(':')
                key = key.strip()
                value = int(value.strip())
                
                if key == 'L':
                    left_speed = value
                elif key == 'R':
                    right_speed = value
        
        return left_speed, right_speed
    
    # Simple format
    if command_string == 'F':
        return 100, 100
    elif command_string == 'L':
        return -100, 100
    elif command_string == 'R':
        return 100, -100
    elif command_string == 'S':
        return 0, 0
    else:
        return 0, 0


def generate_command(left_speed, right_speed):
    """
    Generate a command string from wheel speeds.
    
    Args:
        left_speed: Left wheel speed [-100, 100]
        right_speed: Right wheel speed [-100, 100]
    
    Returns:
        str: Command string in format "L:50;R:40"
    """
    return f"L:{int(left_speed)};R:{int(right_speed)}"


class AutonomousController:
    """Autonomous controller that follows a path"""
    
    def __init__(self, path, robot_x, robot_y, waypoint_threshold=15, angle_threshold=0.1):
        """
        Args:
            path: List of (x, y) waypoint tuples
            robot_x: Initial robot x position
            robot_y: Initial robot y position
            waypoint_threshold: Distance at which a waypoint is considered reached
            angle_threshold: Angle error threshold (radians) to start moving forward
        """
        self.path = path
        self.waypoint_threshold = waypoint_threshold
        self.angle_threshold = angle_threshold
        self.finished = False
        
        # Find closest waypoint to robot's initial position
        if path:
            closest_idx, closest_dist = self._find_closest_waypoint_with_distance(robot_x, robot_y)
            # If already at the closest waypoint, target the next one
            if closest_dist < self.waypoint_threshold:
                self.current_waypoint_idx = (closest_idx + 1) % len(path)
            else:
                # Otherwise, navigate to the closest waypoint first
                self.current_waypoint_idx = closest_idx
        else:
            self.current_waypoint_idx = 0
        
    def compute_wheel_speeds(self, robot):
        """Rotate to face target, then drive straight"""
        if self.finished or not self.path:
            return 0, 0
        
        # Get current waypoint
        target_x, target_y = self.path[self.current_waypoint_idx]
        
        # Calculate distance to waypoint
        dx = target_x - robot.x
        dy = target_y - robot.y
        distance = math.sqrt(dx**2 + dy**2)
        
        # Check if waypoint reached
        if distance < self.waypoint_threshold:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.path):
                self.finished = True
                return 0, 0
            # Get next waypoint
            target_x, target_y = self.path[self.current_waypoint_idx]
            dx = target_x - robot.x
            dy = target_y - robot.y
        
        # Calculate desired angle to target
        target_angle = math.atan2(dy, dx)
        
        # Calculate angle error (normalize to -pi to pi)
        angle_error = target_angle - robot.angle
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi
        
        # Two-state control: rotate then drive
        if abs(angle_error) > self.angle_threshold:
            # Not aligned: rotate in place
            if angle_error > 0:
                # Turn left
                return -50, 50
            else:
                # Turn right
                return 50, -50
        else:
            # Aligned: drive straight forward
            return 100, 100
    
    def compute_command(self, robot):
        """Generate command string for robot navigation"""
        left_speed, right_speed = self.compute_wheel_speeds(robot)
        return generate_command(left_speed, right_speed)
    
    def get_current_waypoint(self):
        """Get the current target waypoint"""
        if self.finished or self.current_waypoint_idx >= len(self.path):
            return None
        return self.path[self.current_waypoint_idx]
    
    def _find_closest_waypoint_with_distance(self, robot_x, robot_y):
        """Find the index and distance of the closest waypoint to the given position"""
        min_dist = float('inf')
        closest_idx = 0
        
        for i, (wx, wy) in enumerate(self.path):
            dist = math.sqrt((wx - robot_x)**2 + (wy - robot_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        return closest_idx, min_dist
    
    def reset(self, robot_x, robot_y):
        """Reset controller to closest waypoint from given position"""
        if self.path:
            closest_idx, closest_dist = self._find_closest_waypoint_with_distance(robot_x, robot_y)
            # If already at the closest waypoint, target the next one
            if closest_dist < self.waypoint_threshold:
                self.current_waypoint_idx = (closest_idx + 1) % len(self.path)
            else:
                # Otherwise, navigate to the closest waypoint first
                self.current_waypoint_idx = closest_idx
        else:
            self.current_waypoint_idx = 0
        self.finished = False
