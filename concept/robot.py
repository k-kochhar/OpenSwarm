"""
Robot kinematics and collision handling.
"""
import math
from utils import Vec2, normalize_angle, angle_difference, distance

class Robot:
    """A firefighting robot with deterministic kinematics."""

    # States
    STATE_IDLE = "IDLE"
    STATE_MOVING_TO_DOCK = "MOVING_TO_DOCK"
    STATE_DOCKED = "DOCKED"
    STATE_PUSHING = "PUSHING"
    STATE_RESPONDING = "RESPONDING"
    STATE_SUPPRESSING = "SUPPRESSING"
    STATE_RETURNING = "RETURNING"
    STATE_COMPLETE = "COMPLETE"

    def __init__(self, name, label, color, spawn_x, spawn_y):
        self.name = name
        self.label = label
        self.color = color

        # Physics
        self.radius = 18  # Fixed radius
        self.pos = Vec2(spawn_x, spawn_y)
        self.theta = 0.0  # Orientation in radians
        self.vel = Vec2(0, 0)

        # Motion parameters
        self.max_speed = 140.0  # px/sec
        self.max_turn_rate = math.radians(240)  # 240 deg/sec

        # State
        self.state = self.STATE_IDLE
        self.target_pos = None
        self.dock_pos = None
        self.dock_angle = None

        # Docking validation
        self.dock_stable_timer = 0.0

    def set_target(self, x, y):
        """Set navigation target."""
        self.target_pos = Vec2(x, y)

    def set_dock(self, x, y, facing_angle):
        """Set docking position and orientation."""
        self.dock_pos = Vec2(x, y)
        self.dock_angle = facing_angle

    def update(self, dt, world, other_robots):
        """Update robot physics."""
        if self.state == self.STATE_COMPLETE:
            self.vel = Vec2(0, 0)
            return

        # Navigate based on state and targets
        # Priority: explicit target_pos > dock navigation > stop
        if self.target_pos:
            self._navigate_to_target(dt)
        elif self.dock_pos and self.state in [self.STATE_MOVING_TO_DOCK, self.STATE_DOCKED, self.STATE_RETURNING, self.STATE_PUSHING]:
            self._navigate_to_dock(dt)
        else:
            # No target - stop
            self.vel = Vec2(0, 0)

        # Apply velocity
        self.pos = self.pos + self.vel * dt

        # Resolve collisions
        self._resolve_collisions(other_robots, world)

    def _navigate_to_target(self, dt):
        """Navigate to target position."""
        to_target = self.target_pos - self.pos
        dist = to_target.length()

        if dist < 5.0:  # Increased stopping threshold
            self.vel = Vec2(0, 0)
            return

        # Desired heading
        desired_theta = math.atan2(to_target.y, to_target.x)

        # Gradual turn
        angle_diff = angle_difference(desired_theta, self.theta)
        max_turn = self.max_turn_rate * dt

        if abs(angle_diff) > max_turn:
            self.theta += max_turn if angle_diff > 0 else -max_turn
        else:
            self.theta = desired_theta

        self.theta = normalize_angle(self.theta)

        # Move forward with speed based on distance (slow down near target)
        speed_factor = min(1.0, dist / 50.0)  # Slow down within 50px
        self.vel = Vec2(math.cos(self.theta), math.sin(self.theta)) * (self.max_speed * speed_factor)

    def _navigate_to_dock(self, dt):
        """Navigate to and align with dock position."""
        if not self.dock_pos:
            self.vel = Vec2(0, 0)
            return

        to_dock = self.dock_pos - self.pos
        dist = to_dock.length()

        # Position error
        if dist > 5.0:
            # Move toward dock
            desired_theta = math.atan2(to_dock.y, to_dock.x)
            angle_diff = angle_difference(desired_theta, self.theta)
            max_turn = self.max_turn_rate * dt

            if abs(angle_diff) > max_turn:
                self.theta += max_turn if angle_diff > 0 else -max_turn
            else:
                self.theta = desired_theta

            self.theta = normalize_angle(self.theta)

            # Slow down as approaching dock
            speed_factor = min(1.0, dist / 30.0)
            self.vel = Vec2(math.cos(self.theta), math.sin(self.theta)) * (self.max_speed * speed_factor)
        else:
            # Close to dock - align orientation
            angle_diff = angle_difference(self.dock_angle, self.theta)
            max_turn = self.max_turn_rate * dt

            if abs(angle_diff) > math.radians(5):  # Still aligning
                if abs(angle_diff) > max_turn:
                    self.theta += max_turn if angle_diff > 0 else -max_turn
                else:
                    self.theta = self.dock_angle
                self.theta = normalize_angle(self.theta)

            # Stop moving
            self.vel = Vec2(0, 0)

    def is_docked_correctly(self):
        """Check if robot is at dock position and aligned."""
        if not self.dock_pos:
            return False

        pos_error = (self.pos - self.dock_pos).length()
        angle_error = abs(angle_difference(self.dock_angle, self.theta))

        return pos_error < 5.0 and angle_error < math.radians(10)

    def _resolve_collisions(self, other_robots, world):
        """Resolve robot-robot and robot-item collisions."""
        # Robot-robot collisions
        for other in other_robots:
            if other is self:
                continue

            dist = (self.pos - other.pos).length()
            min_dist = self.radius + other.radius

            if dist < min_dist and dist > 0.1:
                # Separate along collision normal
                overlap = min_dist - dist
                normal = (self.pos - other.pos).normalized()
                self.pos = self.pos + normal * (overlap / 2)

        # Robot-item collision (square)
        half_size = world.item_size / 2

        # Find closest point on square to robot
        closest_x = max(world.item_pos.x - half_size,
                       min(self.pos.x, world.item_pos.x + half_size))
        closest_y = max(world.item_pos.y - half_size,
                       min(self.pos.y, world.item_pos.y + half_size))

        dist = distance(self.pos.x, self.pos.y, closest_x, closest_y)

        if dist < self.radius:
            # Push robot out
            if dist > 0.1:
                normal = Vec2(self.pos.x - closest_x, self.pos.y - closest_y).normalized()
                self.pos = self.pos + normal * (self.radius - dist)
