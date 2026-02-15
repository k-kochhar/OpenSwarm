"""
Math utilities for the simulation.
"""
import math

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def angle_difference(target, current):
    """Compute shortest angle difference from current to target."""
    diff = target - current
    return normalize_angle(diff)

def distance(x1, y1, x2, y2):
    """Compute Euclidean distance."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def clamp(value, min_val, max_val):
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))

def lerp(a, b, t):
    """Linear interpolation."""
    return a + (b - a) * t

class Vec2:
    """Simple 2D vector."""
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)

    def length(self):
        return math.sqrt(self.x**2 + self.y**2)

    def normalized(self):
        l = self.length()
        if l > 0:
            return Vec2(self.x / l, self.y / l)
        return Vec2(0, 0)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def to_tuple(self):
        return (self.x, self.y)
