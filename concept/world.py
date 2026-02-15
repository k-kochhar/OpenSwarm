"""
World state management.
"""
from utils import Vec2

class World:
    """Contains all world entities and state."""

    def __init__(self, map_width=900, map_height=800):
        self.map_width = map_width
        self.map_height = map_height

        # Item (70x70 square)
        self.item_size = 70
        self.item_pos = Vec2(300, map_height // 2)

        # Destination (Location Bravo) - 110x110 region
        self.bravo_size = 110
        self.bravo_pos = Vec2(map_width - 150, map_height // 2)

        # Fire (bottom of map)
        self.fire_pos = Vec2(500, 650)
        self.fire_active = False
        self.fire_intensity = 0.0  # 0 to 1
        self.fire_radius = 40  # Suppression radius

    def is_item_at_bravo(self):
        """Check if item center has reached destination center."""
        # Calculate distance between item center and bravo center
        dist = (self.item_pos - self.bravo_pos).length()
        # Item must reach within 10px of the destination center
        return dist < 10.0
