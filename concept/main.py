"""
Main simulation loop with Pygame rendering and video export.
"""
import pygame
import math
import os
from datetime import datetime

from world import World
from robot import Robot
from queen import Queen
from state_machine import SimulationStateMachine
from utils import Vec2

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
MAP_WIDTH = 900
MAP_HEIGHT = 800
PANEL_WIDTH = 300
FPS = 60

# Colors
COLOR_BG = (245, 245, 250)
COLOR_GRID = (220, 220, 230)
COLOR_BORDER = (80, 80, 80)
COLOR_BLUE = (50, 120, 200)
COLOR_GREEN = (80, 180, 100)
COLOR_ORANGE = (230, 120, 50)
COLOR_ITEM = (120, 120, 130)
COLOR_BRAVO = (100, 200, 100)
COLOR_FIRE = (255, 80, 20)
COLOR_FIRE_GLOW = (255, 160, 40)
COLOR_PANEL_BG = (25, 25, 30)
COLOR_PANEL_TEXT = (200, 200, 200)
COLOR_PANEL_HEADER = (100, 200, 255)

class Simulation:
    """Main simulation class."""

    def __init__(self, export_frames=False):
        pygame.init()

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Queen Central Intelligence - Multi-Robot Coordination")

        self.clock = pygame.time.Clock()
        self.running = True
        self.export_frames = export_frames
        self.frame_count = 0

        # Simulation time
        self.sim_time = 0.0

        # Initialize world
        self.world = World(MAP_WIDTH, MAP_HEIGHT)

        # Initialize robots at different spawn points
        self.robots = [
            Robot("Alpha", "A", COLOR_BLUE, 200, 100),      # Top of map
            Robot("Beta", "B", COLOR_GREEN, 700, 150),      # Top right of map
            Robot("Charlie", "C", COLOR_ORANGE, 200, 700)   # Bottom of map
        ]

        # Initialize Queen
        self.queen = Queen()

        # Initialize state machine
        self.state_machine = SimulationStateMachine(self.world, self.robots, self.queen)

        # Fonts
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

    def run(self):
        """Main simulation loop."""
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.sim_time += dt

            # Event handling
            self._handle_events()

            # Update simulation
            self._update(dt)

            # Render
            self._render()

            # Export frame if needed
            if self.export_frames:
                self._export_frame()

            # Auto-quit after mission complete + 2 seconds
            if self.state_machine.state == self.state_machine.STATE_COMPLETE:
                if self.state_machine.state_timer > 2.0:
                    self.running = False

        pygame.quit()

    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self._reset()

    def _reset(self):
        """Reset simulation."""
        self.sim_time = 0.0
        self.frame_count = 0

        self.world = World(MAP_WIDTH, MAP_HEIGHT)
        self.robots = [
            Robot("Alpha", "A", COLOR_BLUE, 200, 100),      # Top of map
            Robot("Beta", "B", COLOR_GREEN, 700, 150),      # Top right of map
            Robot("Charlie", "C", COLOR_ORANGE, 200, 700)   # Bottom of map
        ]
        self.queen = Queen()
        self.state_machine = SimulationStateMachine(self.world, self.robots, self.queen)

    def _update(self, dt):
        """Update simulation state."""
        # Update state machine
        self.state_machine.update(dt, self.sim_time)

        # Update robots
        for robot in self.robots:
            robot.update(dt, self.world, self.robots)

    def _render(self):
        """Render the simulation."""
        self.screen.fill(COLOR_BG)

        # Draw map area
        self._draw_map()

        # Draw control panel
        self._draw_panel()

        pygame.display.flip()

    def _draw_map(self):
        """Draw the overhead map."""
        # Grid
        grid_size = 50
        for x in range(0, MAP_WIDTH, grid_size):
            pygame.draw.line(self.screen, COLOR_GRID, (x, 0), (x, MAP_HEIGHT), 1)
        for y in range(0, MAP_HEIGHT, grid_size):
            pygame.draw.line(self.screen, COLOR_GRID, (0, y), (MAP_WIDTH, y), 1)

        # Destination (Location Bravo)
        bravo_rect = pygame.Rect(
            int(self.world.bravo_pos.x - self.world.bravo_size / 2),
            int(self.world.bravo_pos.y - self.world.bravo_size / 2),
            self.world.bravo_size,
            self.world.bravo_size
        )
        pygame.draw.rect(self.screen, COLOR_BRAVO, bravo_rect, 4)
        bravo_label = self.font_medium.render("BRAVO", True, (50, 150, 50))
        label_rect = bravo_label.get_rect(center=(int(self.world.bravo_pos.x),
                                                    int(self.world.bravo_pos.y - self.world.bravo_size / 2 - 20)))
        self.screen.blit(bravo_label, label_rect)

        # Item (Item A)
        item_rect = pygame.Rect(
            int(self.world.item_pos.x - self.world.item_size / 2),
            int(self.world.item_pos.y - self.world.item_size / 2),
            self.world.item_size,
            self.world.item_size
        )
        pygame.draw.rect(self.screen, COLOR_ITEM, item_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), item_rect, 3)
        item_label = self.font_medium.render("ITEM A", True, (255, 255, 255))
        label_rect = item_label.get_rect(center=(int(self.world.item_pos.x), int(self.world.item_pos.y)))
        self.screen.blit(item_label, label_rect)

        # Fire
        if self.world.fire_active or self.world.fire_intensity > 0:
            self._draw_fire()

        # Robots
        for robot in self.robots:
            self._draw_robot(robot)

        # Border
        pygame.draw.rect(self.screen, COLOR_BORDER, (0, 0, MAP_WIDTH, MAP_HEIGHT), 3)

        # Timestamp
        timestamp = f"T+{self.sim_time:06.2f}s"
        ts_text = self.font_small.render(timestamp, True, (100, 100, 100))
        self.screen.blit(ts_text, (10, 10))

        # Camera label
        cam_label = self.font_small.render("OVERHEAD CAM-01", True, (100, 100, 100))
        self.screen.blit(cam_label, (10, MAP_HEIGHT - 25))

    def _draw_robot(self, robot):
        """Draw a robot with ArUco marker and heading arrow."""
        # Robot body
        pygame.draw.circle(self.screen, robot.color,
                          (int(robot.pos.x), int(robot.pos.y)), robot.radius)
        pygame.draw.circle(self.screen, (0, 0, 0),
                          (int(robot.pos.x), int(robot.pos.y)), robot.radius, 2)

        # Heading arrow
        arrow_len = robot.radius * 1.5
        end_x = robot.pos.x + math.cos(robot.theta) * arrow_len
        end_y = robot.pos.y + math.sin(robot.theta) * arrow_len
        pygame.draw.line(self.screen, (0, 0, 0),
                        (robot.pos.x, robot.pos.y), (end_x, end_y), 3)

        # ArUco marker (label)
        label = self.font_large.render(robot.label, True, (255, 255, 255))
        label_rect = label.get_rect(center=(int(robot.pos.x), int(robot.pos.y)))
        self.screen.blit(label, label_rect)

    def _draw_fire(self):
        """Draw animated fire hazard."""
        # Pulsating effect
        pulse = math.sin(self.sim_time * 8) * 0.15 + 0.85
        intensity = self.world.fire_intensity

        if intensity > 0:
            # Glow
            glow_radius = int(50 * intensity * pulse)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*COLOR_FIRE_GLOW, 80),
                             (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (int(self.world.fire_pos.x - glow_radius),
                                         int(self.world.fire_pos.y - glow_radius)))

            # Core
            core_radius = int(30 * intensity * pulse)
            pygame.draw.circle(self.screen, COLOR_FIRE,
                             (int(self.world.fire_pos.x), int(self.world.fire_pos.y)),
                             core_radius)

            # Label
            fire_label = self.font_medium.render("FIRE", True, (255, 255, 255))
            label_rect = fire_label.get_rect(center=(int(self.world.fire_pos.x),
                                                       int(self.world.fire_pos.y)))
            self.screen.blit(fire_label, label_rect)

    def _draw_panel(self):
        """Draw Queen control panel."""
        panel_rect = pygame.Rect(MAP_WIDTH, 0, PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, COLOR_PANEL_HEADER, (MAP_WIDTH, 0), (MAP_WIDTH, WINDOW_HEIGHT), 3)

        y = 20

        # Header
        header = self.font_large.render("QUEEN", True, COLOR_PANEL_HEADER)
        self.screen.blit(header, (MAP_WIDTH + 20, y))
        y += 35
        subtitle = self.font_small.render("CENTRAL INTELLIGENCE", True, COLOR_PANEL_HEADER)
        self.screen.blit(subtitle, (MAP_WIDTH + 20, y))
        y += 40

        # Metrics
        metrics = [
            ("MODE", self.queen.current_mode[:15]),
            ("PUSHING ROBOTS", str(self.queen.pushing_robots)),
            ("PUSH SPEED", f"{self.queen.push_speed_percent}%"),
            ("FIRE STATUS", self.queen.fire_status)
        ]

        for label, value in metrics:
            label_text = self.font_small.render(label + ":", True, COLOR_PANEL_TEXT)
            self.screen.blit(label_text, (MAP_WIDTH + 20, y))
            y += 20

            value_color = COLOR_PANEL_HEADER if "FIRE" in label else (150, 255, 150)
            value_text = self.font_medium.render(value, True, value_color)
            self.screen.blit(value_text, (MAP_WIDTH + 20, y))
            y += 35

        y += 10

        # Log section
        log_header = self.font_medium.render("SYSTEM LOG", True, COLOR_PANEL_HEADER)
        self.screen.blit(log_header, (MAP_WIDTH + 20, y))
        y += 30

        # Draw logs
        for log_msg in self.queen.logs:
            # Word wrap
            words = log_msg.split()
            lines = []
            current_line = []

            for word in words:
                current_line.append(word)
                test_line = ' '.join(current_line)
                if self.font_small.size(test_line)[0] > PANEL_WIDTH - 45:
                    current_line.pop()
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]

            if current_line:
                lines.append(' '.join(current_line))

            for line in lines:
                log_text = self.font_small.render(line, True, COLOR_PANEL_TEXT)
                self.screen.blit(log_text, (MAP_WIDTH + 20, y))
                y += 18

                if y > WINDOW_HEIGHT - 40:
                    break

    def _export_frame(self):
        """Export current frame as PNG."""
        output_dir = "../output/frames"
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{output_dir}/frame_{self.frame_count:05d}.png"
        pygame.image.save(self.screen, filename)
        self.frame_count += 1


def main():
    """Entry point."""
    import sys

    export = "--export" in sys.argv

    if export:
        print("Running simulation with frame export...")
        print("Frames will be saved to output/frames/")

    sim = Simulation(export_frames=export)
    sim.run()

    if export:
        print(f"\nExport complete! {sim.frame_count} frames saved.")
        print("\nTo create MP4 video:")
        print("  cd ../output/frames")
        print(f"  ffmpeg -framerate {FPS} -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p simulation.mp4")


if __name__ == "__main__":
    main()
