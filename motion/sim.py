import pygame
import numpy as np
import json
import math
import os
from robot import Robot
from motion_controller import AutonomousController

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60
BACKGROUND_COLOR = (59, 55, 52)

def load_path(filename):
    """Load paths from JSON file"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            return data.get('path1', []), data.get('path2', [])
    except FileNotFoundError:
        print(f"Path file {filepath} not found")
        return [], []
    except Exception as e:
        print(f"Error loading path: {e}")
        return [], []

def draw_path(screen, path, current_idx=None, color=(100, 100, 200)):
    """Draw the path waypoints and lines"""
    if not path:
        return
    
    # Draw lines between waypoints
    if len(path) > 1:
        pygame.draw.lines(screen, color, False, path, 2)
    
    # Draw waypoints
    for i, (x, y) in enumerate(path):
        if i == current_idx:
            # Current target waypoint - green
            pygame.draw.circle(screen, (0, 255, 0), (int(x), int(y)), 6)
        else:
            # Other waypoints
            pygame.draw.circle(screen, color, (int(x), int(y)), 3)

def main():
    # Initialize screen
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SCALED)
    pygame.display.set_caption("Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)
    
    # Load paths
    path1, path2 = load_path('path.json')
    
    # Store initial positions
    initial_x1 = path1[0][0] if path1 else 100
    initial_y1 = path1[0][1] if path1 else 400
    initial_x2 = path2[0][0] if path2 else 100
    initial_y2 = path2[0][1] if path2 else 100
    
    # Initialize robots with different marker IDs
    robot1 = Robot(initial_x1, initial_y1, angle=0, marker_id=23)
    robot2 = Robot(initial_x2, initial_y2, angle=0, marker_id=24)
    
    # Initialize controllers for both robots
    autonomous_controller1 = AutonomousController(path1, robot1.x, robot1.y)
    autonomous_controller2 = AutonomousController(path2, robot2.x, robot2.y)
    
    # Track current commands
    current_command1 = "L:0;R:0"
    current_command2 = "L:0;R:0"

    # Run simulation loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset robots to initial positions
                    robot1.x = initial_x1
                    robot1.y = initial_y1
                    robot1.angle = 0
                    robot1.set_wheels(0, 0)
                    robot2.x = initial_x2
                    robot2.y = initial_y2
                    robot2.angle = 0
                    robot2.set_wheels(0, 0)
                    # Reset controllers
                    autonomous_controller1.reset(robot1.x, robot1.y)
                    autonomous_controller2.reset(robot2.x, robot2.y)
                    # Reset commands
                    current_command1 = "L:0;R:0"
                    current_command2 = "L:0;R:0"
        
        # Update both robots in autonomous mode using commands
        current_command1 = autonomous_controller1.compute_command(robot1)
        robot1.execute_command(current_command1)
        
        current_command2 = autonomous_controller2.compute_command(robot2)
        robot2.execute_command(current_command2)
        
        # Update both robots
        robot1.update(WINDOW_WIDTH, WINDOW_HEIGHT)
        robot2.update(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Draw
        screen.fill(BACKGROUND_COLOR)
        
        # Draw both paths with different colors
        draw_path(screen, path1, autonomous_controller1.current_waypoint_idx, color=(100, 150, 200))
        draw_path(screen, path2, autonomous_controller2.current_waypoint_idx, color=(200, 150, 100))
        
        # Draw both robots
        robot1.draw(screen)
        robot2.draw(screen)
        
        # Simulate camera and detect markers for both robots
        frame = pygame.surfarray.array3d(screen)
        frame = np.transpose(frame, (1, 0, 2))
        robot1.detect_marker(frame)
        robot2.detect_marker(frame)
        
        # Display info
        angle1_str = f"{math.degrees(robot1.detected_angle):.1f}" if robot1.detected_angle is not None else "--"
        angle2_str = f"{math.degrees(robot2.detected_angle):.1f}" if robot2.detected_angle is not None else "--"
        
        info_lines = [
            f"Robot 1: ({int(robot1.x)}, {int(robot1.y)}), Angle: {angle1_str}°",
            f"Robot 2: ({int(robot2.x)}, {int(robot2.y)}), Angle: {angle2_str}°",
        ]

        # Draw info text
        y_offset = 10
        for line in info_lines:
            text_surface = font.render(line, True, (255, 255, 255))
            screen.blit(text_surface, (10, y_offset))
            y_offset += 20
        
        # Draw command bar at bottom
        command_bar_height = 50
        command_bar_y = WINDOW_HEIGHT - command_bar_height
        pygame.draw.rect(screen, (0, 0, 0), (0, command_bar_y, WINDOW_WIDTH, command_bar_height))
        
        # Draw commands vertically on the left
        command_y = command_bar_y + 5
        command1_surface = font.render(f"Robot 1: {current_command1}", True, (255, 255, 255))
        screen.blit(command1_surface, (10, command_y))
        
        command_y += 22
        command2_surface = font.render(f"Robot 2: {current_command2}", True, (255, 255, 255))
        screen.blit(command2_surface, (10, command_y))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()


if __name__ == "__main__":
    main()

