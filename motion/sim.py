import pygame
import math
import numpy as np
import cv2
import os

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60
BACKGROUND_COLOR = (59, 55, 52)

# Robot parameters
ROBOT_RADIUS = 40
WHEEL_SPEED = 1.0
WHEEL_BASE = ROBOT_RADIUS * 1.5

# ArUco parameters
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 23
MARKER_SIZE = 50

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SCALED | pygame.RESIZABLE)

def generate_robot_image():
    """Generate a single robot image with ArUco marker inside white circle"""
    size = int(ROBOT_RADIUS * 4)
    robot_surface = pygame.Surface((size, size), pygame.SRCALPHA)
    center = size // 2
    
    # Draw white filled circle
    pygame.draw.circle(robot_surface, (255, 255, 255), (center, center), ROBOT_RADIUS)
        
    # Generate ArUco marker
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, MARKER_ID, MARKER_SIZE)
    marker_rgb = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2RGB)
    marker_surface = pygame.surfarray.make_surface(np.transpose(marker_rgb, (1, 0, 2)))
    
    # Blit marker at center of circle
    marker_rect = marker_surface.get_rect(center=(center, center))
    robot_surface.blit(marker_surface, marker_rect)
    
    # Front indicator
    front_x = center + ROBOT_RADIUS - 8
    front_y = center
    pygame.draw.circle(robot_surface, (200, 200, 200), (front_x, front_y), 5)
    
    return robot_surface


class Robot:
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle  # orientation in radians
        self.left_wheel = 0  # -100, 0, or 100
        self.right_wheel = 0  # -100, 0, or 100
        
        # Detection data
        self.detected_angle = None
        
        # Generate robot image once
        self.base_image = generate_robot_image()
        
    def set_wheels(self, left, right):
        """Set wheel speeds (-100, 0, or 100)"""
        self.left_wheel = left
        self.right_wheel = right
    
    def update(self):
        """Update robot position based on differential drive model"""
        # Calculate wheel velocities
        v_left = self.left_wheel * WHEEL_SPEED / 100.0
        v_right = self.right_wheel * WHEEL_SPEED / 100.0
        
        # Differential drive kinematics
        if v_left == v_right:
            # Straight motion
            linear_velocity = v_left
            self.x += linear_velocity * math.cos(self.angle)
            self.y += linear_velocity * math.sin(self.angle)
        else:
            # Calculate rotation
            linear_velocity = (v_left + v_right) / 2.0
            angular_velocity = (v_right - v_left) / WHEEL_BASE
            
            # Update position and orientation
            self.angle += angular_velocity
            self.x += linear_velocity * math.cos(self.angle)
            self.y += linear_velocity * math.sin(self.angle)
        
        # Keep robot within bounds
        self.x = max(ROBOT_RADIUS, min(WINDOW_WIDTH - ROBOT_RADIUS, self.x))
        self.y = max(ROBOT_RADIUS, min(WINDOW_HEIGHT - ROBOT_RADIUS, self.y))
    
    def draw(self, screen):
        """Draw the robot by rotating the pre-generated image"""
        # Rotate robot image
        angle_degrees = -math.degrees(self.angle)
        rotated_image = pygame.transform.rotate(self.base_image, angle_degrees)
        
        # Draw at robot position
        rect = rotated_image.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(rotated_image, rect)
    
    def detect_marker(self, frame):
        """Detect ArUco marker and update detected angle"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and MARKER_ID in ids:
            idx = np.where(ids == MARKER_ID)[0][0]
            marker_corners = corners[idx][0]
            
            # Calculate orientation from marker corners
            top_left = marker_corners[0]
            top_right = marker_corners[1]
            
            dx = top_right[0] - top_left[0]
            dy = top_right[1] - top_left[1]
            self.detected_angle = math.atan2(dy, dx)
            
            return True
        else:
            self.detected_angle = None
            return False


def main():
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SCALED)
    pygame.display.set_caption("Simulation")
    clock = pygame.time.Clock()
    
    robot = Robot(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2, angle=0)
    
    font = pygame.font.Font(None, 20)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_f]:
            # Forward: both wheels forward
            robot.set_wheels(100, 100)
        elif keys[pygame.K_l]:
            # Left: left wheel backward, right wheel forward
            robot.set_wheels(-100, 100)
        elif keys[pygame.K_r]:
            # Right: left wheel forward, right wheel backward
            robot.set_wheels(100, -100)
        elif keys[pygame.K_s]:
            # Stop: both wheels stopped
            robot.set_wheels(0, 0)
        
        # Update robot
        robot.update()
        
        # Draw everything
        screen.fill(BACKGROUND_COLOR)
        robot.draw(screen)
        
        # Simulate camera and detect markers
        frame = pygame.surfarray.array3d(screen)
        frame = np.transpose(frame, (1, 0, 2))
        robot.detect_marker(frame)
        
        # Display info
        info_lines = [
            f"X: {int(robot.x)}, Y: {int(robot.y)}",
        ]
        if robot.detected_angle is not None:
            info_lines.append(f"Angle: {math.degrees(robot.detected_angle):.1f}°")
        else:
            info_lines.append("Angle: --")
        
        y_offset = 10
        for line in info_lines:
            text_surface = font.render(line, True, (255, 255, 255))
            screen.blit(text_surface, (10, y_offset))
            y_offset += 20
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()


if __name__ == "__main__":
    main()
