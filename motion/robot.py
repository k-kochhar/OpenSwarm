from motion_controller import parse_command
import pygame
import math
import numpy as np
import cv2

# Robot parameters
ROBOT_RADIUS = 40
WHEEL_SPEED = 1.0
WHEEL_BASE = ROBOT_RADIUS * 1.5

# ArUco parameters
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_SIZE = 50


def generate_robot_image(marker_id):
    """Generate a single robot image with ArUco marker inside white circle"""
    size = int(ROBOT_RADIUS * 4)
    robot_surface = pygame.Surface((size, size), pygame.SRCALPHA)
    center = size // 2
    
    # Draw white filled circle
    pygame.draw.circle(robot_surface, (255, 255, 255), (center, center), ROBOT_RADIUS)
        
    # Generate ArUco marker
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, MARKER_SIZE)
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
    def __init__(self, x, y, angle=0, marker_id=23):
        self.x = x
        self.y = y
        self.angle = angle  # orientation in radians
        self.left_wheel = 0  # -100, 0, or 100
        self.right_wheel = 0  # -100, 0, or 100
        self.marker_id = marker_id
        
        # Detection data
        self.detected_angle = None
        
        # Generate robot image once
        self.base_image = generate_robot_image(marker_id)
        
    def set_wheels(self, left, right):
        """Set wheel speeds (-100, 0, or 100)"""
        self.left_wheel = left
        self.right_wheel = right
    
    def execute_command(self, command_string):
        """Execute a command string to control the robot"""
        left, right = parse_command(command_string)
        self.set_wheels(left, right)
    
    def update(self, window_width, window_height):
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
        self.x = max(ROBOT_RADIUS, min(window_width - ROBOT_RADIUS, self.x))
        self.y = max(ROBOT_RADIUS, min(window_height - ROBOT_RADIUS, self.y))
    
    def draw(self, screen):
        """Draw the robot by rotating the pre-generated image"""
        angle_degrees = -math.degrees(self.angle)
        rotated_image = pygame.transform.rotozoom(self.base_image, angle_degrees, 1.0)
        
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
        
        if ids is not None and self.marker_id in ids:
            idx = np.where(ids == self.marker_id)[0][0]
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
