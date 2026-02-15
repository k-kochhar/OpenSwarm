import cv2
import numpy as np
import math

def find_cameras():
    """Find all available camera indices"""
    available_cameras = []
    print("Searching for cameras...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"✓ Camera found at index {i}")
            cap.release()
    return available_cameras

def detect_shape_type(contour):
    """Classify shape as: aeroplane, dot_line, or t_shape"""
    area = cv2.contourArea(contour)
    if area < 300:  # Too small
        return None, None

    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0

    # Approximate contour to polygon
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    num_vertices = len(approx)

    # Calculate extent (ratio of contour area to bounding rectangle area)
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0

    # Classify shape
    # Aeroplane: Wide horizontal shape with wings (high aspect ratio, width > height)
    if aspect_ratio > 1.3 and w > h and extent > 0.4:
        return "aeroplane", approx

    # Dot with line: Very elongated shape (like a stick or arrow)
    elif (aspect_ratio > 2.5 or aspect_ratio < 0.4) and extent < 0.6:
        return "dot_line", approx

    # T-shape: Moderate aspect ratio, specific extent
    elif 0.5 < aspect_ratio < 1.5 and extent > 0.3:
        return "t_shape", approx

    return None, None

def get_orientation_dot_line(contour):
    """Get orientation from dot+line shape"""
    # Find the moments
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None, None, None

    # Calculate center
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Fit ellipse to get orientation
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            angle = ellipse[2]  # Angle in degrees

            # Calculate direction point
            angle_rad = math.radians(angle)
            length = 50
            dx = int(cx + length * math.cos(angle_rad))
            dy = int(cy + length * math.sin(angle_rad))

            return (cx, cy), (dx, dy), angle
        except:
            pass

    return (cx, cy), None, 0

def get_orientation_t_shape(contour):
    """Get orientation from T-shape"""
    # Find the moments
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None, None, None

    # Calculate center
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Find the topmost and bottommost points
    pts = contour.reshape(-1, 2)
    top_point = tuple(pts[pts[:, 1].argmin()])
    bottom_point = tuple(pts[pts[:, 1].argmax()])

    # Orientation is from center to top (T points upward)
    dx = top_point[0] - cx
    dy = top_point[1] - cy
    angle = math.degrees(math.atan2(dy, dx))

    # Calculate direction point
    direction_point = top_point

    return (cx, cy), direction_point, angle

def get_orientation_aeroplane(contour):
    """Get orientation from aeroplane shape"""
    # Find the moments
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None, None, None

    # Calculate center
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Get all points
    pts = contour.reshape(-1, 2)

    # Find the rightmost point (nose of plane points right)
    rightmost = tuple(pts[pts[:, 0].argmax()])

    # Calculate angle from center to nose
    dx = rightmost[0] - cx
    dy = rightmost[1] - cy
    angle = math.degrees(math.atan2(dy, dx))

    return (cx, cy), rightmost, angle

# Main detection loop
cameras = find_cameras()

if not cameras:
    print("ERROR: No cameras detected!")
    exit()

camera_index = cameras[0]
print(f"\nUsing camera index: {camera_index}")

cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"ERROR: Cannot open camera {camera_index}")
    exit()

print("\n✓ Camera opened successfully!")
print("\nDetecting shapes:")
print("  Robot 1: ✈️  Aeroplane (black airplane shape)")
print("  Robot 2: ●━ Dot + Line (black dot with line)")
print("  Robot 3: ┳  T-shape (black T)")
print("\nPress 'q' to quit\n")

# Robot tracking
robots = {
    "aeroplane": {"id": 1, "color": (0, 255, 0)},   # Green
    "dot_line": {"id": 2, "color": (255, 0, 0)},    # Blue
    "t_shape": {"id": 3, "color": (0, 0, 255)}      # Red
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold to detect black shapes (adjust 60 based on lighting)
    _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to clean noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Track detected shapes
    detected_shapes = {"aeroplane": [], "dot_line": [], "t_shape": []}

    # Process each contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200:  # Skip very small contours
            continue

        # Detect shape type
        shape_type, approx = detect_shape_type(contour)

        if shape_type:
            detected_shapes[shape_type].append(contour)

    # Draw detected shapes with orientation
    for shape_type, contour_list in detected_shapes.items():
        if contour_list and shape_type in robots:
            # Use the largest contour of this type
            contour = max(contour_list, key=cv2.contourArea)

            robot_info = robots[shape_type]
            robot_id = robot_info["id"]
            color = robot_info["color"]

            # Get orientation based on shape type
            if shape_type == "aeroplane":
                center, direction, angle = get_orientation_aeroplane(contour)
            elif shape_type == "dot_line":
                center, direction, angle = get_orientation_dot_line(contour)
            elif shape_type == "t_shape":
                center, direction, angle = get_orientation_t_shape(contour)
            else:
                continue

            if center:
                # Draw contour
                cv2.drawContours(frame, [contour], -1, color, 2)

                # Draw center point
                cv2.circle(frame, center, 5, color, -1)

                # Draw orientation line
                if direction:
                    cv2.line(frame, center, direction, color, 2)
                    cv2.circle(frame, direction, 3, (255, 255, 255), -1)

                # Display robot info
                text = f"R{robot_id}: ({center[0]}, {center[1]}) {angle:.1f}°"
                cv2.putText(frame, text, (center[0] + 10, center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Print to console
                print(f"Robot {robot_id} ({shape_type}): x={center[0]}, y={center[1]}, angle={angle:.1f}°")

    # Show frames
    cv2.imshow("Shape Detection", frame)
    cv2.imshow("Binary Mask", binary)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nCamera released successfully")
