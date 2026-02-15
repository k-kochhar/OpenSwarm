import cv2
import cv2.aruco as aruco
import numpy as np

def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    """
    Detect ArUco markers in the image

    Parameters:
    - img: Input image
    - markerSize: Size of the marker (4, 5, 6, 7)
    - totalMarkers: Total number of markers in dictionary (50, 100, 250, 1000)
    - draw: Whether to draw detected markers

    Returns:
    - bboxs: Bounding boxes of detected markers
    - ids: IDs of detected markers
    """
    # Convert to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get the ArUco dictionary
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()

    # Detect markers
    bboxs, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)

    # Draw detected markers
    if draw and len(bboxs) > 0:
        aruco.drawDetectedMarkers(img, bboxs, ids)

    return bboxs, ids

def getMarkerInfo(bbox, id, img):
    """
    Extract marker information and draw on image

    Parameters:
    - bbox: Bounding box of the marker
    - id: Marker ID
    - img: Image to draw on

    Returns:
    - center: Center point (x, y)
    - corners: Four corner points
    - angle: Orientation angle
    """
    # Get the four corners
    tl = bbox[0][0]  # Top-left
    tr = bbox[0][1]  # Top-right
    br = bbox[0][2]  # Bottom-right
    bl = bbox[0][3]  # Bottom-left

    # Calculate center point
    center_x = int((tl[0] + tr[0] + br[0] + bl[0]) / 4)
    center_y = int((tl[1] + tr[1] + br[1] + bl[1]) / 4)
    center = (center_x, center_y)

    # Calculate orientation (angle from center to top-right corner)
    dx = tr[0] - center_x
    dy = tr[1] - center_y
    angle = np.degrees(np.arctan2(dy, dx))

    # Draw center point
    cv2.circle(img, center, 5, (0, 255, 0), -1)

    # Draw orientation line (from center to top-right)
    cv2.line(img, center, tuple(tr.astype(int)), (255, 0, 0), 2)

    # Display marker ID and position
    text_pos = (int(tl[0]), int(tl[1] - 15))
    cv2.putText(img, f"ID:{id} ({center_x},{center_y})",
                text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display angle
    angle_pos = (int(tl[0]), int(tl[1] - 35))
    cv2.putText(img, f"Angle: {angle:.1f}",
                angle_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    corners = {'tl': tl, 'tr': tr, 'br': br, 'bl': bl}

    return center, corners, angle

# Camera setup
url = "http://10.19.182.191:8080/video"

print("Connecting to IP Camera...")
print(f"URL: {url}")

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Failed to open stream")
    print("\nTroubleshooting:")
    print("1. Check if phone and PC are on same WiFi")
    print("2. Verify IP address matches IP Webcam app")
    print("3. Make sure IP Webcam app is running")
    exit()

print("✓ Camera connected successfully!")
print("\nArUco Marker Detection Active")
print("Dictionary: 6x6, 250 markers")
print("\nControls:")
print("  ESC - Exit")
print("  's' - Save screenshot")
print("  't' - Toggle marker detection\n")

# Detection toggle
detect_markers = True
screenshot_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect ArUco markers
    if detect_markers:
        bboxs, ids = findArucoMarkers(frame, markerSize=6, totalMarkers=250, draw=True)

        # Process each detected marker
        if ids is not None:
            for bbox, id in zip(bboxs, ids):
                center, corners, angle = getMarkerInfo(bbox, id[0], frame)

                # Print marker info to console
                print(f"Marker ID {id[0]}: Center=({center[0]}, {center[1]}), Angle={angle:.1f}°")

        # Display marker count
        marker_count = len(ids) if ids is not None else 0
        cv2.putText(frame, f"Markers: {marker_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display detection status
    status = "ON" if detect_markers else "OFF"
    status_color = (0, 255, 0) if detect_markers else (0, 0, 255)
    cv2.putText(frame, f"Detection: {status}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Show frame
    cv2.imshow("ArUco Marker Detection - IP Camera", frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC to exit
        print("Exiting...")
        break
    elif key == ord('s'):  # Save screenshot
        screenshot_count += 1
        filename = f'aruco_screenshot_{screenshot_count}.jpg'
        cv2.imwrite(filename, frame)
        print(f"✓ Screenshot saved: {filename}")
    elif key == ord('t'):  # Toggle detection
        detect_markers = not detect_markers
        status = "ON" if detect_markers else "OFF"
        print(f"Detection: {status}")

cap.release()
cv2.destroyAllWindows()
print("Camera released successfully")
