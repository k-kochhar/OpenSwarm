import cv2
import numpy as np
from cv2 import aruco

video_source = 0
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: Could not open video source")
    exit()

# Use the ARUCO_ORIGINAL dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)

# Set up detector parameters
parameters = aruco.DetectorParameters()
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 53
parameters.adaptiveThreshWinSizeStep = 4
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate = 0.01
parameters.maxMarkerPerimeterRate = 8.0
parameters.polygonalApproxAccuracyRate = 0.1
parameters.minCornerDistanceRate = 0.01
parameters.minDistanceToBorder = 1
parameters.minMarkerDistanceRate = 0.01
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
parameters.cornerRefinementWinSize = 5
parameters.cornerRefinementMaxIterations = 50
parameters.cornerRefinementMinAccuracy = 0.01

# Create detector
detector = aruco.ArucoDetector(aruco_dict, parameters)

print("Press 'q' to quit")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame")
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)
    
    # Draw detected markers on the frame
    output_frame = frame.copy()
    if ids is not None:
        aruco.drawDetectedMarkers(output_frame, corners, ids)
        
        # Add text labels for each marker
        for corner, marker_id in zip(corners, ids):
            center = corner[0].mean(axis=0).astype(int)
            
            # Calculate orientation angle from top edge (corner 0 to corner 1)
            top_left = corner[0][0]
            top_right = corner[0][1]
            angle = np.degrees(np.arctan2(top_right[1] - top_left[1], top_right[0] - top_left[0]))
            
            text = f"Orientation: {angle:.1f} deg"
            cv2.putText(output_frame, text, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Print detected markers
        print(f"Detected markers: {ids.flatten().tolist()}")
    
    # Display the resulting frame
    cv2.imshow('ArUco Marker Detection', output_frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()