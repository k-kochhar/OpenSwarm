import cv2, numpy as np, math

def orientation_from_contour(cnt):
    pts = cnt.reshape(-1,2).astype(np.float32)
    mean = pts.mean(axis=0)
    pts0 = pts - mean
    cov = np.cov(pts0.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    v = v / np.linalg.norm(v)

    proj = (pts - mean) @ v
    front = pts[np.argmax(proj)]
    angle = math.degrees(math.atan2(front[1]-mean[1], front[0]-mean[0]))
    return mean, front, angle

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

    if not available_cameras:
        print("✗ No cameras found")
    else:
        print(f"\nTotal cameras found: {len(available_cameras)}")

    return available_cameras

# Find and select camera
cameras = find_cameras()

if not cameras:
    print("ERROR: No cameras detected!")
    print("\nTroubleshooting:")
    print("1. Check if camera is connected")
    print("2. Close other apps using camera (Teams, Zoom, etc.)")
    print("3. Check Windows Privacy settings for camera access")
    exit()

# Use first available camera
camera_index = cameras[0]
print(f"\nUsing camera index: {camera_index}")

cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"ERROR: Cannot open camera {camera_index}")
    exit()

print("Camera opened successfully!")
print("Press ESC to quit\n")

while True:
    ok, frame = cap.read()
    if not ok:
        print("ERROR: Cannot read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # only black -> mask (tune 80 depending on light)
    _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # clean noise
    k = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1200:   # reject small stuff (tune)
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = area / hull_area
        if solidity < 0.6:   # reject thin random edges
            continue

        candidates.append((area, cnt))

    # keep only 3 biggest markers (3 robots)
    candidates.sort(key=lambda x: x[0], reverse=True)
    candidates = candidates[:3]

    for i, (area, cnt) in enumerate(candidates, start=1):
        center, front, ang = orientation_from_contour(cnt)

        c = tuple(center.astype(int))
        f = tuple(front.astype(int))

        cv2.circle(frame, c, 4, (0,255,0), -1)
        cv2.line(frame, c, f, (0,255,0), 2)
        cv2.putText(frame, f"R{i} {ang:.1f}",
                    (c[0]+10, c[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("frame", frame)
    cv2.imshow("mask (only markers)", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
