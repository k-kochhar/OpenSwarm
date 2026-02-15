"""
Robot World overlay — run in a separate terminal.

    python overlay.py

Shows live camera feed with:
  - Colored arrows on each detected ArUco marker
  - Path lines + waypoints for all robots in path.json
  - Active waypoint indices from active_bots.json

Also writes markers.json so actions.py can read positions
without needing its own camera.

Press 'q' to quit.
"""

import cv2
import json
import numpy as np
import sys
import os
from pathlib import Path

ROBOT_SRC = Path(__file__).parent.parent.parent / "robot" / "src"
sys.path.insert(0, str(ROBOT_SRC))

from utils import Camera

HIVE_DIR = Path(__file__).parent.parent
PATH_JSON = ROBOT_SRC / "path.json"
MARKERS_JSON = ROBOT_SRC / "markers.json"
ACTIVE_BOTS_JSON = ROBOT_SRC / "active_bots.json"
SCREENSHOT_PATH = ROBOT_SRC / "screenshot.png"
STATE_JSON = HIVE_DIR / "files" / "state.json"

GRID_SIZE = 32  # must match actions.py

# Per-bot colors (BGR)
BOT_COLORS = [
    (0, 0, 255),    # red
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
]


def color_for_bot(marker_id):
    return BOT_COLORS[marker_id % len(BOT_COLORS)]


def load_paths():
    """Read path.json for all robot paths."""
    if not PATH_JSON.exists():
        return {}
    try:
        data = json.loads(PATH_JSON.read_text())
        return data.get("robots", {})
    except (json.JSONDecodeError, KeyError):
        return {}


def load_active_bots():
    """Read active_bots.json for current waypoint indices."""
    if not ACTIVE_BOTS_JSON.exists():
        return {}
    try:
        return json.loads(ACTIVE_BOTS_JSON.read_text())
    except (json.JSONDecodeError, KeyError):
        return {}


def write_markers(markers):
    """Write detected markers to markers.json for actions.py."""
    out = {}
    for mid, info in markers.items():
        out[str(mid)] = {
            "center": list(info["center"]),
            "orientation_deg": info["orientation_deg"],
            "orientation_rad": info["orientation_rad"],
        }
    try:
        MARKERS_JSON.write_text(json.dumps(out))
    except Exception:
        pass


def load_matrix():
    """Read the obstacle matrix from state.json (written by main.py)."""
    if not STATE_JSON.exists():
        return None
    try:
        data = json.loads(STATE_JSON.read_text())
        matrix = data.get("matrix")
        if matrix is None:
            return None
        # state has 64x64, downsample to 32x32 by taking every 2nd cell
        arr = np.array(matrix, dtype=int)
        if arr.shape == (64, 64):
            scale = 64 // GRID_SIZE
            small = arr[::scale, ::scale]
            return small
        if arr.shape == (GRID_SIZE, GRID_SIZE):
            return arr
        return None
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


BOT_CLEAR_RADIUS = 4  # cells around each bot kept obstacle-free


def detect_obstacles_color(frame, markers):
    """Detect green/blue/black objects as obstacles via color thresholds.
    Returns a 32x32 matrix with bot and obstacle cells marked."""
    h, w = frame.shape[:2]
    matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    # Mark bot positions
    bot_cells = []
    for mid, info in markers.items():
        px, py = info["center"]
        col = max(0, min(int(px / w * GRID_SIZE), GRID_SIZE - 1))
        row = max(0, min(int(py / h * GRID_SIZE), GRID_SIZE - 1))
        matrix[row, col] = 1  # CELL_BOT
        bot_cells.append((row, col))

    # Color detection in HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    mask_blue = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
    obstacle_mask = mask_green | mask_blue | mask_black

    # Downsample to 32x32
    cell_h = h / GRID_SIZE
    cell_w = w / GRID_SIZE
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            y1 = int(r * cell_h)
            y2 = int((r + 1) * cell_h)
            x1 = int(c * cell_w)
            x2 = int((c + 1) * cell_w)
            cell_region = obstacle_mask[y1:y2, x1:x2]
            if cell_region.mean() > 0.3 * 255:
                if matrix[r, c] == 0:
                    matrix[r, c] = 2  # CELL_OBSTACLE

    # Clear radius around each bot
    for br, bc in bot_cells:
        for dr in range(-BOT_CLEAR_RADIUS, BOT_CLEAR_RADIUS + 1):
            for dc in range(-BOT_CLEAR_RADIUS, BOT_CLEAR_RADIUS + 1):
                nr, nc = br + dr, bc + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    if matrix[nr, nc] == 2:
                        matrix[nr, nc] = 0

    return matrix


def draw_obstacles(frame, matrix):
    """Draw semi-transparent obstacle overlay on the frame."""
    if matrix is None:
        return frame

    h, w = frame.shape[:2]
    cell_w = w / GRID_SIZE
    cell_h = h / GRID_SIZE

    overlay = frame.copy()

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if matrix[r, c] == 2:  # obstacle
                x1 = int(c * cell_w)
                y1 = int(r * cell_h)
                x2 = int((c + 1) * cell_w)
                y2 = int((r + 1) * cell_h)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 200), -1)

    # Blend: 30% obstacle color, 70% original
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Draw grid lines so cells are visible
    for r in range(GRID_SIZE + 1):
        y = int(r * cell_h)
        cv2.line(frame, (0, y), (w, y), (80, 80, 80), 1)
    for c in range(GRID_SIZE + 1):
        x = int(c * cell_w)
        cv2.line(frame, (x, 0), (x, h), (80, 80, 80), 1)

    return frame


def draw_overlay(frame, markers, robots, active_bots):
    """Draw paths and robot arrows on frame."""

    # Draw paths from path.json
    for mid_str, robot_data in robots.items():
        mid = int(mid_str)
        path = robot_data.get("path", [])
        if not path:
            continue

        color = color_for_bot(mid)
        device_id = robot_data.get("device_id", f"ESP{mid}")

        # Active waypoint index (if bot is moving)
        active_info = active_bots.get(mid_str, {})
        active_idx = active_info.get("waypoint_idx")
        is_active = active_info.get("active", False)

        # Draw path lines
        for i in range(len(path) - 1):
            pt1 = (int(path[i][0]), int(path[i][1]))
            pt2 = (int(path[i + 1][0]), int(path[i + 1][1]))
            cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)

        # Draw waypoint dots
        for i, (x, y) in enumerate(path):
            if is_active and i == active_idx:
                # Current target: green with colored ring
                cv2.circle(frame, (int(x), int(y)), 7, (0, 255, 0), -1)
                cv2.circle(frame, (int(x), int(y)), 7, color, 2)
            else:
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)

        # Label at path start
        sx, sy = path[0]
        status = "MOVING" if is_active else "idle"
        label = f"Bot {mid} ({device_id}) [{status}]"
        cv2.putText(frame, label, (int(sx) - 10, int(sy) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Draw robot arrows for all detected markers
    for mid, info in markers.items():
        cx, cy = info["center"]
        angle = info["orientation_rad"]
        color = color_for_bot(mid)

        length = 50
        end_x = int(cx + length * np.cos(angle))
        end_y = int(cy + length * np.sin(angle))
        cv2.arrowedLine(frame, (int(cx), int(cy)), (end_x, end_y),
                        color, 3, tipLength=0.3)

        # Marker ID
        cv2.putText(frame, str(mid), (int(cx) + 10, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return frame


def main():
    camera = Camera(video_source=0, marker_id=None)
    print("[overlay] Camera started. Press 'q' to quit.")

    try:
        while True:
            frame, markers = camera.get_markers()
            if frame is None:
                continue

            # Write markers and screenshot so actions.py can read them
            write_markers(markers)
            # Atomic write: temp file + rename to avoid partial reads
            tmp = str(SCREENSHOT_PATH.with_suffix(".tmp.png"))
            cv2.imwrite(tmp, frame)
            os.replace(tmp, str(SCREENSHOT_PATH))

            # Read current paths and active state
            robots = load_paths()
            active_bots = load_active_bots()

            # Color-based obstacle detection
            matrix = detect_obstacles_color(frame, markers)

            # Draw obstacle grid overlay
            frame = draw_obstacles(frame, matrix)

            # Draw paths + robot arrows on top
            frame = draw_overlay(frame, markers, robots, active_bots)

            cv2.imshow("Robot World", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[overlay] Stopped.")
    finally:
        camera.release()
        # Clean up IPC files
        for f in (MARKERS_JSON, SCREENSHOT_PATH):
            if f.exists():
                f.unlink()


if __name__ == "__main__":
    main()
