import json
import math
import threading
import asyncio
import time
import sys
import os
import numpy as np
from pathlib import Path
import cv2

# Add robot src to path so we can use utils
ROBOT_SRC = Path(__file__).parent.parent.parent / "robot" / "src"
sys.path.insert(0, str(ROBOT_SRC))

# Add hive dir so we can import llms
HIVE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(HIVE_DIR))

# Add move_world for neural pathfinder
MOVE_WORLD_DIR = HIVE_DIR / "move_world"
sys.path.insert(0, str(MOVE_WORLD_DIR))

from utils import PathFollower, RobotClient
from pathfinder import NeuralPathfinder

# Lazy-loaded neural pathfinder (singleton)
_neural_pf = None
_neural_pf_lock = threading.Lock()
_CHECKPOINT = MOVE_WORLD_DIR / "checkpoints" / "best_model.pt"


def _get_neural_pathfinder():
    """Get or create the NeuralPathfinder (loaded once, reused)."""
    global _neural_pf
    if _neural_pf is None:
        with _neural_pf_lock:
            if _neural_pf is None:
                _neural_pf = NeuralPathfinder(str(_CHECKPOINT))
                print(f"[pathfind] Neural heuristic loaded on {_neural_pf.device}")
    return _neural_pf

BOT_CLEAR_RADIUS = 4  # cells around each bot kept obstacle-free

# Shared IPC files (overlay.py writes markers.json, we read it)
PATH_JSON = ROBOT_SRC / "path.json"
MARKERS_JSON = ROBOT_SRC / "markers.json"
ACTIVE_BOTS_JSON = ROBOT_SRC / "active_bots.json"
SCREENSHOT_PATH = ROBOT_SRC / "screenshot.png"

GRID_SIZE = 32         # Vision grid (32x32 cells on the camera image)
FULL_GRID = 64         # Scaled grid (64x64 to match move_world)
CELL_FREE = 0
CELL_BOT = 1
CELL_OBSTACLE = 2

# Bot ID mapping: simple bot_id (0,1,2) → ArUco marker_id → ESP device
# The LLM and user see bot 0, 1, 2. Internally we map to marker IDs.
BOTS = {
    0: {"marker_id": 0, "device_id": "ESP1"},
    1: {"marker_id": 3, "device_id": "ESP2"},
    2: {"marker_id": 6, "device_id": "ESP3"},
}

# Reverse lookup: marker_id → bot_id
_MARKER_TO_BOT = {v["marker_id"]: k for k, v in BOTS.items()}

# Active bot controllers: {marker_id: {"thread", "stop", "device_id", "path", "waypoint_idx"}}
_active_bots = {}
_bots_lock = threading.Lock()

# Lock for path.json read-modify-write
_path_json_lock = threading.Lock()


def _resolve_bot(bot_id):
    """Resolve a simple bot_id (0,1,2) to (marker_id, device_id)."""
    bot_id = int(bot_id)
    if bot_id in BOTS:
        return BOTS[bot_id]["marker_id"], BOTS[bot_id]["device_id"]
    # If someone passes a raw marker_id, still handle it
    for bid, info in BOTS.items():
        if info["marker_id"] == bot_id:
            return info["marker_id"], info["device_id"]
    return bot_id, f"ESP{bot_id}"


def _read_markers():
    """Read latest marker positions from markers.json (written by overlay.py)."""
    if not MARKERS_JSON.exists():
        return {}
    try:
        data = json.loads(MARKERS_JSON.read_text())
        return {int(k): v for k, v in data.items()}
    except (json.JSONDecodeError, KeyError, ValueError):
        return {}


def _write_active_bots():
    """Write active bot state so overlay.py can show current waypoint indices."""
    with _bots_lock:
        out = {}
        for mid, entry in _active_bots.items():
            out[str(mid)] = {
                "active": entry["thread"].is_alive(),
                "waypoint_idx": entry["waypoint_idx"][0],
                "device_id": entry["device_id"],
            }
    try:
        ACTIVE_BOTS_JSON.write_text(json.dumps(out))
    except Exception:
        pass


def _load_path_json():
    """Read current path.json. Caller must hold _path_json_lock."""
    if not PATH_JSON.exists():
        return {"robots": {}}
    try:
        return json.loads(PATH_JSON.read_text())
    except (json.JSONDecodeError, KeyError):
        return {"robots": {}}


def _save_path_json(data):
    """Write back to path.json. Caller must hold _path_json_lock."""
    PATH_JSON.write_text(json.dumps(data, indent=2))


def _get_screenshot():
    """Read the latest screenshot from overlay.py."""
    if not SCREENSHOT_PATH.exists():
        raise RuntimeError("No screenshot available — is overlay.py running?")
    data = SCREENSHOT_PATH.read_bytes()
    if len(data) < 100:
        raise RuntimeError("Screenshot file is incomplete")
    return data


def _grid_screenshot(screenshot_bytes):
    """
    Overlay an 8x8 grid with row/col labels on the screenshot.
    Returns PNG bytes of the gridded image.
    """
    arr = np.frombuffer(screenshot_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode screenshot")
    h, w = img.shape[:2]

    cell_w = w / GRID_SIZE
    cell_h = h / GRID_SIZE

    # Draw grid lines + labels
    for r in range(GRID_SIZE + 1):
        y = int(r * cell_h)
        cv2.line(img, (0, y), (w, y), (0, 255, 255), 1)
    for c in range(GRID_SIZE + 1):
        x = int(c * cell_w)
        cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)

    # Label each cell with (row, col)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cx = int(c * cell_w + cell_w * 0.15)
            cy = int(r * cell_h + cell_h * 0.6)
            cv2.putText(img, f"{r},{c}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _pixel_to_grid(px, py, frame_w, frame_h):
    """Convert pixel coordinates to (row, col) in the 8x8 grid."""
    col = int(px / frame_w * GRID_SIZE)
    row = int(py / frame_h * GRID_SIZE)
    return (
        max(0, min(row, GRID_SIZE - 1)),
        max(0, min(col, GRID_SIZE - 1)),
    )


def _grid_to_pixel(row, col, frame_w, frame_h):
    """Convert (row, col) grid cell to pixel center."""
    cell_w = frame_w / GRID_SIZE
    cell_h = frame_h / GRID_SIZE
    px = col * cell_w + cell_w / 2
    py = row * cell_h + cell_h / 2
    return (px, py)


def _get_frame_size():
    """Get frame dimensions from the screenshot."""
    if not SCREENSHOT_PATH.exists():
        return 640, 480  # fallback
    img = cv2.imread(str(SCREENSHOT_PATH))
    if img is None:
        return 640, 480
    h, w = img.shape[:2]
    return w, h


def _detect_world_state():
    """
    Build a world matrix by combining bot positions with
    color-based obstacle detection from the camera screenshot.

    Green, blue, and black pixels are treated as obstacles.
    A radius of BOT_CLEAR_RADIUS cells around each bot is kept free.

    Returns:
        {
            "matrix": list (64x64) — 0=free, 1=bot, 2=obstacle,
            "bots": [{"bot_id": int, "pos": [x,y], "grid_pos": [r,c], ...}]
        }
    """
    markers = _read_markers()
    screenshot_bytes = _get_screenshot()
    frame_w, frame_h = _get_frame_size()

    # Build 32x32 matrix with bot positions
    matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    bot_list = []

    for bid, bot_info in sorted(BOTS.items()):
        info = markers.get(bot_info["marker_id"])
        if info:
            px, py = info["center"]
            r, c = _pixel_to_grid(px, py, frame_w, frame_h)
            matrix[r, c] = CELL_BOT
            bot_list.append({
                "bot_id": bid,
                "pos": list(info["center"]),
                "grid_pos": [r, c],
                "orientation": info["orientation_rad"],
                "orientation_deg": info["orientation_deg"],
            })

    # Decode screenshot for color-based detection
    arr = np.frombuffer(screenshot_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode screenshot")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color masks in HSV
    # Green
    mask_green = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    # Blue
    mask_blue = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
    # Black (low value)
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))

    obstacle_mask = mask_green | mask_blue | mask_black

    # Downsample mask to 32x32 by averaging each cell
    cell_h = frame_h / GRID_SIZE
    cell_w = frame_w / GRID_SIZE
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            y1 = int(r * cell_h)
            y2 = int((r + 1) * cell_h)
            x1 = int(c * cell_w)
            x2 = int((c + 1) * cell_w)
            cell_region = obstacle_mask[y1:y2, x1:x2]
            # If >30% of pixels in the cell are obstacle-colored, mark it
            if cell_region.mean() > 0.3 * 255:
                if matrix[r, c] == CELL_FREE:
                    matrix[r, c] = CELL_OBSTACLE

    # Clear a radius around each bot
    for bot in bot_list:
        br, bc = bot["grid_pos"]
        for dr in range(-BOT_CLEAR_RADIUS, BOT_CLEAR_RADIUS + 1):
            for dc in range(-BOT_CLEAR_RADIUS, BOT_CLEAR_RADIUS + 1):
                nr, nc = br + dr, bc + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    if matrix[nr, nc] == CELL_OBSTACLE:
                        matrix[nr, nc] = CELL_FREE

    # Scale 32x32 → 64x64 (each cell becomes a 2x2 block)
    scale = FULL_GRID // GRID_SIZE
    matrix_64 = np.repeat(np.repeat(matrix, scale, axis=0), scale, axis=1)

    return {
        "matrix": matrix_64.tolist(),
        "bots": bot_list,
    }


def _generate_waypoints(start, target, step_size=40):
    """
    Generate waypoints from start to target using neural heuristic A*
    on the color-detected obstacle grid. Falls back to straight-line
    if no grid or no path found.

    start/target: [x, y] pixel coordinates.
    Returns: list of [x, y] pixel waypoints.
    """
    frame_w, frame_h = _get_frame_size()

    # Try neural A* on the obstacle grid
    try:
        world = _detect_world_state()
        grid = np.array(world["matrix"], dtype=np.int32)

        # Convert obstacle grid to binary (0=free, 1=blocked)
        # Our grid uses 0=free, 1=bot, 2=obstacle — neural model expects 0/1
        binary_grid = (grid == CELL_OBSTACLE).astype(np.int32)

        # Clear start and goal cells
        s_col = int(start[0] / frame_w * FULL_GRID)
        s_row = int(start[1] / frame_h * FULL_GRID)
        t_col = int(target[0] / frame_w * FULL_GRID)
        t_row = int(target[1] / frame_h * FULL_GRID)
        s_row = max(0, min(s_row, FULL_GRID - 1))
        s_col = max(0, min(s_col, FULL_GRID - 1))
        t_row = max(0, min(t_row, FULL_GRID - 1))
        t_col = max(0, min(t_col, FULL_GRID - 1))
        binary_grid[s_row, s_col] = 0
        binary_grid[t_row, t_col] = 0

        pf = _get_neural_pathfinder()
        cell_w = frame_w / FULL_GRID
        cell_h = frame_h / FULL_GRID

        pixel_path = pf.find_path_pixel(
            binary_grid,
            start_xy=(start[0], start[1]),
            goal_xy=(target[0], target[1]),
            grid_origin=(0, 0),
            cell_size=min(cell_w, cell_h),
        )

        if pixel_path:
            # Convert tuples to lists, skip start
            waypoints = [[p[0], p[1]] for p in pixel_path[1:]]
            waypoints[-1] = [float(target[0]), float(target[1])]
            print(f"[pathfind] Neural A* path: {len(waypoints)} waypoints")
            return waypoints

        print("[pathfind] Neural A* found no path — falling back to straight line")
    except Exception as e:
        print(f"[pathfind] Neural pathfinder unavailable ({e}) — using straight line")

    # Fallback: straight-line waypoints
    sx, sy = start
    tx, ty = target
    dx, dy = tx - sx, ty - sy
    dist = math.sqrt(dx**2 + dy**2)

    if dist < step_size:
        return [[tx, ty]]

    n_steps = max(1, int(dist / step_size))
    path = []
    for i in range(1, n_steps + 1):
        frac = i / n_steps
        path.append([sx + dx * frac, sy + dy * frac])
    path[-1] = [tx, ty]
    return path


def _bot_control_loop(marker_id, device_id, path, stop_event, waypoint_idx):
    """
    Background loop that drives a single bot along its path.
    Reads marker positions from markers.json (written by overlay.py),
    sends F/L/R/S commands via WebSocket.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(
            _bot_control_async(marker_id, device_id, path, stop_event, waypoint_idx)
        )
    except Exception as e:
        print(f"[bot {marker_id}] control loop error: {e}")
    finally:
        loop.close()


async def _bot_control_async(marker_id, device_id, path, stop_event, waypoint_idx):
    """Async inner loop for a single bot."""
    path_follower = PathFollower(path)
    initialized = False

    robot = RobotClient(
        host="localhost",
        port=8765,
        device_id=f"CONTROLLER_{marker_id}",
        target_device=device_id,
    )
    connected = await robot.connect()
    if not connected:
        print(f"[bot {marker_id}] websocket connection failed — running without commands")

    last_cmd_time = 0.0
    cmd_interval = 0.5
    last_active_write = 0.0

    try:
        while not stop_event.is_set():
            markers = _read_markers()
            info = markers.get(marker_id)

            if info:
                rx, ry = info["center"]
                angle = info["orientation_rad"]

                if not initialized:
                    path_follower.initialize(rx, ry)
                    initialized = True

                cmd = path_follower.get_command(rx, ry, angle)

                # Update shared waypoint index for overlay (throttled)
                idx = path_follower.get_current_waypoint_idx()
                waypoint_idx[0] = idx if idx is not None else len(path) - 1
                now = time.time()
                if now - last_active_write >= 0.3:
                    _write_active_bots()
                    last_active_write = now

                if now - last_cmd_time >= cmd_interval and connected:
                    await robot.send_command(cmd)
                    last_cmd_time = now

                if path_follower.finished:
                    print(f"[bot {marker_id}] reached target")
                    break

            await asyncio.sleep(0.05)
    except Exception as e:
        print(f"[bot {marker_id}] error during control: {e}")
    finally:
        if connected:
            await robot.send_command("S")
        await robot.disconnect()
        _write_active_bots()


# ---------------------------------------------------------------------------
# Public actions (exposed to main.py / LLM)
# ---------------------------------------------------------------------------

def move_to(target, bot_id=0):
    """
    Move a bot to the target position. Pathfinding (straight-line waypoints)
    is handled automatically.

    Args:
        target: [x, y] target pixel position in the camera frame
        bot_id: which robot to move (0, 1, or 2)

    Returns:
        str — confirmation that the command was sent
    """
    marker_id, device_id = _resolve_bot(bot_id)
    target = [float(target[0]), float(target[1])]

    # Get current position to generate waypoints
    markers = _read_markers()
    info = markers.get(marker_id)
    if info:
        start = info["center"]
    else:
        # No camera data yet — use target as single waypoint
        start = target

    path = _generate_waypoints(start, target)

    # --- update path.json (atomic read-modify-write) ---
    with _path_json_lock:
        data = _load_path_json()
        data["robots"][str(marker_id)] = {
            "device_id": device_id,
            "path": path,
        }
        _save_path_json(data)

    # --- stop any existing movement for this bot ---
    with _bots_lock:
        old = _active_bots.pop(marker_id, None)
    if old:
        old["stop"].set()
        old["thread"].join(timeout=3)

    # --- launch background control thread ---
    stop_evt = threading.Event()
    waypoint_idx = [0]
    t = threading.Thread(
        target=_bot_control_loop,
        args=(marker_id, device_id, path, stop_evt, waypoint_idx),
        daemon=True,
    )
    t.start()

    with _bots_lock:
        _active_bots[marker_id] = {
            "thread": t,
            "stop": stop_evt,
            "device_id": device_id,
            "path": path,
            "waypoint_idx": waypoint_idx,
        }

    _write_active_bots()
    return f"Sent move_to command: bot={marker_id}, target={target}"


def stop(bot_id=0):
    """
    Stop a robot that is currently moving.

    Args:
        bot_id: which robot to stop (0, 1, or 2)

    Returns:
        str — confirmation message
    """
    marker_id, _ = _resolve_bot(bot_id)
    with _bots_lock:
        entry = _active_bots.pop(marker_id, None)
    if entry:
        entry["stop"].set()
        entry["thread"].join(timeout=3)
    _write_active_bots()
    if entry:
        return f"Bot {marker_id} stopped"
    return f"Bot {marker_id} was not moving"


def push_and_exit(bot_id=0):
    """
    Push an obstacle out of the area. Finds the nearest boundary point
    from the bot's current position and drives straight there, pushing
    whatever is in front of it off the edge.

    Use this after moving the bot to an obstacle — call push_and_exit
    to shove it out. The path is a straight line (no pathfinding),
    so obstacles along the way are ignored.

    Args:
        bot_id: which robot to use (0, 1, or 2)

    Returns:
        str — confirmation message
    """
    marker_id, device_id = _resolve_bot(bot_id)
    markers = _read_markers()
    info = markers.get(marker_id)
    if not info:
        return f"Bot {bot_id} not detected — cannot push_and_exit"

    bx, by = info["center"]
    angle = info["orientation_rad"]
    frame_w, frame_h = _get_frame_size()

    # Drive straight in the bot's current facing direction until hitting a boundary
    dx = math.cos(angle)
    dy = math.sin(angle)

    # Ray-march to frame edge
    t_max = max(frame_w, frame_h) * 2
    for step in range(1, int(t_max)):
        ex = bx + dx * step
        ey = by + dy * step
        if ex <= 0 or ex >= frame_w or ey <= 0 or ey >= frame_h:
            target = [float(max(0, min(ex, frame_w))), float(max(0, min(ey, frame_h)))]
            break
    else:
        target = [float(bx + dx * t_max), float(by + dy * t_max)]

    # Straight-line waypoints (no pathfinding — push through obstacles)
    sx, sy = bx, by
    tdx, tdy = target[0] - sx, target[1] - sy
    dist = math.sqrt(tdx**2 + tdy**2)
    n_steps = max(1, int(dist / 40))
    path = []
    for i in range(1, n_steps + 1):
        frac = i / n_steps
        path.append([sx + tdx * frac, sy + tdy * frac])
    path[-1] = target

    # Update path.json
    with _path_json_lock:
        data = _load_path_json()
        data["robots"][str(marker_id)] = {
            "device_id": device_id,
            "path": path,
        }
        _save_path_json(data)

    # Stop any existing movement
    with _bots_lock:
        old = _active_bots.pop(marker_id, None)
    if old:
        old["stop"].set()
        old["thread"].join(timeout=3)

    # Launch control thread
    stop_evt = threading.Event()
    waypoint_idx = [0]
    t = threading.Thread(
        target=_bot_control_loop,
        args=(marker_id, device_id, path, stop_evt, waypoint_idx),
        daemon=True,
    )
    t.start()

    with _bots_lock:
        _active_bots[marker_id] = {
            "thread": t,
            "stop": stop_evt,
            "device_id": device_id,
            "path": path,
            "waypoint_idx": waypoint_idx,
        }

    _write_active_bots()
    return f"Bot {bot_id} pushing toward boundary at {target}"


def get_orientation_coordinates(bot_id=None):
    """
    Get current position and orientation of robots from the camera.

    Args:
        bot_id: which robot to query (0, 1, or 2). If None, returns all.

    Returns:
        dict or list — For a single bot:
            {"bot_id": int, "center": [x, y], "orientation_deg": float, "orientation_rad": float}
        For all bots:
            [{"bot_id": int, "center": [x, y], "orientation_deg": float, "orientation_rad": float}, ...]
    """
    markers = _read_markers()

    if bot_id is not None:
        marker_id, _ = _resolve_bot(bot_id)
        info = markers.get(marker_id)
        if info is None:
            return {"error": f"Bot {bot_id} not detected"}
        return {
            "bot_id": int(bot_id),
            "center": list(info["center"]),
            "orientation_deg": info["orientation_deg"],
            "orientation_rad": info["orientation_rad"],
        }

    results = []
    for bid, bot_info in sorted(BOTS.items()):
        info = markers.get(bot_info["marker_id"])
        if info:
            results.append({
                "bot_id": bid,
                "center": list(info["center"]),
                "orientation_deg": info["orientation_deg"],
                "orientation_rad": info["orientation_rad"],
            })
    return results


# ---------------------------------------------------------------------------
# Framework-required private helpers
# ---------------------------------------------------------------------------

def _get_state():
    """
    Return the current world state (called by main.py's poll loop).

    Includes a 64x64 grid matrix from YOLO (obstacle detection)
    when a screenshot is available. Otherwise returns bot positions only.

    Returns:
        dict with:
            matrix     — 64x64 grid (0=free, 1=bot, 2=obstacle)
            bots       — list of bot dicts (pos, grid_pos, orientation)
            paths      — current path.json data
            active_bots — list of bot_ids currently moving
    """
    # Try the full vision pipeline; fall back to markers-only if unavailable
    try:
        world = _detect_world_state()
        bots = world["bots"]
        matrix = world["matrix"]
    except Exception as e:
        print(f"[state] Vision detection skipped: {e}")
        markers = _read_markers()
        frame_w, frame_h = _get_frame_size()
        bots = []
        for bid, bot_info in sorted(BOTS.items()):
            info = markers.get(bot_info["marker_id"])
            if info:
                px, py = info["center"]
                r, c = _pixel_to_grid(px, py, frame_w, frame_h)
                bots.append({
                    "bot_id": bid,
                    "pos": list(info["center"]),
                    "grid_pos": [r, c],
                    "orientation": info["orientation_rad"],
                    "orientation_deg": info["orientation_deg"],
                })
        matrix = None

    with _path_json_lock:
        path_data = _load_path_json()

    with _bots_lock:
        active = [
            _MARKER_TO_BOT.get(mid, mid)
            for mid, e in _active_bots.items()
            if e["thread"].is_alive()
        ]

    state = {
        "bots": bots,
        "paths": path_data,
        "active_bots": active,
    }
    if matrix is not None:
        state["matrix"] = matrix
    return state
