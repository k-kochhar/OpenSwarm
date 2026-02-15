"""
Mimic World actions — exposed to main.py / LLM.

Coordinates 50 bots to arrange into shapes by sending pathfinding
requests to Modal and writing move commands to the simulation.

Hand tracking mode: captures webcam every 5s, detects hand via
MediaPipe, asks Claude what shape the hand is forming, and
calls form_shape accordingly.
"""

import json
import math
import time
import threading
import sys
import os
import numpy as np
import cv2
import mediapipe as mediapipe
from pathlib import Path
from scipy.optimize import linear_sum_assignment

# Add parent dir for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

GRID_SIZE = 64
WEBCAM_INDEX = 1  # MacBook Pro Camera
HAND_CHECK_INTERVAL = 1

# IPC file paths
FILES_DIR = Path(__file__).parent.parent / "files"
STATE_PATH = FILES_DIR / "mimic_state.json"
SCREENSHOT_PATH = FILES_DIR / "mimic_screenshot.png"
WEBCAM_PATH = FILES_DIR / "webcam_frame.png"
COMMANDS_PATH = FILES_DIR / "mimic_commands.json"

_cmd_lock = threading.Lock()
_hand_tracking_active = False
_tracking_thread = None


def _read_state():
    """Read current simulation state."""
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except (json.JSONDecodeError, KeyError):
        return {}


def _get_state():
    """Return current world state (called by main.py poll loop)."""
    return _read_state()


def _get_screenshot():
    """Read latest screenshot."""
    if not SCREENSHOT_PATH.exists():
        raise RuntimeError("No screenshot — is simulation.py running?")
    return SCREENSHOT_PATH.read_bytes()


def _to_native(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


def _send_commands(commands):
    """Write move commands to IPC file for simulation."""
    with _cmd_lock:
        existing = []
        if COMMANDS_PATH.exists():
            text = COMMANDS_PATH.read_text().strip()
            if text:
                try:
                    existing = json.loads(text)
                except json.JSONDecodeError:
                    existing = []
        existing.extend(_to_native(commands))
        COMMANDS_PATH.write_text(json.dumps(existing))


def _assign_bots_to_targets(bot_positions, target_positions):
    """
    Optimal assignment of bots to targets using Hungarian algorithm.
    Minimizes total manhattan distance across all assignments.

    Returns:
        list of (bot_idx, target_idx) pairs
    """
    n_bots = len(bot_positions)
    n_targets = len(target_positions)
    n = max(n_bots, n_targets)

    # Build cost matrix (manhattan distance)
    cost = np.zeros((n, n), dtype=np.float64)
    for i in range(n_bots):
        for j in range(n_targets):
            br, bc = bot_positions[i]
            tr, tc = target_positions[j]
            cost[i, j] = abs(br - tr) + abs(bc - tc)

    row_ind, col_ind = linear_sum_assignment(cost)
    assignments = []
    for i, j in zip(row_ind, col_ind):
        if i < n_bots and j < n_targets:
            assignments.append((i, j))
    return assignments


def _find_paths_modal(grid, requests):
    """Send pathfinding requests to Modal. Falls back to local if unavailable."""
    try:
        import modal
        Pathfinder = modal.Cls.from_name("mimic-world-pathfinder", "Pathfinder")
        pf = Pathfinder()
        results = pf.find_paths_batch.remote(grid.tolist(), requests)
        print(f"[actions] Modal returned {len(results)} results")
        return results
    except Exception as e:
        print(f"[actions] Modal unavailable ({e}), using local A*")
        return _find_paths_local(grid, requests)


def _build_waves(results):
    """
    Group paths into non-overlapping waves to avoid collisions.

    Two paths overlap if they share any grid cell. Paths in the same
    wave can move simultaneously. Waves are dispatched sequentially.

    Returns:
        list of waves, where each wave is a list of result dicts
    """
    waves = []
    for result in results:
        if not result["path"]:
            continue
        path_cells = set(tuple(p) for p in result["path"])

        # Try to fit into an existing wave
        placed = False
        for wave in waves:
            conflict = False
            for existing in wave:
                existing_cells = set(tuple(p) for p in existing["path"])
                if path_cells & existing_cells:
                    conflict = True
                    break
            if not conflict:
                wave.append(result)
                placed = True
                break

        if not placed:
            waves.append([result])

    return waves


def _wait_for_wave(wave, move_delay_ms=10):
    """Wait for the longest path in a wave to finish.

    Each step takes move_delay_ms in the simulation, so we wait
    (longest_path * move_delay_ms) + a small buffer.
    """
    longest = max(r["length"] for r in wave)
    wait_secs = (longest * move_delay_ms / 1000.0) + 0.1
    time.sleep(wait_secs)


def _find_paths_local(grid, requests):
    """Fallback: local A* without neural heuristic."""
    import heapq

    DIRS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    def astar(grid, start, goal):
        h, w = grid.shape
        sr, sc = start
        gr, gc = goal
        g_cost = {(sr, sc): 0.0}
        parent = {(sr, sc): None}
        closed = set()
        heuristic = lambda r, c: abs(r - gr) + abs(c - gc)
        heap = [(heuristic(sr, sc), 0.0, sr, sc)]
        while heap:
            _f, g, r, c = heapq.heappop(heap)
            if (r, c) in closed:
                continue
            closed.add((r, c))
            if (r, c) == (gr, gc):
                path = []
                node = (gr, gc)
                while node is not None:
                    path.append(list(node))
                    node = parent[node]
                path.reverse()
                return path
            for dr, dc, mc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                    ng = g + mc
                    if (nr, nc) not in g_cost or ng < g_cost[(nr, nc)]:
                        g_cost[(nr, nc)] = ng
                        parent[(nr, nc)] = (r, c)
                        heapq.heappush(heap, (ng + heuristic(nr, nc), ng, nr, nc))
        return None

    results = []
    for req in requests:
        path = astar(grid, tuple(req["start"]), tuple(req["goal"]))
        results.append({
            "bot_id": req["bot_id"],
            "path": path,
            "length": len(path) if path else 0,
        })
    return results


# ---------------------------------------------------------------------------
# Hand tracking internals
# ---------------------------------------------------------------------------

_HAND_MODEL_PATH = str(Path(__file__).parent / "hand_landmarker.task")

_HandLandmarker = mediapipe.tasks.vision.HandLandmarker
_HandLandmarkerOptions = mediapipe.tasks.vision.HandLandmarkerOptions
_BaseOptions = mediapipe.tasks.BaseOptions
_MPImage = mediapipe.Image
_MPImageFormat = mediapipe.ImageFormat

# Lazy-loaded landmarker (created once, reused)
_landmarker = None
_landmarker_lock = threading.Lock()

# Hand skeleton: connections between landmark indices
_HAND_SKELETON = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm cross-connections
    (5, 9), (9, 13), (13, 17),
]

GRID_PAD = 4  # padding from grid edges


def _get_landmarker():
    """Get or create the MediaPipe HandLandmarker (singleton)."""
    global _landmarker
    if _landmarker is None:
        with _landmarker_lock:
            if _landmarker is None:
                options = _HandLandmarkerOptions(
                    base_options=_BaseOptions(model_asset_path=_HAND_MODEL_PATH),
                    num_hands=1,
                    min_hand_detection_confidence=0.3,
                    min_hand_presence_confidence=0.3,
                )
                _landmarker = _HandLandmarker.create_from_options(options)
    return _landmarker


_cap = None
_cap_lock = threading.Lock()


def _get_camera():
    """Get or open the persistent webcam capture."""
    global _cap
    if _cap is None or not _cap.isOpened():
        with _cap_lock:
            if _cap is None or not _cap.isOpened():
                _cap = cv2.VideoCapture(WEBCAM_INDEX)
                if _cap.isOpened():
                    # Let camera warm up
                    for _ in range(5):
                        _cap.read()
                    print(f"[hand] Camera {WEBCAM_INDEX} opened")
                else:
                    print("[hand] Could not open webcam")
    return _cap


def _capture_webcam():
    """Grab the latest frame from the persistent webcam stream."""
    cap = _get_camera()
    if cap is None or not cap.isOpened():
        return None
    ret, frame = cap.read()
    if not ret:
        print("[hand] Failed to read frame")
        return None
    cv2.imwrite(str(WEBCAM_PATH), frame)
    return frame


def _detect_hand(frame):
    """Use MediaPipe to detect hand landmarks. Returns landmarks list or None."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = _MPImage(image_format=_MPImageFormat.SRGB, data=rgb)
    result = _get_landmarker().detect(mp_image)
    if result.hand_landmarks:
        return result.hand_landmarks[0]
    return None


HAND_FIXED_SIZE = 48  # fixed span in grid cells, regardless of real hand size


def _landmarks_to_targets(landmarks, n_bots):
    """
    Convert 21 hand landmarks into n_bots grid positions by
    interpolating points along the hand skeleton.

    Landmarks have normalized x,y in [0,1]. We normalize the hand's
    own coordinate space (relative scaling) so it always appears the
    same size and centered on the grid, regardless of distance or
    position in the webcam frame.
    """
    # Extract (x, y) — mirror x so it looks natural
    pts = np.array([(1.0 - lm.x, lm.y) for lm in landmarks])

    # Center on origin (relative to hand's own center)
    cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
    cy = (pts[:, 1].min() + pts[:, 1].max()) / 2
    pts[:, 0] -= cx
    pts[:, 1] -= cy

    # Scale to fixed size — the hand always occupies HAND_FIXED_SIZE cells
    span = max(pts[:, 0].max() - pts[:, 0].min(),
               pts[:, 1].max() - pts[:, 1].min()) or 1e-6
    scale = HAND_FIXED_SIZE / span
    pts *= scale

    # Place at grid center
    grid_pts = np.zeros_like(pts)
    grid_pts[:, 0] = pts[:, 0] + GRID_SIZE / 2
    grid_pts[:, 1] = pts[:, 1] + GRID_SIZE / 2

    # Compute total skeleton length for proportional sampling
    segments = []
    total_len = 0.0
    for a, b in _HAND_SKELETON:
        d = np.linalg.norm(grid_pts[a] - grid_pts[b])
        segments.append((a, b, d))
        total_len += d

    # Distribute points proportionally along skeleton segments
    # Oversample to ensure we get enough unique cells
    targets = set()
    for a, b, seg_len in segments:
        n_pts = max(2, round(n_bots * 2 * seg_len / total_len))
        for t in np.linspace(0, 1, n_pts):
            p = grid_pts[a] * (1 - t) + grid_pts[b] * t
            r = int(np.clip(round(p[1]), 0, GRID_SIZE - 1))  # y → row
            c = int(np.clip(round(p[0]), 0, GRID_SIZE - 1))  # x → col
            targets.add((r, c))

    targets = list(targets)

    # If too many, keep evenly spaced subset
    if len(targets) > n_bots:
        step = len(targets) / n_bots
        targets = [targets[int(i * step)] for i in range(n_bots)]

    # If too few, pad with neighbors of existing targets
    if len(targets) < n_bots:
        existing = set(targets)
        for r, c in list(targets):
            if len(existing) >= n_bots:
                break
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and (nr, nc) not in existing:
                    existing.add((nr, nc))
                    if len(existing) >= n_bots:
                        break
        targets = list(existing)[:n_bots]

    return targets


def _dispatch_to_targets(target_positions):
    """
    Move bots to arbitrary target positions using the same
    Hungarian + wave dispatch pipeline as form_shape.
    """
    state = _read_state()
    if not state or "bots" not in state:
        return "No simulation state — is simulation.py running?"

    bots = state["bots"]
    grid = np.array(state.get("grid", np.zeros((GRID_SIZE, GRID_SIZE))), dtype=np.int32)

    # For blocked targets, find nearest free cell
    used = set()
    valid_targets = []
    for r, c in target_positions:
        if grid[r, c] == 0 and (r, c) not in used:
            valid_targets.append((r, c))
            used.add((r, c))
        else:
            # Spiral search for nearest free cell
            found = False
            for radius in range(1, GRID_SIZE):
                if found:
                    break
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        if abs(dr) != radius and abs(dc) != radius:
                            continue
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE
                                and grid[nr, nc] == 0 and (nr, nc) not in used):
                            valid_targets.append((nr, nc))
                            used.add((nr, nc))
                            found = True
                            break

    n_assign = min(len(bots), len(valid_targets))
    bot_positions = [tuple(b["pos"]) for b in bots[:n_assign]]

    assignments = _assign_bots_to_targets(bot_positions, valid_targets[:n_assign])

    path_requests = []
    for bot_idx, target_idx in assignments:
        bp = bot_positions[bot_idx]
        tp = valid_targets[target_idx]
        if bp != tp:
            path_requests.append({
                "bot_id": bot_idx,
                "start": list(bp),
                "goal": list(tp),
            })

    if not path_requests:
        return "Bots already in position"

    print(f"[hand] Computing {len(path_requests)} paths...")
    results = _find_paths_modal(grid, path_requests)

    valid_results = [r for r in results if r["path"]]
    if not valid_results:
        return "No paths found"

    waves = _build_waves(valid_results)
    print(f"[hand] {len(valid_results)} paths, {len(waves)} waves")

    for wave_idx, wave in enumerate(waves):
        commands = []
        for result in wave:
            commands.append({
                "action": "move_to",
                "bot": result["bot_id"],
                "target": result["path"][-1],
                "path": result["path"],
            })
        _send_commands(commands)
        if wave_idx < len(waves) - 1:
            _wait_for_wave(wave)

    return f"{len(valid_results)} bots dispatched in {len(waves)} waves"


def _hand_tracking_loop():
    """Background loop: webcam → mediapipe → landmarks → dispatch bots."""
    global _hand_tracking_active

    print(f"[hand] Tracking started (webcam {WEBCAM_INDEX}, every {HAND_CHECK_INTERVAL}s)")

    while _hand_tracking_active:
        frame = _capture_webcam()
        if frame is None:
            time.sleep(HAND_CHECK_INTERVAL)
            continue

        landmarks = _detect_hand(frame)
        if landmarks is None:
            print("[hand] No hand detected — holding position")
            time.sleep(HAND_CHECK_INTERVAL)
            continue

        # Read bot count from state
        state = _read_state()
        n_bots = len(state.get("bots", [])) if state else 50

        targets = _landmarks_to_targets(landmarks, n_bots)

        print(f"[hand] Hand detected — {len(targets)} target positions")
        result = _dispatch_to_targets(targets)
        print(f"[hand] {result}")

        time.sleep(HAND_CHECK_INTERVAL)

    print("[hand] Tracking stopped")


# ---------------------------------------------------------------------------
# Public actions (exposed to main.py / LLM)
# ---------------------------------------------------------------------------

def form_shape(shape_name):
    """
    Arrange bots into a shape using wave-based dispatch.

    1. Generates target positions for the shape
    2. Skips any target that lands on an obstacle (no bot placed there)
    3. Assigns remaining bots to remaining targets (Hungarian algorithm)
    4. Computes all paths via Modal (neural A*)
    5. Groups paths into non-overlapping waves
    6. Dispatches waves sequentially so bots don't collide

    Args:
        shape_name: one of "circle", "square", "triangle", "star", "grid"

    Returns:
        str — summary of the operation
    """
    from simulation import SHAPES

    if shape_name not in SHAPES:
        return f"Unknown shape: {shape_name}. Choose from: {', '.join(SHAPES)}"

    state = _read_state()
    if not state or "bots" not in state:
        return "No simulation state — is simulation.py running?"

    bots = state["bots"]
    grid = np.array(state.get("grid", np.zeros((GRID_SIZE, GRID_SIZE))), dtype=np.int32)

    # Generate target positions and skip any on obstacles
    raw_targets = SHAPES[shape_name]()
    seen = set()
    target_positions = []
    skipped = 0
    for r, c in raw_targets:
        if grid[r, c] == 0 and (r, c) not in seen:
            target_positions.append((r, c))
            seen.add((r, c))
        else:
            skipped += 1

    if skipped:
        print(f"[actions] Skipped {skipped} targets on obstacles")

    # Only assign as many bots as we have valid targets
    n_assign = min(len(bots), len(target_positions))
    bot_positions = [tuple(b["pos"]) for b in bots[:n_assign]]

    # Optimal assignment
    assignments = _assign_bots_to_targets(bot_positions, target_positions[:n_assign])

    # Build pathfinding requests
    path_requests = []
    for bot_idx, target_idx in assignments:
        bp = bot_positions[bot_idx]
        tp = target_positions[target_idx]
        if bp != tp:
            path_requests.append({
                "bot_id": bot_idx,
                "start": list(bp),
                "goal": list(tp),
            })

    if not path_requests:
        return f"All bots already in {shape_name} formation!"

    # Compute all paths via Modal
    print(f"[actions] Computing {len(path_requests)} paths via Modal...")
    results = _find_paths_modal(grid, path_requests)

    # Filter to only successful paths
    valid_results = [r for r in results if r["path"]]
    if not valid_results:
        return f"No paths found for shape '{shape_name}'"

    # Group into non-overlapping waves
    waves = _build_waves(valid_results)
    print(f"[actions] {len(valid_results)} paths split into {len(waves)} waves")

    # Dispatch waves sequentially
    for wave_idx, wave in enumerate(waves):
        print(f"[actions] Wave {wave_idx + 1}/{len(waves)}: {len(wave)} bots")
        commands = []
        for result in wave:
            commands.append({
                "action": "move_to",
                "bot": result["bot_id"],
                "target": result["path"][-1],
                "path": result["path"],
            })
        _send_commands(commands)

        # Wait for this wave to finish before sending next
        if wave_idx < len(waves) - 1:
            _wait_for_wave(wave)

    return (
        f"Shape '{shape_name}': {len(valid_results)}/{len(path_requests)} paths found, "
        f"{len(waves)} waves dispatched. "
        f"{skipped} targets skipped (obstacles)."
    )


def move_bot(target_pos, bot_id=0):
    """
    Move a single bot to a target position using pathfinding.

    Args:
        target_pos: [row, col] target position
        bot_id: which bot to move (0-49)

    Returns:
        str — confirmation message
    """
    state = _read_state()
    if not state or "bots" not in state:
        return "No simulation state"

    bots = state["bots"]
    if bot_id < 0 or bot_id >= len(bots):
        return f"Invalid bot_id: {bot_id}"

    grid = np.array(state.get("grid", np.zeros((GRID_SIZE, GRID_SIZE))), dtype=np.int32)
    start = bots[bot_id]["pos"]
    goal = [int(target_pos[0]), int(target_pos[1])]

    results = _find_paths_modal(grid, [{"bot_id": bot_id, "start": start, "goal": goal}])

    if results and results[0]["path"]:
        _send_commands([{
            "action": "move_to",
            "bot": bot_id,
            "target": goal,
            "path": results[0]["path"],
        }])
        return f"Bot {bot_id} moving to {goal} ({results[0]['length']} steps)"
    return f"No path found for bot {bot_id} to {goal}"


def get_positions(bot_id=None):
    """
    Get current positions of bots.

    Args:
        bot_id: specific bot (0-49), or None for all

    Returns:
        dict or list of bot positions
    """
    state = _read_state()
    if not state or "bots" not in state:
        return {"error": "No simulation state"}

    bots = state["bots"]
    if bot_id is not None:
        bot_id = int(bot_id)
        if bot_id < 0 or bot_id >= len(bots):
            return {"error": f"Invalid bot_id: {bot_id}"}
        return {"bot_id": bot_id, "pos": bots[bot_id]["pos"]}

    return [{"bot_id": i, "pos": b["pos"]} for i, b in enumerate(bots)]


def start_hand_tracking():
    """
    Start webcam hand tracking. Captures a photo every 5 seconds,
    detects hand via MediaPipe, asks Claude what shape the gesture
    represents, and calls form_shape automatically.

    Returns:
        str — confirmation message
    """
    global _hand_tracking_active, _tracking_thread
    if _hand_tracking_active:
        return "Hand tracking is already running"
    _hand_tracking_active = True
    _tracking_thread = threading.Thread(target=_hand_tracking_loop, daemon=True)
    _tracking_thread.start()
    return "Hand tracking started — checking webcam every 5s"


def stop_hand_tracking():
    """
    Stop webcam hand tracking.

    Returns:
        str — confirmation message
    """
    global _hand_tracking_active, _cap
    if not _hand_tracking_active:
        return "Hand tracking is not running"
    _hand_tracking_active = False
    if _cap is not None:
        _cap.release()
        _cap = None
    return "Hand tracking stopped"


# Auto-start hand tracking when module is loaded
start_hand_tracking()
