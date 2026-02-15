"""
Fire World Actions - Generalizable Tool System for Agent Control

This module defines tools that agents can use to interact with the fire world.
Each function represents a tool that can be called by the orchestration layer.

IPC Protocol:
    Reads:  files/sim_state.json      (bot positions, fire locations, stats)
            files/sim_screenshot.png   (visual representation of the grid)
    Writes: files/sim_commands.json    (commands for the simulation to execute)
"""

import json
import threading
from pathlib import Path
import numpy as np

# Grid configuration
GRID_SIZE = 64

# Cell types in the world matrix
CELL_FREE = 0
CELL_BOT = 1
CELL_OBSTACLE = 2
CELL_FIRE = 3

# IPC file paths
FILES_DIR = Path(__file__).parent.parent / "files"
STATE_PATH = FILES_DIR / "sim_state.json"
SCREENSHOT_PATH = FILES_DIR / "sim_screenshot.png"
COMMANDS_PATH = FILES_DIR / "sim_commands.json"

# Thread lock for command writing
_cmd_lock = threading.Lock()


# =============================================================================
# Internal State Access Functions (not exposed as tools)
# =============================================================================

def _get_state():
    """
    Read full simulation state from the IPC file.
    
    Returns:
        dict with keys:
            - bots: list of {pos: [r, c], orientation: float}
            - fires: list of [r, c] positions
            - fire_clusters: list of clusters, each is list of [r, c] positions
            - active_bots: list of bot IDs currently moving
            - stats: {fires_active: int, cells_extinguished: int}
    """
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except (json.JSONDecodeError, KeyError):
        return {}


def _get_bots():
    """Read bot positions and orientations from the simulation."""
    state = _get_state()
    return state.get("bots", [])


def _get_fires():
    """Read fire positions from the simulation."""
    state = _get_state()
    return state.get("fires", [])


def _get_fire_clusters():
    """Read fire cluster information from the simulation."""
    state = _get_state()
    return state.get("fire_clusters", [])


def _write_command(command):
    """
    Write a command to the IPC file for the simulation to process.
    Thread-safe.
    
    Args:
        command: dict with 'action' key and action-specific parameters
    """
    with _cmd_lock:
        commands = []
        if COMMANDS_PATH.exists():
            text = COMMANDS_PATH.read_text().strip()
            if text:
                try:
                    commands = json.loads(text)
                except json.JSONDecodeError:
                    commands = []
        
        commands.append(command)
        COMMANDS_PATH.write_text(json.dumps(commands))


# =============================================================================
# Agent Tools (exposed to the orchestration layer)
# =============================================================================

def move_to(target_pos, bot_id=0):
    """
    Navigate a bot to a target position on the grid.
    
    The simulation handles pathfinding automatically, avoiding obstacles
    and routing around fire hazards when possible.
    
    Args:
        target_pos: (row, col) or [row, col] - target position on the grid (0-63)
        bot_id: int - which bot to move (0 or 1)
    
    Returns:
        str - confirmation message
    """
    if isinstance(target_pos, (list, tuple)):
        target_pos = tuple(target_pos)
    else:
        raise ValueError(f"target_pos must be list or tuple, got {type(target_pos)}")
    
    _write_command({
        "action": "move_to",
        "target": list(target_pos),
        "bot": bot_id,
    })
    
    return f"Bot {bot_id}: Moving to {target_pos}"


def extinguish_flames(bot_id=0):
    """
    Extinguish all fire cells in the cluster adjacent to the bot.
    
    The bot must be positioned next to (within 1 cell of) a fire cluster.
    This will extinguish the entire connected cluster of flames, not just
    one cell. A white smoke effect is displayed during extinguishing.
    
    Args:
        bot_id: int - which bot should extinguish (0 or 1)
    
    Returns:
        str - result message indicating success or if no fires are adjacent
    """
    state = _get_state()
    bots = state.get("bots", [])
    
    if bot_id >= len(bots):
        return f"Error: Bot {bot_id} does not exist"
    
    bot_pos = tuple(bots[bot_id]["pos"])
    fire_clusters = state.get("fire_clusters", [])
    
    # Check if bot is adjacent to any fire
    adjacent = False
    for cluster in fire_clusters:
        for fire_pos in cluster:
            fr, fc = fire_pos
            br, bc = bot_pos
            if abs(fr - br) <= 1 and abs(fc - bc) <= 1 and (fr != br or fc != bc):
                adjacent = True
                break
        if adjacent:
            break
    
    if not adjacent:
        return f"Bot {bot_id}: No fires adjacent to position {bot_pos}. Move closer to a fire cluster first."
    
    _write_command({
        "action": "extinguish",
        "bot": bot_id,
    })
    
    return f"Bot {bot_id}: Extinguishing nearby fire cluster from position {bot_pos}"


def scan_area(bot_id=0):
    """
    Get detailed information about the area around the bot.
    
    Provides information about fires, hazards, and clear paths within
    a 10-cell radius of the bot's current position.
    
    Args:
        bot_id: int - which bot should scan (0 or 1)
    
    Returns:
        dict with:
            - fires_nearby: list of fire positions within range
            - closest_fire: [r, c] position of nearest fire, or None
            - distance_to_closest: float - distance to nearest fire
            - fire_clusters_nearby: list of clusters within range
    """
    state = _get_state()
    bots = state.get("bots", [])
    
    if bot_id >= len(bots):
        return {"error": f"Bot {bot_id} does not exist"}
    
    bot_pos = np.array(bots[bot_id]["pos"])
    fires = [np.array(f) for f in state.get("fires", [])]
    
    # Find fires within 10 cells
    fires_nearby = []
    distances = []
    for fire_pos in fires:
        dist = np.linalg.norm(fire_pos - bot_pos)
        if dist <= 10:
            fires_nearby.append(fire_pos.tolist())
            distances.append(dist)
    
    # Find closest fire
    closest_fire = None
    distance_to_closest = None
    if distances:
        min_idx = np.argmin(distances)
        closest_fire = fires_nearby[min_idx]
        distance_to_closest = float(distances[min_idx])
    
    # Find clusters within range
    fire_clusters = state.get("fire_clusters", [])
    clusters_nearby = []
    for cluster in fire_clusters:
        for fire_pos in cluster:
            dist = np.linalg.norm(np.array(fire_pos) - bot_pos)
            if dist <= 10:
                clusters_nearby.append(cluster)
                break
    
    return {
        "fires_nearby": fires_nearby,
        "closest_fire": closest_fire,
        "distance_to_closest": distance_to_closest,
        "fire_clusters_nearby": clusters_nearby,
        "scan_radius": 10,
    }


def get_status(bot_id=0):
    """
    Get the current status of a specific bot.
    
    Args:
        bot_id: int - which bot to query (0 or 1)
    
    Returns:
        dict with:
            - position: [r, c] current position
            - orientation: float - current facing direction in radians
            - is_moving: bool - whether bot is currently executing a move command
    """
    state = _get_state()
    bots = state.get("bots", [])
    
    if bot_id >= len(bots):
        return {"error": f"Bot {bot_id} does not exist"}
    
    bot = bots[bot_id]
    active_bots = state.get("active_bots", [])
    
    return {
        "position": bot["pos"],
        "orientation": bot["orientation"],
        "is_moving": bot_id in active_bots,
    }


# =============================================================================
# Tool Registration for LLM
# =============================================================================
# All public functions (not starting with _) are automatically discovered
# by main.py and exposed to the LLM as callable tools.
