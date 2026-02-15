import base64
import json
import sys
import os
import threading
import numpy as np
from pathlib import Path

# Add parent dir so we can import llms
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llms import oai

GRID_SIZE = 64

# Cell encoding: 0 = free, 1 = bot, 2 = obstacle
CELL_FREE = 0
CELL_BOT = 1
CELL_OBSTACLE = 2

# IPC file paths
FILES_DIR = Path(__file__).parent.parent / "files"
STATE_PATH = FILES_DIR / "sim_state.json"
SCREENSHOT_PATH = FILES_DIR / "sim_screenshot.png"
COMMANDS_PATH = FILES_DIR / "sim_commands.json"

# Lock to prevent concurrent threads from clobbering each other's commands
_cmd_lock = threading.Lock()

DETECT_OBSTACLES_PROMPT = (
    "You are a grid-world vision system. You are given:\n"
    "1. A screenshot of a 64x64 grid world\n"
    "2. A partially filled 64x64 matrix (JSON) where 1 = bot position, 0 = unknown\n\n"
    "Your job: look at the screenshot and identify all obstacle cells. "
    "Return the COMPLETE 64x64 matrix as JSON where:\n"
    "  0 = free cell\n"
    "  1 = bot\n"
    "  2 = obstacle\n\n"
    "Return ONLY the JSON array (list of 64 lists, each with 64 ints). No explanation."
)


def _get_bots():
    """Read bot positions and orientations from the simulation."""
    if not STATE_PATH.exists():
        return []
    try:
        data = json.loads(STATE_PATH.read_text())
        return [
            {"pos": tuple(b["pos"]), "orientation": b["orientation"]}
            for b in data["bots"]
        ]
    except (json.JSONDecodeError, KeyError):
        return []


def _get_state():
    """Read full simulation state from the IPC file."""
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except (json.JSONDecodeError, KeyError):
        return {}


def _get_screenshot():
    """
    Read the latest screenshot from the simulation.

    Returns:
        bytes — PNG image data
    """
    if not SCREENSHOT_PATH.exists():
        raise RuntimeError("No screenshot available — is the simulation running?")
    return SCREENSHOT_PATH.read_bytes()


def move_to(target_pos, bot_id=0):
    """
    Command a bot to move to target_pos.

    The simulation handles pathfinding internally.
    This writes a command that the simulation picks up.

    Args:
        target_pos: (row, col) target position
        bot_id: which bot to move (0 or 1)

    Returns:
        str — confirmation that command was sent
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

        commands.append({
            "action": "move_to",
            "target": list(target_pos),
            "bot": bot_id,
        })
        COMMANDS_PATH.write_text(json.dumps(commands))
    return f"Sent move_to command: bot={bot_id}, target={target_pos}"


def collect(bot_id=0):
    """
    Collect a coin at the specified bot's current position.

    The bot must already be standing on a coin cell.
    Call move_to first to navigate to a coin, then call collect.

    Args:
        bot_id: which bot should collect (0 or 1)

    Returns:
        str — result of the collect attempt
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

        commands.append({"action": "collect", "bot": bot_id})
        COMMANDS_PATH.write_text(json.dumps(commands))
    return f"Sent collect command for bot {bot_id}"


def _detect_world_state():
    """
    Build a complete world matrix by combining known bot positions with
    LLM-detected obstacles from a screenshot.

    Returns:
        {
            "matrix": np.array (64x64) — 0=free, 1=bot, 2=obstacle,
            "bots": [{"pos": (r,c), "orientation": float}]
        }
    """
    bots = _get_bots()
    screenshot_bytes = _get_screenshot()

    # Build partial matrix with bot positions filled in
    partial = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for bot in bots:
        r, c = bot["pos"]
        partial[r, c] = CELL_BOT

    partial_json = json.dumps(partial.tolist())
    screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

    # Send screenshot + partial matrix to GPT-4o for obstacle detection
    response = oai.client.responses.create(
        model="gpt-4o",
        instructions=DETECT_OBSTACLES_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{screenshot_b64}",
                    },
                    {
                        "type": "input_text",
                        "text": f"Partial matrix (bots marked as 1):\n{partial_json}",
                    },
                ],
            }
        ],
    )

    raw = response.output_text

    # Parse the returned matrix
    try:
        start = raw.index("[")
        end = raw.rindex("]") + 1
        matrix = np.array(json.loads(raw[start:end]), dtype=int)
    except (ValueError, json.JSONDecodeError):
        matrix = partial

    # Re-stamp bot positions in case LLM missed them
    for bot in bots:
        r, c = bot["pos"]
        matrix[r, c] = CELL_BOT

    return {
        "matrix": matrix,
        "bots": bots,
    }
