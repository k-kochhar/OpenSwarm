"""
OpenHive main loop.

Usage:
    python main.py <world_dir>

    e.g. python main.py move_world

Run the simulation first in a separate terminal:
    cd move_world && python simulation.py

Then run main.py — it communicates with the simulation via files.

Flow:
    1. Reads init.md, runs the init prompt to generate a world document
    2. Starts a poll loop (every 3s):
       a. Refreshes world state via detect_world_state (screenshot + bots → matrix)
       b. Saves state to files/state.json
       c. Checks files/tasks.json — if tasks exist, sends to LLM which returns
          function calls. We execute those calls on the world's actions module.
    3. User can type commands at any time — they get added as tasks
"""

import json
import time
import sys
import base64
import threading
import importlib
import inspect
from pathlib import Path

from ohm import chat
from prompts import init_prompt, action_prompt

HIVE_DIR = Path(__file__).parent
TASKS_FILE = HIVE_DIR / "files" / "tasks.json"
WORLD_FILE = HIVE_DIR / "files" / "world.md"
STATE_FILE = HIVE_DIR / "files" / "state.json"
POLL_INTERVAL = 3

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

# Will be set after loading the world's modules
_actions_module = None


def load_tasks():
    if not TASKS_FILE.exists():
        return []
    text = TASKS_FILE.read_text().strip()
    if not text:
        return []
    return json.loads(text)


def save_tasks(tasks):
    TASKS_FILE.write_text(json.dumps(tasks, indent=2))


def add_task(task_text):
    tasks = load_tasks()
    tasks.append(task_text)
    save_tasks(tasks)


def get_available_actions():
    """Inspect the actions module and return a description of callable functions."""
    actions = {}
    for name, fn in inspect.getmembers(_actions_module, inspect.isfunction):
        if name.startswith("_"):
            continue
        sig = inspect.signature(fn)
        doc = fn.__doc__ or ""
        params = []
        for pname, param in sig.parameters.items():
            p = {"name": pname, "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "any"}
            if param.default != inspect.Parameter.empty:
                p["default"] = repr(param.default)
            params.append(p)
        actions[name] = {"params": params, "doc": doc.strip()}
    return actions


def refresh_state():
    """Read full state from the world (fast, no LLM call)."""
    if not hasattr(_actions_module, "_get_state"):
        return None

    try:
        state = _actions_module._get_state()
    except Exception as e:
        print(f"[loop] State refresh skipped: {e}")
        return None

    if not state:
        return None

    STATE_FILE.write_text(json.dumps(state))
    return state


def run_init(world_dir: Path):
    """Read init.md, send through the init prompt, save world document."""
    init_md = (world_dir / "init.md").read_text()
    actions_src = (world_dir / "actions.py").read_text()

    message = (
        f"{init_prompt}\n\n"
        f"--- USER INIT DOCUMENT ---\n{init_md}\n\n"
        f"--- AVAILABLE ACTIONS (code) ---\n{actions_src}"
    )

    print("[init] Generating world document...")
    world_doc = chat(DEFAULT_MODEL, message)
    WORLD_FILE.write_text(world_doc)
    print(f"[init] World document saved to {WORLD_FILE}")
    return world_doc


def _wait_for_bot_idle(bot_id, timeout=30, poll=0.5):
    """Wait until the bot finishes its current movement or timeout."""
    start = time.time()
    print(f"[exec] Bot {bot_id}: waiting for movement to finish...")
    time.sleep(1.0)
    while time.time() - start < timeout:
        try:
            state = _actions_module._get_state()
            if bot_id not in state.get("active_bots", []):
                print(f"[exec] Bot {bot_id}: movement finished")
                return True
        except Exception:
            pass
        time.sleep(poll)
    print(f"[exec] Bot {bot_id}: timeout waiting for movement")
    return False


def _run_bot_sequence(bot_id, calls):
    """Execute a sequence of calls for a single bot, waiting between moves."""
    for call in calls:
        fn_name = call.get("function")
        params = call.get("params", {})

        fn = getattr(_actions_module, fn_name, None)
        if fn is None:
            print(f"[exec] Bot {bot_id}: unknown action {fn_name} — skipping")
            continue

        # Convert list params that should be tuples (positions)
        for k, v in params.items():
            if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                params[k] = tuple(v)

        # Replace "bot" with "bot_id" for the action function
        if "bot" in params:
            params["bot_id"] = params.pop("bot")

        print(f"[exec] Bot {bot_id}: {fn_name}({params})")
        try:
            result = fn(**params)
            print(f"[exec] Bot {bot_id}: {fn_name} → {str(result)[:200]}")
        except Exception as e:
            print(f"[exec] Bot {bot_id}: {fn_name} failed: {e}")
            continue

        # Wait for movement commands to finish before next call
        if fn_name in ("move_to", "push_and_exit"):
            _wait_for_bot_idle(bot_id)


def execute_task(task, world_doc, state, available_actions):
    """
    Send a task + state to the LLM. It returns function calls to execute.

    The LLM responds with JSON:
    {
        "calls": [
            {"function": "move_to", "params": {"target_pos": [50, 60], "bot": 0}},
            ...
        ],
        "new_tasks": ["optional follow-up tasks"]
    }

    Calls are grouped by bot. Each bot's calls run sequentially (waiting
    for moves to complete), but different bots run in parallel.
    """
    actions_desc = json.dumps(available_actions, indent=2)

    state_summary = "No state available yet."
    if state:
        state_summary = json.dumps(state, indent=2)

    message = (
        f"{action_prompt}\n\n"
        f"WORLD DOCUMENT:\n{world_doc}\n\n"
        f"CURRENT STATE:\n{state_summary}\n\n"
        f"AVAILABLE ACTIONS (call these by name with params):\n{actions_desc}\n\n"
        f"TASK: {json.dumps(task)}"
    )

    # Read screenshot and encode for vision
    screenshot_b64 = None
    screenshot_path = getattr(_actions_module, "SCREENSHOT_PATH", None)
    if screenshot_path and screenshot_path.exists():
        try:
            screenshot_b64 = base64.b64encode(screenshot_path.read_bytes()).decode("utf-8")
        except Exception:
            pass

    response = chat(DEFAULT_MODEL, message, image_b64=screenshot_b64)

    # Parse LLM response
    try:
        start = response.index("{")
        end = response.rindex("}") + 1
        parsed = json.loads(response[start:end])
    except (ValueError, json.JSONDecodeError):
        print(f"[exec] Could not parse LLM response: {response[:200]}")
        return []

    # Group calls by bot — each bot's calls run sequentially,
    # but different bots run in parallel
    calls = parsed.get("calls", [])
    bot_calls = {}
    for call in calls:
        p = call.get("params", {})
        bid = p.get("bot", p.get("bot_id", 0))
        bot_calls.setdefault(bid, []).append(call)

    if len(bot_calls) == 1:
        bot_id, seq = next(iter(bot_calls.items()))
        _run_bot_sequence(bot_id, seq)
    else:
        threads = []
        for bot_id, seq in bot_calls.items():
            t = threading.Thread(target=_run_bot_sequence, args=(bot_id, seq))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    return parsed.get("new_tasks", [])


def input_thread():
    """Background thread that reads user input and adds tasks."""
    print("[input] Type a command to add a task (or 'quit' to exit):\n")
    while True:
        try:
            user_input = input("> ").strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("[input] Shutting down...")
            import os
            os._exit(0)
        add_task(user_input)
        print(f"[input] Added task: {user_input}")


def main():
    global _actions_module

    if len(sys.argv) < 2:
        print("Usage: python main.py <world_dir> [--noinit]")
        print("  e.g. python main.py move_world --noinit")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    noinit = "--noinit" in flags

    if not args:
        print("Usage: python main.py <world_dir> [--noinit]")
        sys.exit(1)

    world_dir = HIVE_DIR / args[0]
    if not world_dir.exists():
        print(f"Error: {world_dir} does not exist")
        sys.exit(1)

    # Ensure files dir exists
    TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not TASKS_FILE.exists():
        save_tasks([])

    # Load the world's actions module
    sys.path.insert(0, str(world_dir))
    _actions_module = importlib.import_module("actions")

    # Step 1: Run init (skip with --noinit)
    if noinit:
        if WORLD_FILE.exists():
            world_doc = WORLD_FILE.read_text()
            print("[init] Skipped (--noinit). Using existing world.md")
        else:
            print("[init] --noinit but no world.md found, running init anyway...")
            world_doc = run_init(world_dir)
    else:
        world_doc = run_init(world_dir)
    available_actions = get_available_actions()
    print(f"[init] Available actions: {list(available_actions.keys())}")

    # Step 2: Start user input thread
    t = threading.Thread(target=input_thread, daemon=True)
    t.start()

    # Step 3: Poll loop
    print(f"\n[loop] Running every {POLL_INTERVAL}s... (type commands below)\n")

    try:
        while True:
            # Refresh world state
            print("[loop] Refreshing state...")
            state = refresh_state()

            # Check for tasks
            tasks = load_tasks()
            if tasks:
                task = tasks.pop(0)
                print(f"[loop] Executing task: {task}")

                new_tasks = execute_task(task, world_doc, state, available_actions)

                if new_tasks:
                    tasks.extend(new_tasks)
                    print(f"[loop] Added {len(new_tasks)} new task(s)")

                save_tasks(tasks)
                print(f"[loop] {len(tasks)} task(s) remaining")
            else:
                print("[loop] No tasks.")

            print()
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[loop] Stopped.")


if __name__ == "__main__":
    main()
