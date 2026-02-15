"""
Live simulation of the move_world.

Standalone process — communicates with main.py via files:
    Writes:  files/sim_state.json      (bots + coins, every frame)
             files/sim_screenshot.png   (grid image, every ~500ms)
    Reads:   files/sim_commands.json    (move/collect commands from main.py)

Controls:
    Left click  — set target for nearest bot
    R           — randomize obstacles
    Enter       — submit typed command as a task
    Esc         — quit
"""

import json
import math
import sys
import random
from pathlib import Path

import numpy as np
import pygame

from pathfinder import NeuralPathfinder

GRID_SIZE = 64
CELL_PX = 10
GRID_PX = GRID_SIZE * CELL_PX
INPUT_HEIGHT = 36
WINDOW_W = GRID_PX
WINDOW_H = GRID_PX + INPUT_HEIGHT
FPS = 30
MOVE_DELAY_MS = 60
SCREENSHOT_INTERVAL_MS = 500
NUM_COINS = 10
NUM_BOTS = 2

COLOR_FREE = (30, 30, 30)
COLOR_OBSTACLE = (80, 80, 80)
COLOR_VISITED = (45, 45, 55)
COLOR_COIN = (255, 210, 50)
COLOR_COIN_EDGE = (200, 160, 20)
COLOR_INPUT_BG = (20, 20, 20)
COLOR_INPUT_BORDER = (80, 80, 80)
COLOR_INPUT_TEXT = (220, 220, 220)
COLOR_INPUT_HINT = (100, 100, 100)
COLOR_SCORE = (255, 210, 50)

# Per-bot colors: [body, direction_line, path, target]
BOT_COLORS = [
    ((0, 200, 100), (255, 255, 255), (60, 160, 80), (220, 60, 60)),    # green
    ((100, 150, 255), (255, 255, 255), (60, 100, 220), (200, 80, 200)), # blue
]

# IPC file paths
FILES_DIR = Path(__file__).parent.parent / "files"
STATE_PATH = FILES_DIR / "sim_state.json"
SCREENSHOT_PATH = FILES_DIR / "sim_screenshot.png"
COMMANDS_PATH = FILES_DIR / "sim_commands.json"
TASKS_PATH = FILES_DIR / "tasks.json"

# Pathfinder (simulation does its own pathfinding)
_pf = NeuralPathfinder(str(Path(__file__).parent / "checkpoints" / "best_model.pt"))


def _compute_orientation(prev_pos, curr_pos):
    dr = curr_pos[0] - prev_pos[0]
    dc = curr_pos[1] - prev_pos[1]
    if dr == 0 and dc == 0:
        return None
    return math.atan2(dr, dc)


def _write_state(bots, coins, score):
    data = {
        "bots": [
            {"pos": list(b["pos"]), "orientation": round(b["orientation"], 4)}
            for b in bots
        ],
        "coins": [list(c) for c in coins],
        "score": score,
    }
    STATE_PATH.write_text(json.dumps(data))


def _write_screenshot(screen):
    grid_surface = screen.subsurface(pygame.Rect(0, 0, GRID_PX, GRID_PX))
    pygame.image.save(grid_surface, str(SCREENSHOT_PATH))


def _read_commands():
    if not COMMANDS_PATH.exists():
        return []
    text = COMMANDS_PATH.read_text().strip()
    if not text:
        return []
    try:
        commands = json.loads(text)
    except json.JSONDecodeError:
        return []
    COMMANDS_PATH.write_text("[]")
    return commands


def _add_task(task_text):
    tasks = []
    if TASKS_PATH.exists():
        text = TASKS_PATH.read_text().strip()
        if text:
            try:
                tasks = json.loads(text)
            except json.JSONDecodeError:
                tasks = []
    tasks.append(task_text)
    TASKS_PATH.write_text(json.dumps(tasks, indent=2))


def random_grid(obstacle_pct=0.15):
    """Generate a grid with contiguous obstacle clusters that look like walls/rooms."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    target = int(GRID_SIZE * GRID_SIZE * obstacle_pct)
    filled = 0

    n_rects = random.randint(6, 14)
    for _ in range(n_rects):
        if filled >= target:
            break
        r = random.randint(0, GRID_SIZE - 1)
        c = random.randint(0, GRID_SIZE - 1)
        if random.random() < 0.5:
            h = random.randint(2, 12)
            w = random.randint(1, 3)
        else:
            h = random.randint(1, 3)
            w = random.randint(2, 12)
        for dr in range(h):
            for dc in range(w):
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr, nc] == 0:
                    grid[nr, nc] = 1
                    filled += 1

    n_blobs = random.randint(4, 10)
    for _ in range(n_blobs):
        if filled >= target:
            break
        sr = random.randint(2, GRID_SIZE - 3)
        sc = random.randint(2, GRID_SIZE - 3)
        blob_size = random.randint(5, 25)
        frontier = [(sr, sc)]
        for _ in range(blob_size):
            if not frontier or filled >= target:
                break
            idx = random.randint(0, len(frontier) - 1)
            r, c = frontier.pop(idx)
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and grid[r, c] == 0:
                grid[r, c] = 1
                filled += 1
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr, nc] == 0:
                        frontier.append((nr, nc))

    return grid


def random_free_cell(grid, exclude=set()):
    while True:
        r, c = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        if grid[r, c] == 0 and (r, c) not in exclude:
            return (r, c)


def spawn_coins(grid, bot_positions, n=NUM_COINS):
    coins = set()
    exclude = set(bot_positions)
    for _ in range(n):
        pos = random_free_cell(grid, exclude=exclude | coins)
        coins.add(pos)
    return coins


def make_bot(grid, exclude):
    pos = random_free_cell(grid, exclude=exclude)
    return {
        "pos": pos,
        "orientation": 0.0,
        "target": None,
        "path": [],
        "path_idx": 0,
        "visited": set(),
        "last_move": 0,
    }


def draw(screen, font, grid, bots, coins, score, input_text):
    # --- Grid ---
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rect = pygame.Rect(c * CELL_PX, r * CELL_PX, CELL_PX, CELL_PX)
            if grid[r, c] == 1:
                color = COLOR_OBSTACLE
            else:
                # Check if any bot visited this cell
                visited = any((r, c) in b["visited"] for b in bots)
                color = COLOR_VISITED if visited else COLOR_FREE
            pygame.draw.rect(screen, color, rect)

    # Draw coins
    for (cr, cc) in coins:
        cx = cc * CELL_PX + CELL_PX // 2
        cy = cr * CELL_PX + CELL_PX // 2
        pygame.draw.circle(screen, COLOR_COIN, (cx, cy), CELL_PX // 3 + 1)
        pygame.draw.circle(screen, COLOR_COIN_EDGE, (cx, cy), CELL_PX // 3 + 1, 1)

    # Draw each bot's path, target, and body
    for i, bot in enumerate(bots):
        color_body, color_dir, color_path, color_target = BOT_COLORS[i % len(BOT_COLORS)]

        # Path
        if bot["path"] and bot["path_idx"] < len(bot["path"]):
            for (r, c) in bot["path"][bot["path_idx"]:]:
                rect = pygame.Rect(c * CELL_PX, r * CELL_PX, CELL_PX, CELL_PX)
                pygame.draw.rect(screen, color_path, rect)

        # Target
        if bot["target"]:
            tr, tc = bot["target"]
            rect = pygame.Rect(tc * CELL_PX, tr * CELL_PX, CELL_PX, CELL_PX)
            pygame.draw.rect(screen, color_target, rect)

        # Bot body
        br, bc = bot["pos"]
        bx = bc * CELL_PX + CELL_PX // 2
        by = br * CELL_PX + CELL_PX // 2
        pygame.draw.circle(screen, color_body, (bx, by), CELL_PX // 2)
        dx = int(math.cos(bot["orientation"]) * CELL_PX * 0.6)
        dy = int(math.sin(bot["orientation"]) * CELL_PX * 0.6)
        pygame.draw.line(screen, color_dir, (bx, by), (bx + dx, by + dy), 2)

        # Bot label
        label = font.render(str(i), True, (0, 0, 0))
        screen.blit(label, (bx - label.get_width() // 2, by - label.get_height() // 2))

    # Score display (top-right)
    score_surface = font.render(f"Coins: {score}", True, COLOR_SCORE)
    screen.blit(score_surface, (GRID_PX - score_surface.get_width() - 8, 6))

    # --- Input bar ---
    input_rect = pygame.Rect(0, GRID_PX, WINDOW_W, INPUT_HEIGHT)
    pygame.draw.rect(screen, COLOR_INPUT_BG, input_rect)
    pygame.draw.line(screen, COLOR_INPUT_BORDER, (0, GRID_PX), (WINDOW_W, GRID_PX))

    if input_text:
        text_surface = font.render(f"> {input_text}", True, COLOR_INPUT_TEXT)
    else:
        text_surface = font.render("> type a command...", True, COLOR_INPUT_HINT)

    screen.blit(text_surface, (8, GRID_PX + 8))

    pygame.display.flip()


def main():
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    COMMANDS_PATH.write_text("[]")

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("move_world")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo", 16) or pygame.font.SysFont(None, 18)

    grid = random_grid()

    # Spawn bots
    bots = []
    for _ in range(NUM_BOTS):
        exclude = {b["pos"] for b in bots}
        bots.append(make_bot(grid, exclude))

    coins = spawn_coins(grid, [b["pos"] for b in bots])
    score = 0
    last_screenshot = 0
    input_text = ""

    _write_state(bots, coins, score)

    print(f"[sim] Running. IPC via {FILES_DIR}")
    print(f"[sim] {NUM_BOTS} bots, {len(coins)} coins. Click to move nearest bot, type commands, R to randomize")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r and not input_text:
                    grid = random_grid()
                    bots = []
                    for _ in range(NUM_BOTS):
                        exclude = {b["pos"] for b in bots}
                        bots.append(make_bot(grid, exclude))
                    coins = spawn_coins(grid, [b["pos"] for b in bots])
                    score = 0
                    _write_state(bots, coins, score)
                elif event.key == pygame.K_RETURN:
                    if input_text.strip():
                        _add_task(input_text.strip())
                        print(f"[sim] Task added: {input_text.strip()}")
                        input_text = ""
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                elif event.unicode and event.unicode.isprintable():
                    input_text += event.unicode

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if my < GRID_PX:
                    tc, tr = mx // CELL_PX, my // CELL_PX
                    if 0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE and grid[tr, tc] == 0:
                        # Find nearest bot by manhattan distance
                        nearest = min(
                            range(len(bots)),
                            key=lambda i: abs(bots[i]["pos"][0] - tr) + abs(bots[i]["pos"][1] - tc),
                        )
                        bot = bots[nearest]
                        bot["target"] = (tr, tc)
                        result = _pf.find_path(grid, start=bot["pos"], goal=(tr, tc))
                        if result:
                            bot["path"] = result
                            bot["path_idx"] = 1
                        else:
                            bot["path"] = []
                            bot["path_idx"] = 0
                        bot["visited"] = set()

        # Check for commands from main.py
        commands = _read_commands()
        for cmd in commands:
            action = cmd.get("action")
            bot_idx = cmd.get("bot", 0)
            if bot_idx < 0 or bot_idx >= len(bots):
                print(f"[sim] Invalid bot index {bot_idx}, skipping")
                continue
            bot = bots[bot_idx]

            if action == "move_to":
                tr, tc = cmd["target"]
                tr, tc = int(tr), int(tc)
                if 0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE and grid[tr, tc] == 0:
                    bot["target"] = (tr, tc)
                    result = _pf.find_path(grid, start=bot["pos"], goal=(tr, tc))
                    if result:
                        bot["path"] = result
                        bot["path_idx"] = 1
                        print(f"[sim] Bot {bot_idx}: moving to ({tr}, {tc}) — {len(result)} steps")
                    else:
                        print(f"[sim] Bot {bot_idx}: no path to ({tr}, {tc})")
                        bot["path"] = []
                        bot["path_idx"] = 0
                    bot["visited"] = set()
            elif action == "collect":
                if bot["pos"] in coins:
                    coins.discard(bot["pos"])
                    score += 1
                    print(f"[sim] Bot {bot_idx}: coin collected at {bot['pos']}! Score: {score}")
                    _write_state(bots, coins, score)
                else:
                    print(f"[sim] Bot {bot_idx}: no coin at {bot['pos']}")

        # Animate all bots along their paths
        now = pygame.time.get_ticks()
        state_changed = False
        for i, bot in enumerate(bots):
            if bot["path"] and bot["path_idx"] < len(bot["path"]) and now - bot["last_move"] >= MOVE_DELAY_MS:
                prev_pos = bot["pos"]
                bot["visited"].add(bot["pos"])
                bot["pos"] = bot["path"][bot["path_idx"]]
                bot["path_idx"] += 1
                bot["last_move"] = now
                state_changed = True

                new_orient = _compute_orientation(prev_pos, bot["pos"])
                if new_orient is not None:
                    bot["orientation"] = new_orient

                if bot["path_idx"] >= len(bot["path"]):
                    bot["target"] = None
                    bot["path"] = []

        if state_changed:
            _write_state(bots, coins, score)

        # Save screenshot periodically
        if now - last_screenshot >= SCREENSHOT_INTERVAL_MS:
            _write_screenshot(screen)
            last_screenshot = now

        draw(screen, font, grid, bots, coins, score, input_text)
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
