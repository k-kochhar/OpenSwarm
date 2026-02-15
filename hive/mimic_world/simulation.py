"""
Mimic World simulation — 50 bots arranging into shapes.

Standalone process — communicates with main.py via files:
    Writes:  files/mimic_state.json      (all bot positions, every frame)
             files/mimic_screenshot.png   (grid image, every ~500ms)
    Reads:   files/mimic_commands.json    (move commands from actions.py)

Controls:
    1-5     — load shape preset (circle, square, triangle, star, grid)
    R       — randomize obstacles
    Enter   — submit typed command as a task
    Esc     — quit
"""

import json
import math
import sys
import random
from pathlib import Path

import numpy as np
import pygame

GRID_SIZE = 64
CELL_PX = 10
GRID_PX = GRID_SIZE * CELL_PX
INPUT_HEIGHT = 36
WINDOW_W = GRID_PX
WINDOW_H = GRID_PX + INPUT_HEIGHT
FPS = 30
MOVE_DELAY_MS = 10
SCREENSHOT_INTERVAL_MS = 500
NUM_BOTS = 100

COLOR_FREE = (30, 30, 30)
COLOR_OBSTACLE = (80, 80, 80)
COLOR_TARGET_CELL = (60, 30, 30)
COLOR_INPUT_BG = (20, 20, 20)
COLOR_INPUT_BORDER = (80, 80, 80)
COLOR_INPUT_TEXT = (220, 220, 220)
COLOR_INPUT_HINT = (100, 100, 100)
COLOR_STATUS = (200, 200, 200)

# IPC file paths
FILES_DIR = Path(__file__).parent.parent / "files"
STATE_PATH = FILES_DIR / "mimic_state.json"
SCREENSHOT_PATH = FILES_DIR / "mimic_screenshot.png"
COMMANDS_PATH = FILES_DIR / "mimic_commands.json"
TASKS_PATH = FILES_DIR / "tasks.json"


def _hsv_to_rgb(h, s, v):
    """Convert HSV (0-360, 0-1, 0-1) to RGB (0-255)."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def bot_color(idx):
    """Generate a unique color per bot using golden ratio hue spacing."""
    hue = (idx * 137.508) % 360
    return _hsv_to_rgb(hue, 0.7, 0.9)


# --- Shape generators ---

def shape_circle(cx=32, cy=32, radius=20, n=NUM_BOTS):
    """Generate n positions in a circle."""
    positions = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        r = int(cy + radius * math.sin(angle))
        c = int(cx + radius * math.cos(angle))
        r = max(0, min(GRID_SIZE - 1, r))
        c = max(0, min(GRID_SIZE - 1, c))
        positions.append((r, c))
    return positions


def shape_square(cx=32, cy=32, side=30, n=NUM_BOTS):
    """Generate n positions along a square perimeter."""
    perimeter = side * 4
    positions = []
    for i in range(n):
        t = (i / n) * perimeter
        if t < side:
            r, c = cy - side // 2, cx - side // 2 + int(t)
        elif t < side * 2:
            r, c = cy - side // 2 + int(t - side), cx + side // 2
        elif t < side * 3:
            r, c = cy + side // 2, cx + side // 2 - int(t - side * 2)
        else:
            r, c = cy + side // 2 - int(t - side * 3), cx - side // 2
        r = max(0, min(GRID_SIZE - 1, r))
        c = max(0, min(GRID_SIZE - 1, c))
        positions.append((r, c))
    return positions


def shape_triangle(cx=32, cy=32, size=28, n=NUM_BOTS):
    """Generate n positions along a triangle."""
    # Equilateral triangle vertices
    v0 = (cy - int(size * 0.577), cx)
    v1 = (cy + int(size * 0.289), cx - size // 2)
    v2 = (cy + int(size * 0.289), cx + size // 2)
    verts = [v0, v1, v2]
    perimeter_points = []
    for i in range(3):
        r1, c1 = verts[i]
        r2, c2 = verts[(i + 1) % 3]
        seg_n = n // 3 + (1 if i < n % 3 else 0)
        for j in range(seg_n):
            t = j / seg_n
            r = int(r1 + (r2 - r1) * t)
            c = int(c1 + (c2 - c1) * t)
            r = max(0, min(GRID_SIZE - 1, r))
            c = max(0, min(GRID_SIZE - 1, c))
            perimeter_points.append((r, c))
    return perimeter_points[:n]


def shape_star(cx=32, cy=32, outer=25, inner=12, points=5, n=NUM_BOTS):
    """Generate n positions along a star shape."""
    verts = []
    for i in range(points * 2):
        angle = math.pi / 2 + 2 * math.pi * i / (points * 2)
        rad = outer if i % 2 == 0 else inner
        r = int(cy - rad * math.cos(angle))
        c = int(cx + rad * math.sin(angle))
        verts.append((r, c))
    positions = []
    total_verts = len(verts)
    per_seg = n // total_verts
    extra = n % total_verts
    for i in range(total_verts):
        r1, c1 = verts[i]
        r2, c2 = verts[(i + 1) % total_verts]
        seg_n = per_seg + (1 if i < extra else 0)
        for j in range(seg_n):
            t = j / max(seg_n, 1)
            r = int(r1 + (r2 - r1) * t)
            c = int(c1 + (c2 - c1) * t)
            r = max(0, min(GRID_SIZE - 1, r))
            c = max(0, min(GRID_SIZE - 1, c))
            positions.append((r, c))
    return positions[:n]


def shape_grid_formation(cx=32, cy=32, spacing=4, n=NUM_BOTS):
    """Generate n positions in a grid pattern."""
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    start_r = cy - (rows * spacing) // 2
    start_c = cx - (cols * spacing) // 2
    positions = []
    for i in range(n):
        r = start_r + (i // cols) * spacing
        c = start_c + (i % cols) * spacing
        r = max(0, min(GRID_SIZE - 1, r))
        c = max(0, min(GRID_SIZE - 1, c))
        positions.append((r, c))
    return positions


SHAPES = {
    "circle": shape_circle,
    "square": shape_square,
    "triangle": shape_triangle,
    "star": shape_star,
    "grid": shape_grid_formation,
}

SHAPE_KEYS = {
    pygame.K_1: "circle",
    pygame.K_2: "square",
    pygame.K_3: "triangle",
    pygame.K_4: "star",
    pygame.K_5: "grid",
}


# --- Grid & bot helpers ---

def random_grid(obstacle_pct=0.10):
    """Generate a grid with obstacle clusters."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    target = int(GRID_SIZE * GRID_SIZE * obstacle_pct)
    filled = 0

    n_rects = random.randint(4, 10)
    for _ in range(n_rects):
        if filled >= target:
            break
        r = random.randint(0, GRID_SIZE - 1)
        c = random.randint(0, GRID_SIZE - 1)
        h = random.randint(2, 8)
        w = random.randint(1, 3) if random.random() < 0.5 else random.randint(2, 8)
        for dr in range(h):
            for dc in range(w):
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr, nc] == 0:
                    grid[nr, nc] = 1
                    filled += 1

    return grid


def random_free_cell(grid, exclude=set()):
    while True:
        r, c = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        if grid[r, c] == 0 and (r, c) not in exclude:
            return (r, c)


def make_bots(grid, n=NUM_BOTS):
    bots = []
    exclude = set()
    for i in range(n):
        pos = random_free_cell(grid, exclude)
        exclude.add(pos)
        bots.append({
            "pos": pos,
            "orientation": random.uniform(0, 2 * math.pi),
            "target": None,
            "path": [],
            "path_idx": 0,
            "last_move": 0,
        })
    return bots


def _compute_orientation(prev, curr):
    dr = curr[0] - prev[0]
    dc = curr[1] - prev[1]
    if dr == 0 and dc == 0:
        return None
    return math.atan2(dr, dc)


# --- IPC ---

def _write_state(bots, grid, target_shape=None):
    data = {
        "bots": [
            {"pos": list(b["pos"]), "orientation": round(b["orientation"], 4)}
            for b in bots
        ],
        "grid": grid.tolist(),
        "num_bots": len(bots),
    }
    if target_shape:
        data["target_shape"] = target_shape
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


# --- Drawing ---

def draw(screen, font, grid, bots, target_positions, input_text, shape_name):
    # Grid
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rect = pygame.Rect(c * CELL_PX, r * CELL_PX, CELL_PX, CELL_PX)
            if grid[r, c] == 1:
                color = COLOR_OBSTACLE
            else:
                color = COLOR_FREE
            pygame.draw.rect(screen, color, rect)

    # Target shape positions
    if target_positions:
        for (r, c) in target_positions:
            rect = pygame.Rect(c * CELL_PX, r * CELL_PX, CELL_PX, CELL_PX)
            pygame.draw.rect(screen, COLOR_TARGET_CELL, rect)

    # Bot paths (faint)
    for i, bot in enumerate(bots):
        if bot["path"] and bot["path_idx"] < len(bot["path"]):
            color = bot_color(i)
            faint = (color[0] // 4, color[1] // 4, color[2] // 4)
            for (r, c) in bot["path"][bot["path_idx"]:]:
                rect = pygame.Rect(c * CELL_PX, r * CELL_PX, CELL_PX, CELL_PX)
                pygame.draw.rect(screen, faint, rect)

    # Bots
    for i, bot in enumerate(bots):
        br, bc = bot["pos"]
        bx = bc * CELL_PX + CELL_PX // 2
        by = br * CELL_PX + CELL_PX // 2
        color = bot_color(i)
        pygame.draw.circle(screen, color, (bx, by), CELL_PX // 2)
        dx = int(math.cos(bot["orientation"]) * CELL_PX * 0.5)
        dy = int(math.sin(bot["orientation"]) * CELL_PX * 0.5)
        pygame.draw.line(screen, (255, 255, 255), (bx, by), (bx + dx, by + dy), 1)

    # Status
    moving = sum(1 for b in bots if b["path"] and b["path_idx"] < len(b["path"]))
    status = f"Bots: {len(bots)} | Moving: {moving}"
    if shape_name:
        status += f" | Shape: {shape_name}"
    status_surface = font.render(status, True, COLOR_STATUS)
    screen.blit(status_surface, (8, 6))

    # Input bar
    input_rect = pygame.Rect(0, GRID_PX, WINDOW_W, INPUT_HEIGHT)
    pygame.draw.rect(screen, COLOR_INPUT_BG, input_rect)
    pygame.draw.line(screen, COLOR_INPUT_BORDER, (0, GRID_PX), (WINDOW_W, GRID_PX))

    if input_text:
        text_surface = font.render(f"> {input_text}", True, COLOR_INPUT_TEXT)
    else:
        text_surface = font.render("> type command... (1-5: shapes, R: regen)", True, COLOR_INPUT_HINT)
    screen.blit(text_surface, (8, GRID_PX + 8))

    pygame.display.flip()


# --- Main ---

def main():
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    COMMANDS_PATH.write_text("[]")

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("mimic_world — 100 bots")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo", 14) or pygame.font.SysFont(None, 16)

    grid = random_grid()
    bots = make_bots(grid)
    target_positions = []
    shape_name = None
    last_screenshot = 0
    input_text = ""

    _write_state(bots, grid)

    print(f"[sim] Running. {NUM_BOTS} bots. Keys: 1=circle 2=square 3=triangle 4=star 5=grid R=regen")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key in SHAPE_KEYS and not input_text:
                    shape_name = SHAPE_KEYS[event.key]
                    target_positions = SHAPES[shape_name]()
                    # Ensure targets don't land on obstacles
                    target_positions = [
                        (r, c) if grid[r, c] == 0
                        else random_free_cell(grid, set(target_positions))
                        for r, c in target_positions
                    ]
                    _write_state(bots, grid, shape_name)
                    print(f"[sim] Shape set: {shape_name}")

                elif event.key == pygame.K_r and not input_text:
                    grid = random_grid()
                    bots = make_bots(grid)
                    target_positions = []
                    shape_name = None
                    _write_state(bots, grid)
                    print("[sim] Regenerated")

                elif event.key == pygame.K_RETURN:
                    if input_text.strip():
                        _add_task(input_text.strip())
                        print(f"[sim] Task added: {input_text.strip()}")
                        input_text = ""

                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]

                elif event.unicode and event.unicode.isprintable():
                    input_text += event.unicode

        # Process commands from actions.py
        commands = _read_commands()
        for cmd in commands:
            action = cmd.get("action")
            bot_idx = cmd.get("bot", 0)
            if bot_idx < 0 or bot_idx >= len(bots):
                continue
            bot = bots[bot_idx]

            if action == "move_to":
                tr, tc = cmd["target"]
                tr, tc = int(tr), int(tc)
                if 0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE and grid[tr, tc] == 0:
                    bot["target"] = (tr, tc)
                    path = cmd.get("path")
                    if path:
                        bot["path"] = [tuple(p) for p in path]
                        bot["path_idx"] = 1
                    else:
                        bot["path"] = []
                        bot["path_idx"] = 0

        # Animate bots
        now = pygame.time.get_ticks()
        state_changed = False
        for bot in bots:
            if bot["path"] and bot["path_idx"] < len(bot["path"]) and now - bot["last_move"] >= MOVE_DELAY_MS:
                prev = bot["pos"]
                bot["pos"] = bot["path"][bot["path_idx"]]
                bot["path_idx"] += 1
                bot["last_move"] = now
                state_changed = True

                orient = _compute_orientation(prev, bot["pos"])
                if orient is not None:
                    bot["orientation"] = orient

                if bot["path_idx"] >= len(bot["path"]):
                    bot["target"] = None
                    bot["path"] = []

        if state_changed:
            _write_state(bots, grid, shape_name)

        if now - last_screenshot >= SCREENSHOT_INTERVAL_MS:
            _write_screenshot(screen)
            last_screenshot = now

        draw(screen, font, grid, bots, target_positions, input_text, shape_name)
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
