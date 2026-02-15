"""
Fire World Simulation

Standalone process -- communicates with main.py via files:
    Writes:  files/sim_state.json      (bots, fires, stats, every frame)
             files/sim_screenshot.png   (grid image, every ~500ms)
    Reads:   files/sim_commands.json    (move/extinguish commands from main.py)

The world features:
    - Random fire clusters that spawn periodically
    - Two firefighting bots that navigate and extinguish fires
    - Visual smoke effects when extinguishing fires
    - Pathfinding through obstacles and around fires

Controls:
    Left click  -- set target for nearest bot
    E           -- extinguish fires near clicked bot
    R           -- reset world (new grid, bots, and fires)
    F           -- spawn fire at cursor
    Enter       -- submit typed command as a task
    Esc         -- quit
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
NUM_BOTS = 3

# Fire spawning parameters
FIRE_SPAWN_INTERVAL = 8000  # ms between automatic fire spawns
MIN_CLUSTER_SIZE = 2
MAX_CLUSTER_SIZE = 8
FIRE_SPREAD_CHANCE = 0.02  # chance per frame for fire to spread to adjacent cell
FIRE_SPREAD_INTERVAL_MS = 1000  # minimum time between spread attempts

# Colors
COLOR_FREE = (30, 30, 30)
COLOR_OBSTACLE = (80, 80, 80)
COLOR_VISITED = (45, 45, 55)
COLOR_FIRE = (255, 80, 20)
COLOR_FIRE_CORE = (255, 200, 50)
COLOR_INPUT_BG = (20, 20, 20)
COLOR_INPUT_BORDER = (80, 80, 80)
COLOR_INPUT_TEXT = (220, 220, 220)
COLOR_INPUT_HINT = (100, 100, 100)
COLOR_STATS = (100, 200, 255)

# Per-bot colors: [body, direction_line, path, target]
BOT_COLORS = [
    ((0, 200, 255), (255, 255, 255), (60, 160, 220), (220, 60, 60)),    # cyan
    ((255, 150, 0), (255, 255, 255), (220, 120, 40), (200, 80, 200)),   # orange
    ((150, 255, 100), (255, 255, 255), (100, 200, 80), (220, 60, 180)), # green
]

# Smoke particle effect
class SmokeParticle:
    def __init__(self, r, c):
        self.x = c * CELL_PX + CELL_PX // 2 + random.randint(-3, 3)
        self.y = r * CELL_PX + CELL_PX // 2 + random.randint(-3, 3)
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-1.5, -0.5)
        self.life = random.randint(30, 60)
        self.max_life = self.life
        self.size = random.randint(3, 7)
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        return self.life > 0
    
    def draw(self, screen):
        alpha = int(255 * (self.life / self.max_life))
        color = (255, 255, 255, alpha)
        # Draw white smoke
        s = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, color, (self.size, self.size), self.size)
        screen.blit(s, (int(self.x - self.size), int(self.y - self.size)))


# IPC file paths
FILES_DIR = Path(__file__).parent.parent / "files"
STATE_PATH = FILES_DIR / "sim_state.json"
SCREENSHOT_PATH = FILES_DIR / "sim_screenshot.png"
COMMANDS_PATH = FILES_DIR / "sim_commands.json"
TASKS_PATH = FILES_DIR / "tasks.json"

# Pathfinder
_pf = NeuralPathfinder(str(Path(__file__).parent / "checkpoints" / "best_model.pt"))


def _compute_orientation(prev_pos, curr_pos):
    """Compute orientation angle from movement direction."""
    dr = curr_pos[0] - prev_pos[0]
    dc = curr_pos[1] - prev_pos[1]
    if dr == 0 and dc == 0:
        return None
    return math.atan2(dr, dc)


def _write_state(bots, fires, fire_clusters, stats):
    """Write current state to IPC file."""
    data = {
        "bots": [
            {"pos": list(b["pos"]), "orientation": round(b["orientation"], 4)}
            for b in bots
        ],
        "fires": [list(f) for f in fires],
        "fire_clusters": [[list(cell) for cell in cluster] for cluster in fire_clusters],
        "active_bots": [i for i, b in enumerate(bots) if b["path"] and b["path_idx"] < len(b["path"])],
        "stats": stats,
    }
    STATE_PATH.write_text(json.dumps(data))


def _write_screenshot(screen):
    """Save screenshot of grid area to file."""
    grid_surface = screen.subsurface(pygame.Rect(0, 0, GRID_PX, GRID_PX))
    pygame.image.save(grid_surface, str(SCREENSHOT_PATH))


def _read_commands():
    """Read and clear command queue from IPC file."""
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
    """Add a task to the task queue."""
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


def random_grid(obstacle_pct=0.10):
    """Generate a grid with some obstacles (fewer than move_world to allow fire spread)."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    target = int(GRID_SIZE * GRID_SIZE * obstacle_pct)
    filled = 0

    # Create wall-like structures
    n_rects = random.randint(4, 8)
    for _ in range(n_rects):
        if filled >= target:
            break
        r = random.randint(0, GRID_SIZE - 1)
        c = random.randint(0, GRID_SIZE - 1)
        if random.random() < 0.5:
            h = random.randint(2, 10)
            w = random.randint(1, 2)
        else:
            h = random.randint(1, 2)
            w = random.randint(2, 10)
        for dr in range(h):
            for dc in range(w):
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr, nc] == 0:
                    grid[nr, nc] = 1
                    filled += 1

    return grid


def random_free_cell(grid, fires, exclude=set()):
    """Find a random free cell (not obstacle, not fire, not excluded)."""
    attempts = 0
    while attempts < 1000:
        r, c = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        if grid[r, c] == 0 and (r, c) not in fires and (r, c) not in exclude:
            return (r, c)
        attempts += 1
    return None


def spawn_fire_cluster(grid, fires, exclude=set()):
    """
    Spawn a cluster of fire cells at a random location.
    Returns set of new fire cells.
    """
    start = random_free_cell(grid, fires, exclude)
    if start is None:
        return set()
    
    cluster_size = random.randint(MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE)
    cluster = {start}
    frontier = [start]
    
    for _ in range(cluster_size - 1):
        if not frontier:
            break
        # Pick a random cell from frontier
        cell = random.choice(frontier)
        r, c = cell
        
        # Try adjacent cells
        neighbors = [(r+dr, c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]]
        random.shuffle(neighbors)
        
        for nr, nc in neighbors:
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                if grid[nr, nc] == 0 and (nr, nc) not in fires and (nr, nc) not in cluster:
                    cluster.add((nr, nc))
                    frontier.append((nr, nc))
                    break
    
    return cluster


def find_fire_clusters(fires):
    """
    Group fire cells into connected clusters using flood fill.
    Returns list of clusters, each cluster is a set of (r, c) positions.
    """
    unvisited = set(fires)
    clusters = []
    
    while unvisited:
        start = unvisited.pop()
        cluster = {start}
        frontier = [start]
        
        while frontier:
            r, c = frontier.pop()
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in unvisited:
                    unvisited.remove((nr, nc))
                    cluster.add((nr, nc))
                    frontier.append((nr, nc))
        
        clusters.append(cluster)
    
    return clusters


def spread_fire(grid, fires):
    """
    Attempt to spread fire to adjacent free cells.
    Returns set of new fire cells.
    """
    new_fires = set()
    for r, c in fires:
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                if grid[nr, nc] == 0 and (nr, nc) not in fires and random.random() < FIRE_SPREAD_CHANCE:
                    new_fires.add((nr, nc))
    return new_fires


def extinguish_cluster(bot_pos, fire_clusters):
    """
    Find and extinguish the fire cluster adjacent to bot_pos.
    Returns (extinguished_cluster, remaining_fires_set).
    """
    br, bc = bot_pos
    
    for cluster in fire_clusters:
        # Check if any cell in cluster is adjacent to bot
        for fr, fc in cluster:
            if abs(fr - br) <= 1 and abs(fc - bc) <= 1 and (fr != br or fc != bc):
                # Found adjacent cluster - extinguish it
                return cluster
    
    return None


def make_bot(grid, fires, exclude):
    """Create a bot at a random free position."""
    pos = random_free_cell(grid, fires, exclude=exclude)
    if pos is None:
        # Fallback: try to find any free cell
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r, c] == 0 and (r, c) not in fires and (r, c) not in exclude:
                    pos = (r, c)
                    break
            if pos is not None:
                break
        if pos is None:
            pos = (GRID_SIZE // 2, GRID_SIZE // 2)  # last resort fallback
    return {
        "pos": pos,
        "orientation": 0.0,
        "target": None,
        "path": [],
        "path_idx": 0,
        "visited": set(),
        "last_move": 0,
    }


def draw(screen, font, grid, bots, fires, smoke_particles, stats, input_text):
    """Render the entire scene."""
    # Grid background
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rect = pygame.Rect(c * CELL_PX, r * CELL_PX, CELL_PX, CELL_PX)
            if grid[r, c] == 1:
                color = COLOR_OBSTACLE
            else:
                visited = any((r, c) in b["visited"] for b in bots)
                color = COLOR_VISITED if visited else COLOR_FREE
            pygame.draw.rect(screen, color, rect)

    # Draw fires
    for (fr, fc) in fires:
        rect = pygame.Rect(fc * CELL_PX, fr * CELL_PX, CELL_PX, CELL_PX)
        pygame.draw.rect(screen, COLOR_FIRE, rect)
        # Inner glow
        inner_rect = pygame.Rect(fc * CELL_PX + 2, fr * CELL_PX + 2, CELL_PX - 4, CELL_PX - 4)
        pygame.draw.rect(screen, COLOR_FIRE_CORE, inner_rect)

    # Draw smoke particles
    for particle in smoke_particles:
        particle.draw(screen)

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

    # Stats display (top-left)
    stats_lines = [
        f"Fires: {stats['fires_active']}",
        f"Extinguished: {stats['cells_extinguished']}",
    ]
    y_offset = 6
    for line in stats_lines:
        stats_surface = font.render(line, True, COLOR_STATS)
        screen.blit(stats_surface, (8, y_offset))
        y_offset += 18

    # Input bar
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
    pygame.display.set_caption("fire_world")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo", 16) or pygame.font.SysFont(None, 18)

    grid = random_grid()
    fires = set()
    smoke_particles = []

    # Spawn initial fire cluster
    fires.update(spawn_fire_cluster(grid, fires))

    # Spawn bots
    bots = []
    for _ in range(NUM_BOTS):
        exclude = {b["pos"] for b in bots}
        bots.append(make_bot(grid, fires, exclude))

    stats = {
        "fires_active": len(fires),
        "cells_extinguished": 0,
    }

    last_screenshot = 0
    last_fire_spawn = pygame.time.get_ticks()
    last_fire_spread = pygame.time.get_ticks()
    input_text = ""

    fire_clusters = find_fire_clusters(fires)
    _write_state(bots, fires, fire_clusters, stats)

    print(f"[sim] Fire World running. IPC via {FILES_DIR}")
    print(f"[sim] {NUM_BOTS} bots ready. R=reset world, F=fire at cursor, E=extinguish, Click=move")

    running = True
    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r and not input_text:
                    # Reset world - regenerate grid, bots, and fires
                    grid = random_grid()
                    fires = set()
                    smoke_particles = []
                    fires.update(spawn_fire_cluster(grid, fires))
                    bots = []
                    for _ in range(NUM_BOTS):
                        exclude = {b["pos"] for b in bots}
                        bots.append(make_bot(grid, fires, exclude))
                    stats = {
                        "fires_active": len(fires),
                        "cells_extinguished": 0,
                    }
                    fire_clusters = find_fire_clusters(fires)
                    _write_state(bots, fires, fire_clusters, stats)
                    print(f"[sim] World reset: new grid, {NUM_BOTS} bots, {len(fires)} fire cells")
                elif event.key == pygame.K_e and not input_text:
                    # Manual extinguish for testing
                    mx, my = pygame.mouse.get_pos()
                    if my < GRID_PX:
                        mc, mr = mx // CELL_PX, my // CELL_PX
                        # Find nearest bot
                        nearest = min(
                            range(len(bots)),
                            key=lambda i: abs(bots[i]["pos"][0] - mr) + abs(bots[i]["pos"][1] - mc),
                        )
                        bot_pos = bots[nearest]["pos"]
                        cluster = extinguish_cluster(bot_pos, fire_clusters)
                        if cluster:
                            for cell in cluster:
                                fires.discard(cell)
                                for _ in range(5):
                                    smoke_particles.append(SmokeParticle(*cell))
                            stats["cells_extinguished"] += len(cluster)
                            print(f"[sim] Bot {nearest}: extinguished {len(cluster)} cells")
                elif event.key == pygame.K_f and not input_text:
                    # Spawn fire at cursor
                    mx, my = pygame.mouse.get_pos()
                    if my < GRID_PX:
                        mc, mr = mx // CELL_PX, my // CELL_PX
                        if 0 <= mr < GRID_SIZE and 0 <= mc < GRID_SIZE and grid[mr, mc] == 0:
                            fires.add((mr, mc))
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
                    if 0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE and grid[tr, tc] == 0 and (tr, tc) not in fires:
                        # Find nearest bot
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
                        print(f"[sim] Bot {bot_idx}: moving to ({tr}, {tc}) -- {len(result)} steps")
                    else:
                        print(f"[sim] Bot {bot_idx}: no path to ({tr}, {tc})")
                        bot["path"] = []
                        bot["path_idx"] = 0
                    bot["visited"] = set()
            
            elif action == "extinguish":
                cluster = extinguish_cluster(bot["pos"], fire_clusters)
                if cluster:
                    for cell in cluster:
                        fires.discard(cell)
                        # Create smoke particles
                        for _ in range(5):
                            smoke_particles.append(SmokeParticle(*cell))
                    stats["cells_extinguished"] += len(cluster)
                    print(f"[sim] Bot {bot_idx}: extinguished {len(cluster)} cells at cluster near {bot['pos']}")
                else:
                    print(f"[sim] Bot {bot_idx}: no fire cluster adjacent to {bot['pos']}")

        # Spawn new fire clusters periodically
        if now - last_fire_spawn >= FIRE_SPAWN_INTERVAL:
            new_fires = spawn_fire_cluster(grid, fires, exclude={b["pos"] for b in bots})
            if new_fires:
                fires.update(new_fires)
                print(f"[sim] Auto-spawned fire cluster: {len(new_fires)} cells")
            last_fire_spawn = now

        # Spread fire occasionally
        if now - last_fire_spread >= FIRE_SPREAD_INTERVAL_MS:
            new_fires = spread_fire(grid, fires)
            if new_fires:
                fires.update(new_fires)
            last_fire_spread = now

        # Animate bots along their paths
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

        # Update fire clusters
        fire_clusters = find_fire_clusters(fires)
        stats["fires_active"] = len(fires)

        # Update smoke particles
        smoke_particles = [p for p in smoke_particles if p.update()]

        if state_changed or len(smoke_particles) > 0:
            _write_state(bots, fires, fire_clusters, stats)

        # Save screenshot periodically
        if now - last_screenshot >= SCREENSHOT_INTERVAL_MS:
            _write_screenshot(screen)
            last_screenshot = now

        draw(screen, font, grid, bots, fires, smoke_particles, stats, input_text)
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
