"""
Neural heuristic pathfinder.

Takes a grid (obstacle map) and finds the shortest path from a bot's
current position to a target position using A* with the learned neural
heuristic.

Usage:
    from pathfinder import NeuralPathfinder

    pf = NeuralPathfinder("checkpoints/best_model.pt")
    path = pf.find_path(grid, start=(10, 5), goal=(50, 60))
    # path is a list of (row, col) tuples, or None if no path exists
"""

import heapq
import math
from pathlib import Path

import numpy as np
import torch

from model import HeuristicNetV2, HeuristicNet

_DIRS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
         (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def _astar(grid, start, goal, heuristic_fn):
    """A* search with a custom heuristic. Returns path or None."""
    h, w = grid.shape
    sr, sc = start
    gr, gc = goal

    g_cost = {(sr, sc): 0.0}
    parent = {(sr, sc): None}
    closed = set()
    heap = [(heuristic_fn(sr, sc), 0.0, sr, sc)]

    while heap:
        _f, g, r, c = heapq.heappop(heap)
        if (r, c) in closed:
            continue
        closed.add((r, c))

        if (r, c) == (gr, gc):
            path = []
            node = (gr, gc)
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path

        for dr, dc, move_cost in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                ng = g + move_cost
                if (nr, nc) not in g_cost or ng < g_cost[(nr, nc)]:
                    g_cost[(nr, nc)] = ng
                    f = ng + heuristic_fn(nr, nc)
                    parent[(nr, nc)] = (r, c)
                    heapq.heappush(heap, (f, ng, nr, nc))

    return None  # no path


class NeuralPathfinder:
    """
    Pathfinder that uses a trained neural heuristic for A*.

    Load once, call find_path() repeatedly for different start/goal pairs.
    The model runs a single forward pass to produce a heuristic map for the
    entire grid, then A* uses that map for lookup.
    """

    def __init__(self, checkpoint_path, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        version = ckpt.get("model_version", "v1")
        if version == "v2":
            self.model = HeuristicNetV2(
                num_iterations=ckpt.get("num_iterations", 4)
            )
        else:
            self.model = HeuristicNet()

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        self.max_cost = ckpt["max_cost"]

    def _get_heuristic_map(self, grid, goal):
        """Run the CNN and return a (H, W) heuristic cost array."""
        H, W = grid.shape
        ch_obstacle = grid.astype(np.float32)
        ch_goal = np.zeros((H, W), dtype=np.float32)
        ch_goal[goal[0], goal[1]] = 1.0

        x = np.stack([ch_obstacle, ch_goal])[np.newaxis]  # (1, 2, H, W)
        x_t = torch.from_numpy(x).to(self.device)

        with torch.no_grad():
            pred = self.model(x_t)

        h_map = pred.squeeze().cpu().numpy() * self.max_cost
        return np.maximum(h_map, 0.0)

    def find_path(self, grid, start, goal):
        """
        Find a path from start to goal on the given grid.

        Args:
            grid:  (H, W) numpy array — 0 = free, 1 = obstacle
            start: (row, col) tuple — bot's current position
            goal:  (row, col) tuple — target position

        Returns:
            List of (row, col) tuples from start to goal, or None if
            no path exists.
        """
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        h_map = self._get_heuristic_map(grid, goal)

        def heuristic(r, c):
            return float(h_map[r, c])

        return _astar(grid, start, goal, heuristic)

    def find_path_pixel(self, grid, start_xy, goal_xy, grid_origin=(0, 0),
                        cell_size=1.0):
        """
        Convenience wrapper for pixel-coordinate inputs.

        Converts pixel (x, y) coords to grid (row, col), runs pathfinding,
        and returns the path in pixel (x, y) coords.

        Args:
            grid:        (H, W) obstacle map
            start_xy:    (x, y) bot position in pixels
            goal_xy:     (x, y) target position in pixels
            grid_origin: (x, y) pixel position of grid cell (0, 0)
            cell_size:   pixels per grid cell

        Returns:
            List of (x, y) pixel tuples, or None if no path.
        """
        ox, oy = grid_origin

        # pixel (x, y) -> grid (row, col)
        start_col = int((start_xy[0] - ox) / cell_size)
        start_row = int((start_xy[1] - oy) / cell_size)
        goal_col = int((goal_xy[0] - ox) / cell_size)
        goal_row = int((goal_xy[1] - oy) / cell_size)

        H, W = grid.shape
        start_row = max(0, min(H - 1, start_row))
        start_col = max(0, min(W - 1, start_col))
        goal_row = max(0, min(H - 1, goal_row))
        goal_col = max(0, min(W - 1, goal_col))

        path = self.find_path(grid, (start_row, start_col), (goal_row, goal_col))
        if path is None:
            return None

        # grid (row, col) -> pixel (x, y), centered in each cell
        return [(ox + c * cell_size + cell_size / 2,
                 oy + r * cell_size + cell_size / 2) for r, c in path]


if __name__ == "__main__":
    # Quick demo
    import time

    pf = NeuralPathfinder("checkpoints/best_model.pt")
    print(f"Model loaded on {pf.device}")

    # Load a sample and test
    sample_dir = Path("data/samples_64")
    files = sorted(sample_dir.glob("*.npz"))
    if files:
        d = np.load(files[0], allow_pickle=True)
        grid = d["grid"]
        start = tuple(d["start"])
        goal = tuple(d["goal"])

        t0 = time.time()
        path = pf.find_path(grid, start, goal)
        elapsed = (time.time() - t0) * 1000

        if path:
            print(f"Path found: {len(path)} steps in {elapsed:.1f}ms")
            print(f"  Start: {start} -> Goal: {goal}")
        else:
            print("No path found")
    else:
        print("No sample data found for demo")
