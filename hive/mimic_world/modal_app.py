"""
Modal inference service for mimic_world.

Hosts the neural A* pathfinder on GPU for fast parallel pathfinding.
50 bots can request paths simultaneously via .map().

Deploy:
    modal deploy modal_app.py

Dev mode (live reload):
    modal serve modal_app.py
"""

import modal
from pathlib import Path

app = modal.App("mimic-world-pathfinder")

# Image with model code + checkpoint mounted at runtime
move_world_dir = str(Path(__file__).parent.parent / "move_world")

pathfinder_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
    .add_local_dir(
        move_world_dir,
        remote_path="/root/move_world",
        ignore=lambda p: not str(p).endswith((".py", ".pt")),
    )
)


@app.cls(
    gpu="T4",
    image=pathfinder_image,
    min_containers=1,       # always warm — no cold starts
    max_containers=10,      # scale up under load
    buffer_containers=2,    # reserve capacity for burst
    scaledown_window=120,   # keep alive 2 min after last request
)
@modal.concurrent(max_inputs=50, target_inputs=25)
class Pathfinder:
    """Neural A* pathfinder running on GPU.

    Model loads once per container via @modal.enter.
    Each request runs a single forward pass + A* search.
    Up to 50 concurrent requests per container.
    """

    @modal.enter()
    def load_model(self):
        import sys
        sys.path.insert(0, "/root/move_world")
        from pathfinder import NeuralPathfinder

        self.pf = NeuralPathfinder("/root/move_world/checkpoints/best_model.pt")
        print(f"[modal] Model loaded on {self.pf.device}")

    @modal.method()
    def find_path(self, grid_list: list, start: list, goal: list) -> dict:
        """Find shortest path using neural A*.

        Args:
            grid_list: 64x64 grid as nested list (0=free, 1=obstacle)
            start: [row, col]
            goal: [row, col]

        Returns:
            {"path": [[r,c], ...], "length": int} or {"path": None, "length": 0}
        """
        import numpy as np
        grid = np.array(grid_list, dtype=np.int32)
        path = self.pf.find_path(grid, tuple(start), tuple(goal))
        if path is None:
            return {"path": None, "length": 0}
        return {"path": [list(p) for p in path], "length": len(path)}

    @modal.method()
    def find_paths_batch(self, grid_list: list, requests: list) -> list:
        """Find paths for multiple bots in one call.

        Args:
            grid_list: 64x64 grid as nested list
            requests: [{"bot_id": int, "start": [r,c], "goal": [r,c]}, ...]

        Returns:
            [{"bot_id": int, "path": [[r,c], ...], "length": int}, ...]
        """
        import numpy as np
        grid = np.array(grid_list, dtype=np.int32)
        results = []
        for req in requests:
            start = tuple(req["start"])
            goal = tuple(req["goal"])
            path = self.pf.find_path(grid, start, goal)
            if path is None:
                results.append({"bot_id": req["bot_id"], "path": None, "length": 0})
            else:
                results.append({
                    "bot_id": req["bot_id"],
                    "path": [list(p) for p in path],
                    "length": len(path),
                })
        return results


@app.local_entrypoint()
def main():
    """Quick test: fan out 50 pathfinding requests."""
    import numpy as np
    import time

    pf = Pathfinder()

    # Create a simple test grid
    grid = np.zeros((64, 64), dtype=np.int32)
    # Add some obstacles
    grid[10:20, 30:33] = 1
    grid[30:33, 10:50] = 1
    grid[45:48, 20:40] = 1
    grid_list = grid.tolist()

    # Generate 50 random start/goal pairs
    requests = []
    for i in range(50):
        while True:
            sr, sc = np.random.randint(0, 64, 2)
            if grid[sr, sc] == 0:
                break
        while True:
            gr, gc = np.random.randint(0, 64, 2)
            if grid[gr, gc] == 0:
                break
        requests.append({
            "bot_id": i,
            "start": [int(sr), int(sc)],
            "goal": [int(gr), int(gc)],
        })

    # Option 1: Single batch call (all 50 in one request)
    print("[test] Batch call with 50 paths...")
    t0 = time.time()
    results = pf.find_paths_batch.remote(grid_list, requests)
    elapsed = time.time() - t0
    found = sum(1 for r in results if r["path"] is not None)
    print(f"[test] Batch: {found}/50 paths found in {elapsed:.2f}s")

    # Option 2: Fan out with .map() (50 parallel requests)
    print("\n[test] Parallel .map() with 50 paths...")
    t0 = time.time()
    starts = [r["start"] for r in requests]
    goals = [r["goal"] for r in requests]
    grids = [grid_list] * 50
    results = list(pf.find_path.map(grids, starts, goals))
    elapsed = time.time() - t0
    found = sum(1 for r in results if r["path"] is not None)
    print(f"[test] Map: {found}/50 paths found in {elapsed:.2f}s")
