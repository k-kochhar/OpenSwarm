A simulated world with 50 bots on a 64x64 grid that mirror your hand's shape.

The system captures the MacBook webcam every 5 seconds, detects hand landmarks via MediaPipe, maps the hand skeleton onto the grid, and dispatches bots to form the hand's outline. No LLM is involved in the hand tracking loop.

Hand tracking starts automatically when the module loads.

Bots are identified by index (0-49). Coordinates are (row, col) on a 64x64 grid.

Available actions:
- start_hand_tracking() — start webcam hand tracking (auto-started on load).
- stop_hand_tracking() — stop webcam hand tracking.
- form_shape(shape_name) — manually arrange bots into a predefined shape: "circle", "square", "triangle", "star", "grid".
- move_bot(target_pos, bot_id) — move a single bot to [row, col].
- get_positions(bot_id) — get current position of a bot, or all bots if bot_id=None.

How hand tracking works:
1. Webcam captures a frame every 5 seconds.
2. MediaPipe detects 21 hand landmarks.
3. Landmarks are mapped onto the 64x64 grid (mirrored, scaled, centered).
4. Points are interpolated along the hand skeleton to produce ~50 target positions.
5. Bots are assigned to targets (Hungarian algorithm) and dispatched in collision-free waves.
6. If no hand is visible, bots hold their current position.
7. If the hand hasn't moved significantly, no update is sent.
