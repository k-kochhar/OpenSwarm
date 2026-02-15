A 64x64 grid world. Each cell is either free or an obstacle.

There are two bots (bot 0 and bot 1) that start at random positions on the grid. Scattered around the grid are gold coins on free cells.

Each bot can do two things:
- move_to(target_pos, bot_id) — move the specified bot to a target [row, col] position. Pathfinding through obstacles is handled automatically.
- collect(bot_id) — pick up a coin at the specified bot's current position. The bot must be standing on a coin for this to work.

To collect a coin: first move_to the coin's position with the right bot_id, then call collect(bot_id).

Coin positions are included in the state (list of [row, col] positions).
Bot positions are in the state as a list — bot_id is the index (0 or 1).

Use both bots in parallel to collect coins faster.
