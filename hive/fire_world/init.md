# Fire World Simulation

A 64x64 grid world where autonomous robots respond to fire emergencies.

## Environment

The world is a grid where fires spontaneously appear in small clusters (2-8 connected cells). Fires spread slowly over time if not extinguished. The environment continuously generates new fire clusters at random locations.

## Agents

There are two firefighting robots (bot 0 and bot 1) that start at random safe positions on the grid.

## Available Tools

Each bot has access to the following capabilities:

### Navigation
- **move_to(target_pos, bot_id)** — Navigate the specified bot to a target [row, col] position. Pathfinding through obstacles and around fires is handled automatically.

### Fire Fighting
- **extinguish_flames(bot_id)** — Extinguish all fire cells in the cluster adjacent to the bot's current position. The bot must be positioned next to a fire cluster for this to work. Creates a white smoke effect during extinguishing.

- **scan_area(bot_id)** — Get detailed information about fires and hazards within a 10-cell radius of the bot.

## Mission

The robots must work together to detect and extinguish fires across the grid. Fire positions and cluster information are included in the state provided at each decision cycle.

Effective strategies:
- Use both bots in parallel to cover more ground
- Position bots adjacent to fire clusters before extinguishing
- Monitor for new fire outbreaks
- Prioritize large fire clusters that pose greater risk

The simulation tracks the number of cells currently on fire and the total cells extinguished over time.
