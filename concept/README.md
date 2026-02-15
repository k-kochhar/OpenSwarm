## Multi-Robot Fire-Fighting Simulation

A deterministic 2D overhead simulation of a Queen central intelligence coordinating 3 firefighting robots.

### Requirements

```bash
pip install pygame
```

### Running the Simulation

**Interactive mode:**
```bash
cd simulation
python main.py
```

**Export video frames:**
```bash
python main.py --export
```

This will save frames to `../output/frames/` and print ffmpeg command to create MP4.

### Controls

- **R**: Restart simulation
- **ESC**: Quit

### Project Structure

```
simulation/
├── main.py              # Pygame loop, rendering, video export
├── world.py             # World state, item, bravo, fire
├── robot.py             # Robot kinematics, collision handling
├── queen.py             # Logging system, metrics
├── state_machine.py     # State machine and transitions
├── utils.py             # Math helpers
└── README.md            # This file
```

### Simulation Details

**Specifications:**
- Window: 1200×800 (900×800 map + 300×800 panel)
- Robot radius: 18px
- Robot max speed: 140 px/s
- Turn rate: 240°/s
- Item: 70×70px square
- Destination: 110×110px region
- Fire suppression radius: 40px
- FPS: 60

**T-Formation:**
- Alpha: LEFT face of item
- Beta: RIGHT face of item
- Charlie: BOTTOM face of item

**Push Speeds:**
- 3 robots: 90 px/s (100%)
- 2 robots: 18 px/s (20%)
- 1 robot: 0 px/s

**Timeline:**
1. Robots spawn at different locations
2. Robots navigate to T-formation around item
3. Formation validates (must be stable for 0.5s)
4. Transport begins toward Location Bravo
5. Fire appears when item_x > 450
6. ALL robots pause for 1 second (thinking)
7. Charlie diverts to extinguish fire
8. Alpha & Beta continue at 20% speed
9. Fire suppressed in exactly 1 second
10. Charlie returns to formation
11. Full speed resumes when Charlie redocks
12. Mission completes at Bravo

**Physics:**
- Deterministic kinematics
- Gradual rotation (no instant turns)
- Robot-robot collision resolution
- Robot-item collision resolution
- No teleporting, no overlaps

### Creating Video

After running with `--export`:

```bash
cd ../output/frames
ffmpeg -framerate 60 -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p simulation.mp4
```

### State Machine

```
INIT → FORMING → VALIDATING_FORMATION → TRANSPORTING
                                            ↓
                                      FIRE_DETECTED
                                            ↓
                                        THINKING (1s pause)
                                            ↓
                                        RESPONDING
                                            ↓
                                       SUPPRESSING (1s)
                                            ↓
                                        RETURNING
                                            ↓
                                      TRANSPORTING
                                            ↓
                                        COMPLETE
```

### Queen Log Messages

The Queen logs these exact messages at appropriate state transitions:

1. "We need to move Item A to Location Bravo"
2. "Assigning Alpha, Beta, Charlie to transport task"
3. "Robots forming transport configuration"
4. "Formation established. Beginning transport"
5. "Alert: Fire detected at (x=520, y=320)"
6. "Pausing transport"
7. "Evaluating response strategy"
8. "Send Robot Charlie to extinguish the fire. Robots Alpha and Beta continue to move Item A to Location Bravo"
9. "Charlie engaging suppression protocol"
10. "Fire extinguished"
11. "Robot Charlie returning to transport formation"
12. "Item A delivered to Location Bravo"
13. "Mission complete"

### Code Quality

- Pure deterministic behavior (no randomness except visual effects)
- Clean modular architecture
- Explicit state machine
- Strict formation validation
- Physical collision resolution
- Frame-perfect timing
