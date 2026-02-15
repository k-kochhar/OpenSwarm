```json
{
  "world": {
    "name": "ArUco Vision-Tracked Robot Swarm",
    "description": "A physical robot swarm tracked via overhead camera using ArUco markers. Robots operate in a 2D plane with pixel-based coordinate system from camera frame. Obstacles are detected via color-based computer vision (green, blue, black objects).",
    "boundaries": {
      "shape": "rectangle",
      "dimensions": {
        "width_px": 640,
        "height_px": 480,
        "note": "Pixel dimensions in camera frame; actual physical size unknown"
      },
      "origin": "[0, 0] — top-left corner of camera frame (x increases right, y increases down)"
    }
  },

  "agents": {
    "count": 3,
    "types": [
      {
        "type_id": "default",
        "count": 3,
        "physical": {
          "size": {
            "note": "Size not specified in init document",
            "estimated_radius_cm": 10
          },
          "speed_max": null,
          "battery_life": null,
          "weight_capacity": null
        },
        "sensors": [
          "aruco_marker",
          "none_onboard"
        ],
        "actuators": [
          "differential_drive",
          "pusher_plate"
        ]
      }
    ],
    "notes": [
      "Robots are ESP-based devices controlled over WebSocket",
      "Position tracking via overhead camera reading ArUco markers (IDs: 0, 3, 6)",
      "Bot 0 = marker 0 (ESP1), Bot 1 = marker 3 (ESP2), Bot 2 = marker 6 (ESP3)",
      "All sensing is external (camera-based); robots have no onboard sensors exposed to swarm controller"
    ]
  },

  "actions": {
    "built_in": [
      "move_to(target, bot_id) — navigate agent to target [x, y] pixel coordinates (pathfinding handled automatically)",
      "stop(bot_id) — halt agent immediately"
    ],
    "extra": [
      {
        "name": "push_and_exit",
        "description": "Push an obstacle off the playing field. Robot drives in straight line toward nearest boundary edge, shoving whatever is in front of it out of the area. No pathfinding used—goes straight through obstacles.",
        "requires": ["differential_drive", "pusher_plate"],
        "parameters": ["bot_id — which robot to use (0, 1, or 2)"],
        "usage_pattern": "1) Call get_orientation_coordinates to locate bot and obstacle, 2) Use move_to to position bot at obstacle, 3) Call push_and_exit to shove it off the edge"
      },
      {
        "name": "get_orientation_coordinates",
        "description": "Query current pixel position and orientation of robots from overhead camera tracking system. Returns real-time marker data.",
        "requires": ["aruco_marker", "external_camera"],
        "parameters": ["bot_id — which robot to query (0, 1, 2), or None to get all detected robots"],
        "returns": "Single bot: {bot_id, center: [x,y], orientation_deg, orientation_rad}. All bots: list of such dicts."
      }
    ]
  },

  "obstacles": {
    "known_types": [
      {
        "type": "color_detected_object",
        "description": "Objects detected via color-based computer vision (green, blue, or black colored items)",
        "typical_size": {
          "note": "Size varies; detected at cell level in 32x32 grid"
        },
        "movable": true,
        "frequency": "common"
      }
    ],
    "static_obstacles": [],
    "dynamic_obstacles_expected": true,
    "detection_method": "HSV color segmentation on camera feed (green: 35-85 hue, blue: 100-130 hue, black: value < 50). Downsampled to 32×32 grid, then scaled to 64×64 for pathfinding. Cells with >30% obstacle-colored pixels marked as blocked.",
    "notes": [
      "A 4-cell radius around each bot is kept clear of obstacle markings to prevent false positives",
      "Obstacles can be physically moved using push_and_exit action",
      "Real-time detection updates every frame from camera"
    ]
  },

  "missions": {
    "type": "user_input",
    "config": {
      "user_input": {
        "enabled": true,
        "description": "User issues commands to position robots and clear obstacles. Queen executes tasks by querying robot positions, planning movements, and coordinating multi-bot operations.",
        "examples": [
          "Move bot 0 to position [320, 240]",
          "Clear all obstacles from the area",
          "Position bot 1 near the blue object and push it out",
          "Stop all robots",
          "Show me where all bots are currently located",
          "Have bot 2 push the green obstacle off the field"
        ]
      },
      "bot_controlled": {
        "enabled": false,
        "description": null,
        "trigger": null,
        "response": null,
        "examples": []
      }
    },
    "notes": [
      "No autonomous monitoring configured in init document",
      "All tasks are user-initiated commands",
      "Framework provides screenshot of current camera view with each task",
      "Multiple robots can execute moves in parallel"
    ]
  },

  "rules": [
    "Always call get_orientation_coordinates(bot_id=None) before planning movements to get current positions of all robots",
    "Multiple robots can move simultaneously in parallel",
    "When removing obstacles: (1) get positions, (2) move_to the obstacle, (3) push_and_exit to clear it",
    "push_and_exit uses straight-line movement with no pathfinding—ensure clear path or accept collision",
    "Robots must respect boundary limits (0 ≤ x < frame_width, 0 ≤ y < frame_height)",
    "A 4-cell safety radius around each bot is automatically kept clear of obstacle markings during vision processing",
    "Coordinate system: x increases rightward, y increases downward (standard image coordinates)",
    "Bot IDs for commands are 0, 1, 2 (not the internal marker IDs 0, 3, 6)",
    "Stop commands immediately halt robot movement and cancel active path following",
    "WebSocket connection to ESP devices required for command execution—verify overlay.py is running",
    "Path planning generates intermediate waypoints every ~40 pixels for smooth tracking",
    "Vision system operates on 32×32 detection grid, scaled to 64×64 for pathfinding"
  ],

  "assumptions": [
    "Robot physical size assumed ~10cm radius (not specified in init document)",
    "Camera frame dimensions assumed 640×480 pixels (detected from screenshot, may vary)",
    "Robot maximum speed not specified—controlled via WebSocket at fixed command interval (0.5s)",
    "Battery life not specified—assumed adequate for session duration",
    "No weight capacity specified—push_and_exit assumed capable of moving detected obstacles",
    "Playing field is flat 2D surface viewed from directly above",
    "ArUco markers are always visible and correctly oriented for camera detection",
    "Obstacle colors (green, blue, black) are distinct enough for reliable HSV segmentation",
    "ESP WebSocket server running on localhost:8765",
    "overlay.py process is running and continuously updating markers.json and screenshot.png",
    "Physical robots respond to F (forward), L (left), R (right), S (stop) commands",
    "PathFollower control loop operates at ~20Hz (0.05s sleep interval)",
    "Robots have sufficient pushing force to move obstacles via push_and_exit",
    "Grid cell size approximately 20×15 pixels (640/32 × 480/32)",
    "No collision avoidance between robots during parallel movement—user must plan safe paths"
  ]
}
```