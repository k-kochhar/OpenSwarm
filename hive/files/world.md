```json
{
  "world": {
    "name": "ArUco Overhead Camera Robot Arena",
    "description": "A physical robot environment tracked by an overhead camera using ArUco markers. Robots are controlled via WebSocket and navigate in pixel coordinate space within the camera's field of view. Obstacles are detected via color-based computer vision (green, blue, and black objects).",
    "boundaries": {
      "shape": "rectangle",
      "dimensions": {
        "width_pixels": 640,
        "height_pixels": 480,
        "note": "Actual dimensions read from camera frame at runtime"
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
            "note": "ESP-based wheeled robots with ArUco markers",
            "estimated_diameter_cm": 15
          },
          "speed_max": 0.2,
          "battery_life": null,
          "weight_capacity": null
        },
        "sensors": [
          "overhead_camera (shared)",
          "ArUco marker (passive position tracking)"
        ],
        "actuators": [
          "wheels (differential drive)",
          "pusher (front-facing surface for pushing obstacles)"
        ]
      }
    ]
  },

  "actions": {
    "built_in": [
      "move_to(x, y) — navigate agent to target coordinates (handled by path planner)",
      "stop() — halt agent immediately"
    ],
    "extra": [
      {
        "name": "get_orientation_coordinates",
        "description": "Query current pixel position and orientation (degrees and radians) of one or all robots from the overhead camera system",
        "requires": ["overhead_camera", "ArUco marker"],
        "parameters": [
          "bot_id (int or None) — specific robot ID (0, 1, or 2), or None for all robots"
        ]
      },
      {
        "name": "push_and_exit",
        "description": "Push an obstacle straight toward the nearest boundary edge and off the playing field. The robot drives in a straight line in its current facing direction without pathfinding, shoving whatever is in front of it. Used for obstacle removal.",
        "requires": ["wheels", "pusher", "overhead_camera"],
        "parameters": [
          "bot_id (int) — which robot performs the push (0, 1, or 2)"
        ]
      }
    ]
  },

  "obstacles": {
    "known_types": [
      {
        "type": "colored_object",
        "typical_size": {
          "note": "Variable — detected by color (green, blue, or black objects in HSV space)",
          "estimated_range_cm": "5-30"
        },
        "movable": true,
        "frequency": "common"
      },
      {
        "type": "boundary_wall",
        "typical_size": {
          "note": "Frame edges — hard boundaries of the camera view"
        },
        "movable": false,
        "frequency": "common"
      }
    ],
    "static_obstacles": [],
    "dynamic_obstacles_expected": true
  },

  "missions": {
    "type": "user_input",
    "config": {
      "user_input": {
        "enabled": true,
        "description": "The queen receives task commands from a human operator and executes them using the available robot actions. Tasks typically involve navigation, obstacle clearing, or multi-robot coordination.",
        "examples": [
          "Move bot 0 to position [320, 240]",
          "Clear the green obstacle using bot 1",
          "Get all robot positions",
          "Push the blue box off the field with bot 2",
          "Move all three robots to form a line",
          "Stop bot 1"
        ]
      },
      "bot_controlled": {
        "enabled": false,
        "description": null,
        "trigger": {
          "type": null,
          "interval_seconds": null,
          "condition": null
        },
        "response": null,
        "examples": []
      }
    }
  },

  "rules": [
    "Always call get_orientation_coordinates(bot_id=None) before planning moves to know current robot positions",
    "Multiple robots can move in parallel — coordinate movements to avoid collisions",
    "To remove an obstacle: (1) get positions with get_orientation_coordinates, (2) move_to the obstacle location, (3) call push_and_exit to shove it off the field",
    "Robots must stay within the camera frame boundaries (0 ≤ x ≤ frame_width, 0 ≤ y ≤ frame_height)",
    "The coordinate system uses pixels: x increases rightward, y increases downward from the top-left origin",
    "Pathfinding for move_to uses neural A* with color-based obstacle detection (green, blue, black objects)",
    "A 4-cell radius around each robot is kept clear in the obstacle grid to prevent false collision detection",
    "push_and_exit drives straight toward the nearest boundary — no pathfinding, obstacles along the way are pushed through",
    "Robots are identified by bot_id (0, 1, or 2) in all commands",
    "The overhead camera provides real-time position and orientation tracking via ArUco markers",
    "If a robot is not detected by the camera, positioning commands will fail — ensure markers are visible"
  ],

  "assumptions": [
    "Robot diameter assumed to be ~15cm based on typical ESP-based differential drive robots",
    "Camera frame dimensions assumed to be 640×480 pixels (default) but read from runtime screenshot",
    "Robot maximum speed assumed to be ~0.2 m/s based on typical hobby robot actuators",
    "Battery life not specified — assumed to be managed externally or sufficient for task duration",
    "Weight capacity not specified — robots appear to push lightweight obstacles only",
    "Obstacle detection relies on color thresholding (green, blue, black in HSV space) — assumes obstacles have these colors",
    "The world operates in a 2D plane — no elevation or 3D obstacles considered",
    "ArUco markers are assumed to be unique and correctly mapped (marker IDs 0, 3, 6 → bot IDs 0, 1, 2)",
    "WebSocket connection to ESP devices assumed to be reliable and running on localhost:8765",
    "The overhead camera system (overlay.py) is assumed to be running and continuously updating markers.json and screenshot.png",
    "Grid resolution is 32×32 for obstacle detection, scaled to 64×64 for neural pathfinding",
    "Collision avoidance between robots during parallel movement is handled by maintaining awareness of all bot positions before issuing move commands"
  ]
}
```