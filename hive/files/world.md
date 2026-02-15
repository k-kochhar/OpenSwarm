```json
{
  "world": {
    "name": "Fire World Simulation",
    "description": "A 64x64 grid world where autonomous robots respond to fire emergencies.",
    "boundaries": {
      "shape": "rectangle",
      "dimensions": {"width": 64, "height": 64},
      "origin": [0, 0]
    }
  },

  "agents": {
    "count": 2,
    "types": [
      {
        "type_id": "firefighting_bot",
        "count": 2,
        "physical": {
          "size": {"width": 15, "length": 15, "height": 15},
          "speed_max": null,
          "battery_life": null,
          "weight_capacity": null
        },
        "sensors": ["camera"],
        "actuators": ["wheels", "extinguisher"]
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
        "name": "extinguish_flames",
        "description": "Extinguish all fire cells in the cluster adjacent to the bot's current position.",
        "requires": ["extinguisher"],
        "parameters": ["bot_id"]
      },
      {
        "name": "scan_area",
        "description": "Get detailed information about fires and hazards within a 10-cell radius of the bot.",
        "requires": ["camera"],
        "parameters": ["bot_id"]
      }
    ]
  },

  "obstacles": {
    "known_types": [
      {
        "type": "fire",
        "typical_size": {"width": 1, "length": 1, "height": 0},
        "movable": false,
        "frequency": "common"
      }
    ],
    "static_obstacles": [],
    "dynamic_obstacles_expected": true
  },

  "missions": {
    "type": "bot_controlled",
    "config": {
      "user_input": {
        "enabled": false,
        "description": null,
        "examples": []
      },
      "bot_controlled": {
        "enabled": true,
        "description": "The queen autonomously monitors and acts on fire clusters appearing in the grid.",
        "trigger": {
          "type": "interval",
          "interval_seconds": 3,
          "condition": "new fire cluster detected"
        },
        "response": "Deploy bots to extinguish fires",
        "examples": [
          "Bots reposition themselves to cover larger areas",
          "Bots extinguish detected fire clusters"
        ]
      }
    }
  },

  "rules": [
    "Agents must maintain at least 20cm distance from each other",
    "If battery below 10%, agent must return to base"
  ],

  "assumptions": [
    "Assumed physical size of the robots to be 15cm based on 'small robots.'",
    "Assumed 'dynamic_obstacles_expected' is true due to fire spreading.",
    "Assumed frequencies of fire clusters based on 'common' occurrences."
  ]
}
```