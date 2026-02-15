"""
Explicit state machine for the simulation.
"""
import math
from robot import Robot
from utils import Vec2

class SimulationStateMachine:
    """Manages the overall simulation state and transitions."""

    # States
    STATE_INIT = "INIT"
    STATE_SPAWNED = "SPAWNED"
    STATE_FORMING = "FORMING"
    STATE_VALIDATING_FORMATION = "VALIDATING_FORMATION"
    STATE_FORMATION_READY = "FORMATION_READY"
    STATE_TRANSPORTING = "TRANSPORTING"
    STATE_FIRE_REACTION_DELAY = "FIRE_REACTION_DELAY"
    STATE_FIRE_DETECTED = "FIRE_DETECTED"
    STATE_THINKING = "THINKING"
    STATE_RESPONDING = "RESPONDING"
    STATE_SUPPRESSING = "SUPPRESSING"
    STATE_RETURNING = "RETURNING"
    STATE_REUNITED = "REUNITED"
    STATE_COMPLETE = "COMPLETE"

    def __init__(self, world, robots, queen):
        self.world = world
        self.robots = robots
        self.queen = queen

        self.state = self.STATE_INIT
        self.state_timer = 0.0
        self.formation_stable_timer = 0.0
        self.fire_triggered = False
        self.charlie_rejoined = False

        # Docking assignments
        self.dock_assignments = {
            'Alpha': 'LEFT',
            'Beta': 'RIGHT',
            'Charlie': 'BOTTOM'
        }

    def update(self, dt, sim_time):
        """Update state machine."""
        self.state_timer += dt

        if self.state == self.STATE_INIT:
            self._state_init(sim_time)

        elif self.state == self.STATE_SPAWNED:
            self._state_spawned(dt, sim_time)

        elif self.state == self.STATE_FORMING:
            self._state_forming(dt, sim_time)

        elif self.state == self.STATE_VALIDATING_FORMATION:
            self._state_validating_formation(dt, sim_time)

        elif self.state == self.STATE_FORMATION_READY:
            self._state_formation_ready(dt, sim_time)

        elif self.state == self.STATE_TRANSPORTING:
            self._state_transporting(dt, sim_time)

        elif self.state == self.STATE_FIRE_REACTION_DELAY:
            self._state_fire_reaction_delay(dt, sim_time)

        elif self.state == self.STATE_FIRE_DETECTED:
            self._state_fire_detected(sim_time)

        elif self.state == self.STATE_THINKING:
            self._state_thinking(dt, sim_time)

        elif self.state == self.STATE_RESPONDING:
            self._state_responding(dt, sim_time)

        elif self.state == self.STATE_SUPPRESSING:
            self._state_suppressing(dt, sim_time)

        elif self.state == self.STATE_RETURNING:
            self._state_returning(dt, sim_time)

        elif self.state == self.STATE_REUNITED:
            self._state_reunited(dt, sim_time)

        elif self.state == self.STATE_COMPLETE:
            pass

        # Update Queen metrics
        self._update_queen_metrics()

    def _state_init(self, sim_time):
        """Initialize mission."""
        self.queen.log("We need to move Item A to Location Bravo", sim_time)
        self.queen.log("Assigning Alpha, Beta, Charlie to transport task", sim_time)

        # Wait 5 seconds before starting
        self._change_state(self.STATE_SPAWNED)

    def _state_spawned(self, dt, sim_time):
        """Wait 3 seconds after spawning."""
        if self.state_timer >= 3.0:
            self.queen.log("Robots forming transport configuration", sim_time)

            # Compute dock positions
            self._compute_dock_positions()

            # Send robots to docks
            for robot in self.robots:
                robot.state = Robot.STATE_MOVING_TO_DOCK

            self._change_state(self.STATE_FORMING)

    def _state_forming(self, dt, sim_time):
        """Robots moving to formation."""
        all_docked = all(robot.is_docked_correctly() for robot in self.robots)

        if all_docked:
            self._change_state(self.STATE_VALIDATING_FORMATION)
            self.formation_stable_timer = 0.0

    def _state_validating_formation(self, dt, sim_time):
        """Validate formation stability."""
        all_stable = all(robot.is_docked_correctly() for robot in self.robots)

        if all_stable:
            self.formation_stable_timer += dt
            if self.formation_stable_timer >= 0.5:
                self.queen.log("Formation established", sim_time)
                for robot in self.robots:
                    robot.state = Robot.STATE_PUSHING
                    robot.target_pos = None  # Clear targets, use dock navigation
                # Wait 5 seconds before starting transport
                self._change_state(self.STATE_FORMATION_READY)
        else:
            self.formation_stable_timer = 0.0

    def _state_formation_ready(self, dt, sim_time):
        """Wait 3 seconds after formation before starting transport."""
        if self.state_timer >= 3.0:
            self.queen.log("Beginning transport", sim_time)
            self._change_state(self.STATE_TRANSPORTING)

    def _state_transporting(self, dt, sim_time):
        """Transporting item to Bravo."""
        # Move item
        pushing_robots = [r for r in self.robots if r.state == Robot.STATE_PUSHING]
        self._move_item(pushing_robots, dt, full_speed=True)

        # Update dock positions
        self._compute_dock_positions()

        # Check for fire trigger - fire appears but robots continue for 2s
        if not self.fire_triggered and self.world.item_pos.x > 450:
            self.world.fire_active = True
            self.world.fire_intensity = 0.0
            self._change_state(self.STATE_FIRE_REACTION_DELAY)
            self.fire_triggered = True
            return

        # Check mission complete
        if self.world.is_item_at_bravo():
            self._change_state(self.STATE_COMPLETE)
            self.queen.log("Item A delivered to Location Bravo", sim_time)
            self.queen.log("Mission complete", sim_time)
            for robot in self.robots:
                robot.state = Robot.STATE_COMPLETE
                robot.vel = Vec2(0, 0)

    def _state_fire_reaction_delay(self, dt, sim_time):
        """Fire appeared - robots continue for 2 seconds before reacting."""
        # Continue transporting as normal
        pushing_robots = [r for r in self.robots if r.state == Robot.STATE_PUSHING]
        self._move_item(pushing_robots, dt, full_speed=True)
        self._compute_dock_positions()

        # Ramp up fire intensity during reaction delay
        self.world.fire_intensity = min(1.0, self.world.fire_intensity + dt * 0.5)

        # After 2 seconds, react to fire
        if self.state_timer >= 2.0:
            self._change_state(self.STATE_FIRE_DETECTED)

    def _state_fire_detected(self, sim_time):
        """Robots noticed the fire - freeze and assess."""
        self.queen.log(f"Alert: Fire detected at (x={int(self.world.fire_pos.x)}, y={int(self.world.fire_pos.y)})", sim_time)
        self.queen.log("Pausing transport", sim_time)

        # Freeze all robots by putting them in IDLE state
        for robot in self.robots:
            robot.state = Robot.STATE_IDLE
            robot.vel = Vec2(0, 0)

        self._change_state(self.STATE_THINKING)

    def _state_thinking(self, dt, sim_time):
        """3 second thinking pause."""
        # Fire intensity already ramped during reaction delay, keep it at max
        self.world.fire_intensity = 1.0

        if self.state_timer >= 3.0:
            self.queen.log("Evaluating response strategy", sim_time)
            self.queen.log("Send Robot Charlie to extinguish the fire. Robots Alpha and Beta continue to move Item A to Location Bravo", sim_time)

            # Assign Charlie to fire
            charlie = next(r for r in self.robots if r.name == "Charlie")
            charlie.state = Robot.STATE_RESPONDING
            charlie.set_target(self.world.fire_pos.x, self.world.fire_pos.y)

            # Alpha and Beta continue pushing
            for robot in self.robots:
                if robot.name in ["Alpha", "Beta"]:
                    robot.state = Robot.STATE_PUSHING

            self._change_state(self.STATE_RESPONDING)

    def _state_responding(self, dt, sim_time):
        """Charlie responding to fire, Alpha/Beta pushing."""
        charlie = next(r for r in self.robots if r.name == "Charlie")

        # Move item at 20% speed
        pushing_robots = [r for r in self.robots if r.state == Robot.STATE_PUSHING]
        self._move_item(pushing_robots, dt, full_speed=False)

        # Update dock positions for Alpha/Beta
        self._compute_dock_positions()

        # Check if Charlie reached fire
        dist = (charlie.pos - self.world.fire_pos).length()
        if dist < self.world.fire_radius:
            self.queen.log("Charlie engaging suppression protocol", sim_time)
            charlie.state = Robot.STATE_SUPPRESSING
            charlie.vel = Vec2(0, 0)
            self._change_state(self.STATE_SUPPRESSING)

        # Check mission complete (rare but possible)
        if self.world.is_item_at_bravo():
            self._change_state(self.STATE_COMPLETE)
            self.queen.log("Item A delivered to Location Bravo", sim_time)
            self.queen.log("Mission complete", sim_time)

    def _state_suppressing(self, dt, sim_time):
        """Charlie suppressing fire for 1 second."""
        # Continue moving item at 20% speed
        pushing_robots = [r for r in self.robots if r.state == Robot.STATE_PUSHING]
        self._move_item(pushing_robots, dt, full_speed=False)
        self._compute_dock_positions()

        # Decrease fire intensity over 1 second
        self.world.fire_intensity = max(0, self.world.fire_intensity - dt)

        if self.state_timer >= 1.0:
            self.world.fire_active = False
            self.world.fire_intensity = 0.0

            self.queen.log("Fire extinguished", sim_time)
            self.queen.log("Robot Charlie returning to transport formation", sim_time)

            charlie = next(r for r in self.robots if r.name == "Charlie")
            charlie.state = Robot.STATE_RETURNING
            charlie.target_pos = None  # Clear fire target

            # Compute charlie's dock for return
            self._compute_dock_positions()
            # Dock info is already set by _compute_dock_positions

            self._change_state(self.STATE_RETURNING)

    def _state_returning(self, dt, sim_time):
        """Charlie returning to formation."""
        charlie = next(r for r in self.robots if r.name == "Charlie")

        # Continue moving item at 20% speed
        pushing_robots = [r for r in self.robots if r.state == Robot.STATE_PUSHING]
        self._move_item(pushing_robots, dt, full_speed=False)

        # Update dock positions BEFORE checking if Charlie is docked
        self._compute_dock_positions()

        # Check if Charlie has rejoined
        # Use relaxed distance check since dock is moving
        if charlie.dock_pos:
            dist_to_dock = (charlie.pos - charlie.dock_pos).length()
            # Charlie just needs to be close to the moving formation
            if dist_to_dock < 35.0:
                charlie.dock_stable_timer += dt
                if charlie.dock_stable_timer >= 0.2:  # Reduced from 0.3s
                    charlie.state = Robot.STATE_PUSHING
                    charlie.target_pos = None  # Clear target, use dock navigation
                    self.charlie_rejoined = True
                    self.queen.log("Robot Charlie has rejoined the formation", sim_time)
                    # Wait 5 seconds before resuming full speed
                    self._change_state(self.STATE_REUNITED)
            else:
                charlie.dock_stable_timer = 0.0
        else:
            charlie.dock_stable_timer = 0.0

        # Check mission complete
        if self.world.is_item_at_bravo():
            self._change_state(self.STATE_COMPLETE)
            self.queen.log("Item A delivered to Location Bravo", sim_time)
            self.queen.log("Mission complete", sim_time)
            for robot in self.robots:
                robot.state = Robot.STATE_COMPLETE
                robot.vel = Vec2(0, 0)

    def _state_reunited(self, dt, sim_time):
        """Wait 3 seconds after Charlie reunites - everything pauses."""
        # Freeze all robots during the pause
        for robot in self.robots:
            robot.vel = Vec2(0, 0)

        if self.state_timer >= 3.0:
            self.queen.log("Full transport capacity restored", sim_time)
            self._change_state(self.STATE_TRANSPORTING)

        # Check mission complete
        if self.world.is_item_at_bravo():
            self._change_state(self.STATE_COMPLETE)
            self.queen.log("Item A delivered to Location Bravo", sim_time)
            self.queen.log("Mission complete", sim_time)
            for robot in self.robots:
                robot.state = Robot.STATE_COMPLETE
                robot.vel = Vec2(0, 0)

    def _compute_dock_positions(self):
        """Compute dock positions for all robots."""
        item_half = self.world.item_size / 2
        buffer = 6

        for robot in self.robots:
            face = self.dock_assignments[robot.name]

            if face == 'LEFT':
                normal = Vec2(-1, 0)
                dock_offset = Vec2(-(item_half + robot.radius + buffer), 0)
                facing_angle = 0  # Face right (toward item center)

            elif face == 'RIGHT':
                normal = Vec2(1, 0)
                dock_offset = Vec2(item_half + robot.radius + buffer, 0)
                facing_angle = math.pi  # Face left

            elif face == 'BOTTOM':
                normal = Vec2(0, 1)
                dock_offset = Vec2(0, item_half + robot.radius + buffer)
                facing_angle = -math.pi / 2  # Face up

            dock_pos = self.world.item_pos + dock_offset
            robot.set_dock(dock_pos.x, dock_pos.y, facing_angle)

    def _move_item(self, pushing_robots, dt, full_speed):
        """Move item based on pushing robots."""
        count = len(pushing_robots)

        if count == 0:
            return

        # Speed based on robot count (halved from original)
        if full_speed:
            if count >= 3:
                speed = 45.0  # Half of original 90.0
            elif count == 2:
                speed = 9.0  # 20% of 45
            else:
                speed = 0
        else:
            # Reduced capacity (2 robots)
            speed = 9.0 if count >= 2 else 0  # Half of original 18.0

        if speed == 0:
            return

        # Direction toward Bravo
        to_bravo = self.world.bravo_pos - self.world.item_pos
        direction = to_bravo.normalized()

        # Move item
        self.world.item_pos = self.world.item_pos + direction * speed * dt

    def _update_queen_metrics(self):
        """Update Queen panel metrics."""
        pushing_count = len([r for r in self.robots if r.state == Robot.STATE_PUSHING])

        if pushing_count >= 3:
            speed_percent = 100
        elif pushing_count == 2:
            speed_percent = 20
        else:
            speed_percent = 0

        if self.world.fire_active:
            fire_status = "ACTIVE"
        elif self.fire_triggered:
            fire_status = "EXTINGUISHED"
        else:
            fire_status = "NONE"

        self.queen.update_metrics(self.state, pushing_count, speed_percent, fire_status)

    def _change_state(self, new_state):
        """Change state and reset timer."""
        self.state = new_state
        self.state_timer = 0.0
