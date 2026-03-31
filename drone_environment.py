"""
Gymnasium environment for CoppeliaSim drone obstacle avoidance.

The drone's goal is to fly freely, explore the environment, and
avoid obstacles (trees, people, buildings) for as long as possible.

Observation (10D):
    - 4 LiDAR distances (front, back, left, right)
    - 3 drone position (x, y, z)
    - 3 drone velocity (vx, vy, vz)

Action (3D):
    - Velocity adjustments (dx, dy, dz) in [-1, 1]
    - Applied on top of baseline safety rules

Reward:
    + Survival bonus each step
    + Exploration bonus for visiting new grid cells
    - Proximity penalty when near obstacles
    - Collision penalty (episode ends)
    - Out-of-bounds penalty (episode ends)

Baseline rules (agent can override):
    - Slow down when obstacles are close
    - Gain altitude when obstacles are very close
"""

import math
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


"""
Gymnasium environment for CoppeliaSim drone obstacle avoidance.

The drone's goal is to fly freely, explore the environment, and
avoid obstacles (trees, people, buildings) for as long as possible.
No waypoints or destinations — purely learning-based navigation.

Follows the standard Gymnasium API:
    env.reset()  → (observation, info)
    env.step()   → (observation, reward, terminated, truncated, info)

Observation (10D):
    [0:4]  — 4 LiDAR distances (front, back, left, right)
    [4:7]  — drone position (x, y, z)
    [7:10] — drone velocity (vx, vy, vz)

Action (3D):
    Velocity adjustments (dx, dy, dz) in [-1, 1]
    Applied on top of baseline safety rules

Reward:
    +1.0  per step survived
    +2.0  for visiting a new grid cell
    -var  proximity penalty (scales with closeness to obstacles)
    -0.5  hovering penalty (speed below 0.05 m/s)
    -50   collision (episode terminates)
    -50   out of bounds (episode terminates)

Baseline safety rules (agent can override):
    - Slow down when obstacles are within proximity_threshold
    - Gain altitude when obstacles are within altitude_boost_threshold
    - Push away from obstacle direction
"""

import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from lidar import get_lidar_handles, read_lidar_array
from navigation import get_drone_pos_array, get_drone_velocity, set_target


class DroneAvoidanceEnv(gym.Env):
    """
    CoppeliaSim drone environment for obstacle avoidance.

    Implements the standard Gymnasium Env interface so it can be
    used with any compatible RL library (stable-baselines3, etc).

    The agent controls the drone's velocity in 3D space while
    baseline safety rules provide a foundation that the agent
    can learn to work with or override.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        max_steps=2000,                    # Max steps before episode ends
        speed_scale=0.05,                  # Scales agent action into movement
        collision_distance=0.3,            # LiDAR distance that counts as a crash (m)
        proximity_threshold=1.0,           # Distance to start avoiding obstacles (m)
        altitude_boost_threshold=0.5,      # Distance to start gaining altitude (m)
        boundary_min=-5.0,                 # Min x/y boundary of the flying area (m)
        boundary_max=5.0,                  # Max x/y boundary of the flying area (m)
        min_altitude=0.3,                  # Lowest allowed flight height (m)
        max_altitude=4.0,                  # Highest allowed flight height (m)
        flight_height=1.5,                 # Starting flight height (m)
        ideal_altitude=1.5,                # Ideal cruising altitude (m) — reward for staying near
        altitude_boost_cap=2.5,            # Max altitude the baseline boost can push to (m)
        exploration_grid_size=0.5,         # Size of grid cells for exploration tracking (m)
        render_mode=None,                  # Gymnasium render mode
    ):
        """
        Initializes the drone environment.

        Sets up the observation and action spaces, stores all
        configuration parameters, and prepares internal state.
        Does NOT connect to CoppeliaSim yet — that happens
        on the first call to reset().

        Args:
            max_steps (int): Max steps before episode is truncated.
            speed_scale (float): Multiplier applied to actions.
            collision_distance (float): LiDAR reading below this
                triggers a crash and ends the episode.
            proximity_threshold (float): LiDAR reading below this
                activates baseline safety rules.
            altitude_boost_threshold (float): LiDAR reading below
                this triggers an altitude increase.
            boundary_min (float): Min x/y world coordinate.
            boundary_max (float): Max x/y world coordinate.
            min_altitude (float): Minimum allowed z position.
            max_altitude (float): Maximum allowed z position.
            flight_height (float): Starting altitude on reset.
            ideal_altitude (float): Ideal cruising altitude. The
                agent is rewarded for staying near this height
                and penalized for drifting away.
            altitude_boost_cap (float): Maximum altitude the
                baseline altitude boost rule can push to.
            exploration_grid_size (float): Grid cell size for
                tracking which areas the drone has visited.
            render_mode (str, optional): Gymnasium render mode.
        """
        super().__init__()

        # ── Config ──
        self.max_steps = max_steps
        self.speed_scale = speed_scale
        self.collision_distance = collision_distance
        self.proximity_threshold = proximity_threshold
        self.altitude_boost_threshold = altitude_boost_threshold
        self.boundary_min = boundary_min
        self.boundary_max = boundary_max
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.flight_height = flight_height
        self.ideal_altitude = ideal_altitude
        self.altitude_boost_cap = altitude_boost_cap
        self.exploration_grid_size = exploration_grid_size
        self.render_mode = render_mode

        # Observation space
        # 4 lidar readings + 3 position + 3 velocity = 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # Action space
        # 3D velocity adjustment (x, y, z) each in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # CoppeliaSim handles (set on first reset)
        self.client = None
        self.sim = None
        self.drone = None
        self.target = None
        self.lidar_handles = {}

        # ── Episode state ──
        self.step_count = 0
        self.visited_cells = set()
        self.steps_in_current_cell = 0
        self.last_cell = None
        self._connected = False

    #  Connection
    def _connect(self):
        """
        Establishes connection to CoppeliaSim.

        Grabs handles for the drone, target dummy, and all four
        LiDAR sensors. Skips if already connected.
        """
        if self._connected:
            return

        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')

        self.drone = self.sim.getObject('/Quadcopter')
        self.target = self.sim.getObject('/target')
        self.lidar_handles = get_lidar_handles(self.sim)

        self._connected = True

    #  Observation
    def _get_observation(self):
        """
        Builds the 10D observation vector from the current sim state.

        Reads all four LiDAR sensors, the drone's world position,
        and its linear velocity, then concatenates them into a
        single flat array.

        Returns:
            np.ndarray: Shape (10,) float32 observation.
                [0:4]  — LiDAR distances (F, B, L, R) in metres
                [4:7]  — Drone position (x, y, z) in metres
                [7:10] — Drone velocity (vx, vy, vz) in m/s
        """
        lidar = read_lidar_array(self.sim, self.lidar_handles)
        pos = get_drone_pos_array(self.sim, self.drone)
        vel = get_drone_velocity(self.sim, self.drone)

        return np.concatenate([lidar, pos, vel])


    #  Exploration tracking
    def _get_grid_cell(self, pos):
        """
        Converts a world position to a discrete grid cell.

        Used to track which areas the drone has visited so the
        exploration reward can encourage covering new ground.

        Args:
            pos (np.ndarray): ``[x, y, z]`` position in metres.

        Returns:
            tuple[int, int, int]: Grid cell indices (gx, gy, gz).
        """
        gx = int(pos[0] / self.exploration_grid_size)
        gy = int(pos[1] / self.exploration_grid_size)
        gz = int(pos[2] / self.exploration_grid_size)
        return (gx, gy, gz)


    #  Baseline safety rules
    def _apply_baseline_rules(self, action, lidar):
        """
        Applies a light repulsion nudge when obstacles are close.

        The agent's action is the primary control — this just
        adds a gentle push away from detected obstacles to help
        early training. No speed reduction or altitude changes.

        Args:
            action (np.ndarray): Raw agent action, shape (3,).
            lidar (np.ndarray): LiDAR readings [F, B, L, R].

        Returns:
            np.ndarray: Modified action with repulsion adjustment.
        """
        adjusted = action.copy()
        min_lidar = np.min(lidar)

        # Repulsion from obstacle direction
        if min_lidar < self.proximity_threshold:
            front, back, left, right = lidar

            repulsion_x = 0.0
            repulsion_y = 0.0

            if front < self.proximity_threshold:
                repulsion_x -= (self.proximity_threshold - front)
            if back < self.proximity_threshold:
                repulsion_x += (self.proximity_threshold - back)
            if left < self.proximity_threshold:
                repulsion_y += (self.proximity_threshold - left)
            if right < self.proximity_threshold:
                repulsion_y -= (self.proximity_threshold - right)

            repulsion_scale = 0.3
            adjusted[0] += repulsion_x * repulsion_scale
            adjusted[1] += repulsion_y * repulsion_scale

        return adjusted


    #  Reward
    def _compute_reward(self, obs):
        """
        Computes the reward for the current step.

        The reward signal is a combination of:
            +1.0  — survival bonus (awarded every step)
            +2.0  — exploration bonus (new grid cell visited)
            +0.5  — altitude reward (at ideal cruising altitude)
            -var  — proximity penalty (scales with closeness)
            -var  — altitude penalty (scales with distance from ideal)
            -0.5  — hovering penalty (speed below 0.05 m/s)
            -50   — collision (episode terminates)
            -50   — out of bounds (episode terminates)

        Args:
            obs (np.ndarray): Current 10D observation vector.

        Returns:
            tuple[float, bool, dict]:
                reward (float): Scalar reward value.
                terminated (bool): Whether the episode ended
                    due to collision or out of bounds.
                info (dict): Additional diagnostics about what
                    triggered the reward components.
        """
        lidar = obs[:4]
        pos = obs[4:7]
        vel = obs[7:10]

        reward = 0.0
        terminated = False
        info = {}

        # Survival reward
        reward += 1.0

        # Exploration reward — strong incentive to visit new areas
        cell = self._get_grid_cell(pos)
        if cell not in self.visited_cells:
            self.visited_cells.add(cell)
            reward += 5.0
            info['new_cell'] = True

        # Movement bonus — reward any forward motion
        speed = np.linalg.norm(vel)
        if speed > 0.1:
            reward += 0.5

        # Stagnation penalty — punish camping in one spot
        if cell == self.last_cell:
            self.steps_in_current_cell += 1
        else:
            self.steps_in_current_cell = 0
            self.last_cell = cell

        if self.steps_in_current_cell > 100:
            stagnation_penalty = min((self.steps_in_current_cell - 100) * 0.05, 2.0)
            reward -= stagnation_penalty

        # Obstacle proximity penalty
        min_lidar = np.min(lidar)
        if min_lidar < self.proximity_threshold:
            proximity_penalty = (self.proximity_threshold - min_lidar) * 3.0
            reward -= proximity_penalty

        # Collision
        if min_lidar < self.collision_distance:
            reward -= 50.0
            terminated = True
            info['collision'] = True

        # Out of bounds
        if (
            pos[0] < self.boundary_min
            or pos[0] > self.boundary_max
            or pos[1] < self.boundary_min
            or pos[1] > self.boundary_max
            or pos[2] < self.min_altitude
            or pos[2] > self.max_altitude
        ):
            reward -= 50.0
            terminated = True
            info['out_of_bounds'] = True

        # Hovering penalty
        if speed < 0.05:
            reward -= 0.3

        # Altitude reward/penalty — gentle but effective
        altitude_diff = abs(pos[2] - self.ideal_altitude)
        if altitude_diff < 0.3:
            reward += 0.5
        elif altitude_diff < 1.0:
            reward -= altitude_diff * 1.0
        else:
            # Squared penalty only kicks in above 1m deviation
            reward -= (altitude_diff ** 2) * 2.0

        # Max steps
        if self.step_count >= self.max_steps:
            info['timeout'] = True

        return reward, terminated, info


    #  Gymnasium API: reset
    def reset(self, seed=None, options=None):
        """
        Resets the environment for a new episode.

        Follows the Gymnasium API:
            observation, info = env.reset()

        Stops any running simulation, restarts it, clears the
        step counter and visited cells, and places the drone
        at its starting flight height.

        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional reset options.

        Returns:
            tuple[np.ndarray, dict]:
                observation (np.ndarray): Initial 10D observation.
                info (dict): Reset diagnostics.
        """
        # Initialize the RNG (Gymnasium requirement)
        super().reset(seed=seed)

        # Connect to CoppeliaSim on first call
        self._connect()

        # Stop any running simulation
        try:
            self.sim.stopSimulation()
            while self.sim.getSimulationState() != self.sim.simulation_stopped:
                time.sleep(0.1)
        except Exception:
            pass

        # Start a fresh simulation
        self.sim.startSimulation()
        time.sleep(0.5)

        # Reset episode state
        self.step_count = 0
        self.visited_cells = set()
        self.steps_in_current_cell = 0
        self.last_cell = None

        # Place target at drone's position at starting height
        pos = get_drone_pos_array(self.sim, self.drone)
        set_target(self.sim, self.target, pos[0], pos[1], self.flight_height)

        # Build first observation
        observation = self._get_observation()
        info = {'visited_cells': 0}

        return observation, info


    #  Gymnasium API: step
    def step(self, action):
        """
        Executes one environment step.

        Follows the Gymnasium API:
            observation, reward, terminated, truncated, info = env.step(action)

        Takes the agent's raw action, applies baseline safety
        rules on top of it, scales the result, and moves the
        drone's target position accordingly. Then steps the
        simulation forward and computes the new observation
        and reward.

        Args:
            action (np.ndarray): Shape (3,) array with values
                in [-1, 1] representing velocity adjustments
                in (x, y, z).

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]:
                observation (np.ndarray): New 10D observation.
                reward (float): Reward for this step.
                terminated (bool): True if episode ended due to
                    collision or going out of bounds.
                truncated (bool): True if episode was cut short
                    because max_steps was reached.
                info (dict): Step diagnostics including step
                    count, visited cells, and min LiDAR reading.
        """
        self.step_count += 1

        # Clip raw action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Read current LiDAR for baseline rules
        lidar = read_lidar_array(self.sim, self.lidar_handles)

        # Apply baseline safety rules, then scale
        adjusted_action = self._apply_baseline_rules(action, lidar)
        scaled_action = adjusted_action * self.speed_scale

        # Get current target position and apply the action
        current_target = self.sim.getObjectPosition(
            self.target, self.sim.handle_world
        )
        new_x = current_target[0] + scaled_action[0]
        new_y = current_target[1] + scaled_action[1]
        new_z = current_target[2] + scaled_action[2]

        # Clamp altitude to allowed range
        new_z = max(self.min_altitude, min(self.max_altitude, new_z))

        set_target(self.sim, self.target, new_x, new_y, new_z)

        # Advance the simulation one step
        self.sim.step()

        # Build observation and compute reward
        observation = self._get_observation()
        reward, terminated, info = self._compute_reward(observation)
        truncated = self.step_count >= self.max_steps

        # Attach step diagnostics
        info['step'] = self.step_count
        info['visited_cells'] = len(self.visited_cells)
        info['min_lidar'] = float(np.min(observation[:4]))

        return observation, reward, terminated, truncated, info

    # ──────────────────────────────────────────────
    #  Gymnasium API: close
    # ──────────────────────────────────────────────

    def close(self):
        """
        Stops the simulation and cleans up.

        Safe to call even if the simulation is not running.
        Should be called when finished with the environment.
        """
        if self._connected:
            try:
                self.sim.stopSimulation()
            except Exception:
                pass