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
    +5.0  for visiting a new grid cell
    +0.5  movement bonus (speed above 0.1 m/s)
    -var  proximity penalty (scales with closeness to obstacles)
    -var  stagnation penalty (camping in one cell)
    -var  altitude penalty (scales with distance from ideal)
    -0.3  hovering penalty (speed below 0.05 m/s)
    -50   collision (episode terminates)
    -50   out of bounds (episode terminates)

Baseline safety rules:
    - Light horizontal repulsion when obstacles are detected
"""

import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from lidar import get_lidar_handles, read_lidar_array, set_lidar_resolution
from navigation import get_drone_pos_array, get_drone_velocity, set_target


class DroneAvoidanceEnv(gym.Env):
    """
    CoppeliaSim drone environment for obstacle avoidance.

    Implements the standard Gymnasium Env interface so it can be
    used with any compatible RL library (stable-baselines3, etc).

    The agent controls the drone's velocity in 3D space. Altitude
    is clamped and penalized to discourage cheating by flying above
    obstacles. Baseline repulsion provides a light horizontal nudge.
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
        profile_every=0,                   # Print timing stats every N steps (0 disables)
        disable_visualization=True,        # Toggle visualization off on reset
        lidar_resolution=32,               # Vision sensor resolution (square). None to skip
        headless=False,                    # Skip display toggles when running headless
        host="localhost",                  # Remote API host
        port=23000,                        # Remote API port (unique per sim instance)
        cntport=None,                      # Remote API control port (optional)
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
        self.profile_every = profile_every
        self.disable_visualization = disable_visualization
        self.lidar_resolution = lidar_resolution
        self.headless = headless
        self.host = host
        self.port = port
        self.cntport = cntport

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

        # Profiling accumulators (seconds)
        self._profile = {
            "steps": 0,
            "lidar_baseline": 0.0,
            "target_move": 0.0,
            "sim_step": 0.0,
            "observation": 0.0,
            "reward": 0.0,
            "total": 0.0,
        }

    #  Connection
    def _connect(self):
        """
        Establishes connection to CoppeliaSim.

        Grabs handles for the drone, target dummy, and all four
        LiDAR sensors. Skips if already connected.
        """
        if self._connected:
            return

        self.client = RemoteAPIClient(host=self.host, port=self.port, cntport=self.cntport)
        self.sim = self.client.require('sim')

        self.drone = self.sim.getObject('/Quadcopter')
        self.target = self.sim.getObject('/target')
        self.lidar_handles = get_lidar_handles(self.sim)

        self._connected = True

    def _apply_sim_settings(self):
        """
        Applies sim settings that tend to reset on simulation start.
        """
        if self.disable_visualization and not self.headless:
            try:
                self.sim.setBoolParam(self.sim.boolparam_display_enabled, False)
            except Exception as e:
                print(f"  Visualization toggle failed: {e}")

        if self.lidar_resolution:
            set_lidar_resolution(self.sim, self.lidar_handles, self.lidar_resolution)

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

            repulsion_scale = 0.8
            adjusted[0] += repulsion_x * repulsion_scale
            adjusted[1] += repulsion_y * repulsion_scale

        return adjusted


    #  Reward
    def _compute_reward(self, obs):
        """
        Computes the reward for the current step.

        The reward signal is a combination of:
            +1.0  — survival bonus (awarded every step)
            +5.0  — exploration bonus (new grid cell visited)
            +0.5  — movement bonus (speed above 0.1 m/s)
            -var  — proximity penalty (scales with closeness)
            -var  — stagnation penalty (camping in one cell)
            -var  — altitude penalty (scales with distance from ideal)
            -0.3  — hovering penalty (speed below 0.05 m/s)
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

        # Reward tuning parameters (grouped for easy adjustment)
        survival_reward = 1.0
        exploration_reward = 12
        movement_reward = 1
        stagnation_start = 100
        stagnation_rate = 0.05
        stagnation_cap = 2.0
        proximity_penalty_scale = 3.5
        collision_penalty = 50.0
        out_of_bounds_penalty = 50.0
        hovering_penalty = 0.25
        altitude_bonus = 0.5
        altitude_soft_band = 0.3
        altitude_linear_band = 1.0
        altitude_linear_scale = 1.5
        altitude_quadratic_scale = 3.0

        reward = 0.0
        terminated = False
        info = {}

        # Survival reward
        reward += survival_reward

        # Exploration reward — strong incentive to visit new areas
        cell = self._get_grid_cell(pos)
        if cell not in self.visited_cells:
            self.visited_cells.add(cell)
            reward += exploration_reward
            info['new_cell'] = True

        # Movement bonus — reward any horizontal motion
        speed = np.linalg.norm(vel[:2])
        if speed > 0.1:
            reward += movement_reward

        # Stagnation penalty — punish camping in one spot
        if cell == self.last_cell:
            self.steps_in_current_cell += 1
        else:
            self.steps_in_current_cell = 0
            self.last_cell = cell

        if self.steps_in_current_cell > stagnation_start:
            stagnation_penalty = min(
                (self.steps_in_current_cell - stagnation_start) * stagnation_rate,
                stagnation_cap,
            )
            reward -= stagnation_penalty

        # Obstacle proximity penalty
        min_lidar = np.min(lidar)
        if min_lidar < self.proximity_threshold:
            proximity_penalty = (
                self.proximity_threshold - min_lidar
            ) * proximity_penalty_scale
            reward -= proximity_penalty

        # Collision
        if min_lidar < self.collision_distance:
            reward -= collision_penalty
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
            reward -= out_of_bounds_penalty
            terminated = True
            info['out_of_bounds'] = True

        # Hovering penalty
        if speed < 0.1:
            reward -= hovering_penalty

        # Altitude reward/penalty — keep drone near ideal height
        altitude_diff = abs(pos[2] - self.ideal_altitude)
        if altitude_diff < altitude_soft_band:
            reward += altitude_bonus
        elif altitude_diff < altitude_linear_band:
            reward -= altitude_diff * altitude_linear_scale
        else:
            # Squared penalty kicks in hard above 1m deviation
            reward -= (altitude_diff ** 2) * altitude_quadratic_scale

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

        # Apply sim settings before starting
        self._apply_sim_settings()

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

        do_profile = self.profile_every and self.profile_every > 0
        if do_profile:
            t_total_start = time.perf_counter()

        # Clip raw action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Read current LiDAR for baseline rules
        if do_profile:
            t0 = time.perf_counter()
        lidar = read_lidar_array(self.sim, self.lidar_handles)
        if do_profile:
            self._profile["lidar_baseline"] += time.perf_counter() - t0

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

        if do_profile:
            t0 = time.perf_counter()
        set_target(self.sim, self.target, new_x, new_y, new_z)
        if do_profile:
            self._profile["target_move"] += time.perf_counter() - t0

        # Advance the simulation one step
        if do_profile:
            t0 = time.perf_counter()
        self.sim.step()
        if do_profile:
            self._profile["sim_step"] += time.perf_counter() - t0

        # Build observation and compute reward
        if do_profile:
            t0 = time.perf_counter()
        observation = self._get_observation()
        if do_profile:
            self._profile["observation"] += time.perf_counter() - t0

        if do_profile:
            t0 = time.perf_counter()
        reward, terminated, info = self._compute_reward(observation)
        if do_profile:
            self._profile["reward"] += time.perf_counter() - t0
        truncated = self.step_count >= self.max_steps

        # Attach step diagnostics
        info['step'] = self.step_count
        info['visited_cells'] = len(self.visited_cells)
        info['min_lidar'] = float(np.min(observation[:4]))

        if do_profile:
            self._profile["steps"] += 1
            self._profile["total"] += time.perf_counter() - t_total_start

            if self._profile["steps"] % self.profile_every == 0:
                steps = self._profile["steps"]
                def ms(key):
                    return (self._profile[key] / steps) * 1000.0

                print(
                    "Timing avg (ms/step) | "
                    f"total {ms('total'):.2f} | "
                    f"lidar {ms('lidar_baseline'):.2f} | "
                    f"target {ms('target_move'):.2f} | "
                    f"sim.step {ms('sim_step'):.2f} | "
                    f"obs {ms('observation'):.2f} | "
                    f"reward {ms('reward'):.2f}"
                )

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
