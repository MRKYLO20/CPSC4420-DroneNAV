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
