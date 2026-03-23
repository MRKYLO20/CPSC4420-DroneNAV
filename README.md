# CoppeliaSim Drone Obstacle Avoidance

A reinforcement learning system for training a quadcopter in CoppeliaSim to autonomously explore and avoid obstacles using PPO (Proximal Policy Optimization) and LiDAR sensor feedback.

## Prerequisites

- [CoppeliaSim](https://www.coppeliarobotics.com/) (EDU or Player)
- Python 3.10+
- NVIDIA GPU recommended (CUDA support for faster training)

## Installation

```bash
pip install coppeliasim-zmqremoteapi-client numpy gymnasium stable-baselines3 tensorboard
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## CoppeliaSim Scene Setup

The scene should contain the following objects:

- `/Quadcopter` — the drone model (with a built-in flight script)
- `/target` — a dummy object the drone's flight script follows (set to non-detectable so LiDAR ignores it)
- `/Quadcopter/base/lidarFront/body/sensor` — front LiDAR vision sensor
- `/Quadcopter/base/lidarBack/body/sensor` — back LiDAR vision sensor
- `/Quadcopter/base/lidarLeft/body/sensor` — left LiDAR vision sensor
- `/Quadcopter/base/lidarRight/body/sensor` — right LiDAR vision sensor
- Static obstacles (trees, buildings, etc.) and dynamic obstacles (people on paths)

## Usage

### Training

1. Open your scene in CoppeliaSim (do not start the simulation).
2. Run the training script:

```bash
python train.py
```

The script will automatically start/stop the simulation, train the drone using PPO, and save model checkpoints to `./models/checkpoints/`. Training progress can be monitored with TensorBoard:

```bash
tensorboard --logdir ./logs/
```

### Testing a Trained Model

```bash
python train.py --test
python train.py --test --model models/checkpoints/drone_ppo_50000_steps
```

### Resuming Training

If training was interrupted or you want to continue training from a checkpoint:

```bash
python train.py --resume --model models/checkpoints/drone_ppo_10000_steps
python train.py --resume --model models/drone_ppo_final
```

This loads the saved model weights and continues training for another `TOTAL_TIMESTEPS` steps. TensorBoard logs will continue in the same run.

## How It Works

The drone learns to fly freely through the environment, exploring while avoiding obstacles. There are no waypoints or predefined paths — the agent learns purely from experience.

**Observation (10D):** 4 LiDAR distances (front, back, left, right), 3D drone position, and 3D drone velocity.

**Action (3D):** Velocity adjustments in x, y, and z, applied on top of baseline safety rules.

**Reward:** The agent earns rewards for surviving each timestep and exploring new areas, while receiving penalties for getting close to obstacles, collisions, going out of bounds, or hovering in place.

**Baseline Safety Rules:** Rule-based behaviors (slow down near obstacles, gain altitude when very close, push away from obstacle direction) provide a foundation that the RL agent learns to work with or override.

## Structure

| File | Description |
|------|-------------|
| `train.py` | PPO training and testing — all tunable config at the top |
| `drone_environment.py` | Gymnasium environment wrapping CoppeliaSim for RL training |
| `navigation.py` | Position reading, velocity, distance calculation, and target movement |
| `lidar.py` | Reads and formats depth data from 4 vision-based LiDAR sensors |

## Configuration

### Environment Parameters

Adjustable at the top of `train.py` in `ENV_CONFIG`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | `2000` | Max steps before episode ends |
| `speed_scale` | `0.05` | Scales agent action into movement |
| `collision_distance` | `0.3` | LiDAR distance that counts as a crash (m) |
| `proximity_threshold` | `1.0` | Distance to start avoiding obstacles (m) |
| `altitude_boost_threshold` | `0.5` | Distance to start gaining altitude (m) |
| `flight_height` | `1.5` | Starting flight height (m) |
| `exploration_grid_size` | `0.5` | Grid cell size for exploration tracking (m) |

### PPO Hyperparameters

Adjustable at the top of `train.py` in `PPO_CONFIG`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `3e-4` | Learning rate (alpha) |
| `n_steps` | `2048` | Steps per rollout before policy update |
| `batch_size` | `64` | Minibatch size for each gradient step |
| `n_epochs` | `10` | PPO update epochs per rollout |
| `gamma` | `0.99` | Discount factor for future rewards |
| `ent_coef` | `0.01` | Entropy coefficient (encourages exploration) |