"""
PPO training script for drone obstacle avoidance.

The drone learns to fly freely through the environment, exploring
while avoiding obstacles using LiDAR sensor input.

Requirements:
    pip install stable-baselines3 gymnasium numpy torch

Usage:
    Train:   python train.py
    Resume:  python train.py --resume --model models/checkpoints/drone_ppo_10000_steps
    Test:    python train.py --test
    Custom:  python train.py --test --model models/checkpoints/drone_ppo_50000_steps
"""

import os
import sys
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

from drone_environment import DroneAvoidanceEnv



#  Environment Config
ENV_CONFIG = dict(
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
    ideal_altitude=1.5,                # Ideal cruising altitude — rewarded for staying near (m)
    altitude_boost_cap=2.0,            # Max altitude the baseline boost can push to (m)
    exploration_grid_size=0.5,         # Size of grid cells for exploration tracking (m)
)


#  PPO parameters
PPO_CONFIG = dict(
    learning_rate=3e-4,                # Learning rate (alpha)
    n_steps=2048,                      # Steps per rollout before policy update
    batch_size=64,                     # Minibatch size for each gradient step
    n_epochs=10,                       # Number of PPO update epochs per rollout
    gamma=0.99,                        # Discount factor for future rewards
    gae_lambda=0.95,                   # GAE lambda for advantage estimation
    clip_range=0.2,                    # PPO clipping range for policy updates
    ent_coef=0.01,                     # Entropy coefficient (encourages exploration)
    vf_coef=0.5,                       # Value function loss weight
    max_grad_norm=0.5,                 # Max gradient norm for clipping
)


#  Training Config
TOTAL_TIMESTEPS = 500_000              # Total training timesteps
CHECKPOINT_FREQ = 10_000               # Save model every N steps
HIDDEN_LAYERS = [256, 256]             # Policy and value network architecture
DEVICE = "cpu"                         # Device to train on ("cpu", "cuda", or "auto")


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback that logs additional metrics during training.

    Tracks collision count, visited grid cells, and minimum LiDAR
    distance per step for TensorBoard monitoring.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        """
        Called at every training step. Extracts info from
        completed episodes and records custom metrics.

        Returns:
            bool: Always True (continue training).
        """
        infos = self.locals.get("infos", [])
        for info in infos:
            if "collision" in info:
                self.episode_count += 1
                self.logger.record("custom/collisions", self.episode_count)
            if "visited_cells" in info:
                self.logger.record("custom/visited_cells", info["visited_cells"])
            if "min_lidar" in info:
                self.logger.record("custom/min_lidar", info["min_lidar"])
        return True


def make_env():
    """
    Creates and wraps the drone environment.

    Applies the Monitor wrapper for episode stat tracking
    (reward, length) which stable-baselines3 uses for logging.

    Returns:
        Monitor: Wrapped DroneAvoidanceEnv instance.
    """
    env = DroneAvoidanceEnv(**ENV_CONFIG)
    env = Monitor(env)
    return env


def train(resume_path=None):
    """
    Runs PPO training.

    Creates the environment, builds the PPO model with the
    configured hyperparameters, and trains for TOTAL_TIMESTEPS.
    Saves checkpoints every CHECKPOINT_FREQ steps and a final
    model when training completes or is interrupted with Ctrl+C.

    If resume_path is provided, loads a previously saved model
    and continues training from where it left off.

    Args:
        resume_path (str, optional): Path to a saved model to
            resume training from (without .zip extension).
    """

    os.makedirs("models", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Device
    if DEVICE == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = DEVICE
    print(f"Training on: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Environment
    env = make_env()

    # PPO Model — load existing or create new
    if resume_path:
        print(f"Resuming training from: {resume_path}")
        model = PPO.load(
            resume_path,
            env=env,
            device=device,
            tensorboard_log="./logs/",
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            **PPO_CONFIG,
            verbose=1,
            device=device,
            tensorboard_log="./logs/",
            policy_kwargs=dict(
                net_arch=dict(
                    pi=HIDDEN_LAYERS,
                    vf=HIDDEN_LAYERS,
                ),
                activation_fn=torch.nn.ReLU,
            ),
        )

    print("\n" + "=" * 60)
    print("  Drone Obstacle Avoidance — PPO Training")
    print("=" * 60)
    print(f"  Observation:  {env.observation_space.shape}  (4 lidar + 3 pos + 3 vel)")
    print(f"  Action:       {env.action_space.shape}  (dx, dy, dz)")
    print(f"  Device:       {device}")
    print(f"  Network:      {' → '.join(str(n) for n in HIDDEN_LAYERS)} (ReLU)")
    print(f"  LR:           {PPO_CONFIG['learning_rate']}")
    print(f"  Batch size:   {PPO_CONFIG['batch_size']}")
    print(f"  Gamma:        {PPO_CONFIG['gamma']}")
    print(f"  Max steps:    {ENV_CONFIG['max_steps']} per episode")
    print(f"  Total:        {TOTAL_TIMESTEPS:,} timesteps")
    if resume_path:
        print(f"  Resumed from: {resume_path}")
    print("=" * 60 + "\n")

    # ── Callbacks ──
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path="./models/checkpoints/",
        name_prefix="drone_ppo",
    )
    metrics_cb = TrainingMetricsCallback()

    # ── Train ──
    print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
    print("Monitor with: tensorboard --logdir ./logs/\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_cb, metrics_cb],
            progress_bar=True,
            reset_num_timesteps=resume_path is None,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving current model...")

    # ── Save ──
    model.save("models/drone_ppo_final")
    print("Model saved to models/drone_ppo_final.zip")

    env.close()


def test(model_path="models/drone_ppo_final"):
    """
    Runs a trained model in the environment.

    Loads a saved PPO model and runs a single episode with
    deterministic actions, printing live telemetry every
    50 steps.

    Args:
        model_path (str): Path to the saved model file
            (without .zip extension).
    """
    env = make_env()
    model = PPO.load(model_path, env=env)
    print(f"Loaded model: {model_path}")

    obs, info = env.reset()
    total_reward = 0
    step = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if step % 50 == 0:
            lidar = obs[:4]
            print(
                f"  Step {step:4d} | "
                f"Reward: {total_reward:7.1f} | "
                f"Cells: {info.get('visited_cells', 0):3d} | "
                f"LiDAR  F:{lidar[0]:.2f}  B:{lidar[1]:.2f}  "
                f"L:{lidar[2]:.2f}  R:{lidar[3]:.2f}"
            )

        if terminated or truncated:
            print(f"\n  Episode done: {info}")
            print(f"  Total reward:   {total_reward:.1f}")
            print(f"  Steps survived: {step}")
            print(f"  Cells explored: {info.get('visited_cells', 0)}")
            break

    env.close()


if __name__ == "__main__":
    if "--test" in sys.argv:
        path = "models/drone_ppo_final"
        for i, arg in enumerate(sys.argv):
            if arg == "--model" and i + 1 < len(sys.argv):
                path = sys.argv[i + 1]
        test(path)
    elif "--resume" in sys.argv:
        path = "models/drone_ppo_final"
        for i, arg in enumerate(sys.argv):
            if arg == "--model" and i + 1 < len(sys.argv):
                path = sys.argv[i + 1]
        train(resume_path=path)
    else:
        train()