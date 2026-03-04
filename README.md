# CoppeliaSim Drone Waypoint Navigation

A Python-based quadcopter controller for CoppeliaSim that autonomously navigates through a series of waypoints using vision-based LiDAR sensors for distance feedback.

## Prerequisites

- [CoppeliaSim](https://www.coppeliarobotics.com/) (EDU or Player)
- Python 3.8+

## Installation

```bash
pip install coppeliasim-zmqremoteapi-client numpy
```

## CoppeliaSim Scene Setup

The scene should contain the following objects:

- `/Quadcopter` — the drone model (with a built-in flight script)
- `/target` — a dummy object the drone's flight script follows
- `/Quadcopter/base/lidarLeft/body/sensor` — left LiDAR vision sensor
- `/Quadcopter/base/lidarRight/body/sensor` — right LiDAR vision sensor
- `/pos1` through `/pos6` — dummy objects used as waypoints

## Usage

1. Open your scene in CoppeliaSim.
2. Make sure the ZeroMQ remote API is enabled (it is by default in recent versions).
3. Run the controller:

```bash
python drone_controller.py
```

The drone will take off, visit each waypoint in sequence, and land when finished. Live distance and LiDAR readings are printed to the terminal.

## Project Structure

| File | Description |
|------|-------------|
| `drone_controller.py` | Main entry point — connects to CoppeliaSim and runs the waypoint loop |
| `navigation.py` | Position reading, distance calculation, and target movement |
| `lidar.py` | Reads and formats depth data from vision-based LiDAR sensors |
| `waypoints.py` | Loads waypoint positions from CoppeliaSim dummy objects |

## Configuration

Key parameters can be adjusted at the top of `drone_controller.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FLIGHT_HEIGHT` | `1.5` | Altitude (m) assigned to low waypoints |
| `REACH_THRESHOLD` | `0.5` | Distance (m) at which a waypoint is considered reached |
| `SPEED` | `0.005` | Movement interpolation factor (0–1) |
| `PAUSE_AT_WP` | `0` | Seconds to pause at each waypoint |
