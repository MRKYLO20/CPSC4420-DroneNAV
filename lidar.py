import numpy as np


def get_lidar_min_distance(sim, lidar_handle):
    """Reads the minimum distance from a vision-based LiDAR sensor."""
    try:
        result = sim.getVisionSensorDepth(lidar_handle, 0)
        if result is None:
            return None

        depth_data, resolution = result[0], result[1]

        if not depth_data or len(depth_data) == 0:
            return None

        # depth_data is a raw byte buffer — unpack it as float32
        depth_array = np.frombuffer(depth_data, dtype=np.float32)
        valid = depth_array[depth_array > 0.01]

        if len(valid) == 0:
            return None

        return float(np.min(valid))

    except Exception as e:
        print(f'  LiDAR read error: {e}')
        return None


def read_lidar(sim, lidar_handles):
    """
    Reads all lidar sensors and returns minimum distances.

    Args:
        sim: CoppeliaSim API object
        lidar_handles: dict of {'name': handle} pairs

    Returns:
        dict of {'name': min_distance} pairs
    """
    readings = {}
    for name, handle in lidar_handles.items():
        readings[name] = get_lidar_min_distance(sim, handle)
    return readings


def format_lidar_status(lidar_data):
    """Returns a formatted string of LiDAR readings."""
    parts = []
    for name, dist in lidar_data.items():
        if dist is not None:
            parts.append(f"{name}: {dist:.2f}m")
        else:
            parts.append(f"{name}: N/A")
    return " | ".join(parts)