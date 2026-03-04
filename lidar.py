import numpy as np


def get_lidar_min_distance(sim, lidar_handle):
    """
    Retrieves depth buffer data from a CoppeliaSim vision sensor,
    unpacks the raw bytes as float32, and returns the smallest
    valid distance reading.

    Args:
        sim: CoppeliaSim remote API object.
        lidar_handle (int): Handle to the vision sensor object.

    Returns:
        float: The minimum valid depth reading in metres,
            or None if no valid reading is available.
    """
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
    Reads all LiDAR sensors and returns minimum distances.

    Args:
        sim: CoppeliaSim remote API object.
        lidar_handles (dict[str, int]): Mapping of sensor name to
            its CoppeliaSim object handle.

    Returns:
        dict[str, float | None]: Mapping of sensor name to its
            minimum distance reading, or None if unavailable.
    """
    readings = {}
    for name, handle in lidar_handles.items():
        readings[name] = get_lidar_min_distance(sim, handle)
    return readings


def format_lidar_status(lidar_data):
    """Returns a formatted string of LiDAR readings.

    Args:
        lidar_data (dict[str, float | None]): Mapping of sensor name
            to distance value as returned by ``read_lidar``.

    Returns:
        str: Pipe-separated status string,
            e.g. ``"L: 1.23m | R: N/A"``.
    """
    parts = []
    for name, dist in lidar_data.items():
        if dist is not None:
            parts.append(f"{name}: {dist:.2f}m")
        else:
            parts.append(f"{name}: N/A")
    return " | ".join(parts)