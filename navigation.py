import math


def get_drone_pos(sim, drone_handle):
    """
    Gets the world-frame position of the drone.

    Args:
        sim: CoppeliaSim remote API object.
        drone_handle (int): Handle to the drone object.

    Returns:
        list[float]: ``[x, y, z]`` position in world coordinates.
    """
    return sim.getObjectPosition(drone_handle, sim.handle_world)


def set_target(sim, target_handle, x, y, z):
    """
    Sets the target position in world coordinates.

    Args:
        sim: CoppeliaSim remote API object.
        target_handle (int): Handle to the target dummy object.
        x (float): Target x-coordinate in metres.
        y (float): Target y-coordinate in metres.
        z (float): Target z-coordinate in metres.
    """
    sim.setObjectPosition(target_handle, sim.handle_world, [x, y, z])


def distance(a, b):
    """Calculates the Euclidean distance between two 3-D points.

    Args:
        a (list[float]): First point as ``[x, y, z]``.
        b (list[float]): Second point as ``[x, y, z]``.

    Returns:
        float: Straight-line distance between *a* and *b*.
    """
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def move_toward(sim, target_handle, waypoint, speed):
    """
    Moves the target incrementally toward a waypoint.

    Applies linear interpolation between the current target position
    and the desired waypoint, scaled by ``speed``.

    Args:
        sim: CoppeliaSim remote API object.
        target_handle (int): Handle to the target dummy object.
        waypoint (list[float]): Desired ``[x, y, z]`` position.
        speed (float): Interpolation factor in the range ``(0, 1]``.
            Smaller values produce smoother, slower movement.
    """
    current = sim.getObjectPosition(target_handle, sim.handle_world)
    new_x = current[0] + (waypoint[0] - current[0]) * speed
    new_y = current[1] + (waypoint[1] - current[1]) * speed
    new_z = current[2] + (waypoint[2] - current[2]) * speed
    set_target(sim, target_handle, new_x, new_y, new_z)