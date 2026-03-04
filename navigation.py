import math


def get_drone_pos(sim, drone_handle):
    return sim.getObjectPosition(drone_handle, sim.handle_world)


def set_target(sim, target_handle, x, y, z):
    sim.setObjectPosition(target_handle, sim.handle_world, [x, y, z])


def distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def move_toward(sim, target_handle, waypoint, speed):
    """Moves the target incrementally toward a waypoint."""
    current = sim.getObjectPosition(target_handle, sim.handle_world)
    new_x = current[0] + (waypoint[0] - current[0]) * speed
    new_y = current[1] + (waypoint[1] - current[1]) * speed
    new_z = current[2] + (waypoint[2] - current[2]) * speed
    set_target(sim, target_handle, new_x, new_y, new_z)
