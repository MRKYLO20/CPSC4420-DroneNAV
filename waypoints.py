def load_waypoints(sim, waypoint_names, flight_height):
    """Loads waypoint positions from CoppeliaSim dummy objects.

    For each named dummy, retrieves its world position. If the
    z-coordinate is too low, it is overridden with
    ``flight_height`` to ensure the drone stays airborne.

    Args:
        sim: CoppeliaSim remote API object.
        waypoint_names (list[str]): Object path names to look up,
            e.g. ``['/pos1', '/pos2']``.
        flight_height (float): Altitude in metres to assign when a
            waypoint's z-value is below 0.5 m.

    Returns:
        list[list[float]]: List of ``[x, y, z]`` waypoint positions
            in world coordinates. Waypoints that could not be loaded
            are omitted.
    """
    waypoints = []
    for name in waypoint_names:
        try:
            handle = sim.getObject(name)
            pos = sim.getObjectPosition(handle, sim.handle_world)
            if pos[2] < 0.5:
                pos[2] = flight_height
            waypoints.append(pos)
            print(f'Loaded waypoint {name}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}')
        except Exception as e:
            print(f'Could not load {name}: {e}')
    return waypoints