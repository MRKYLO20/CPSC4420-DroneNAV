def load_waypoints(sim, waypoint_names, flight_height):
    """Loads waypoint positions from CoppeliaSim dummy objects."""
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
