import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from navigation import get_drone_pos, set_target, distance, move_toward
from waypoints import load_waypoints
from lidar import read_lidar, format_lidar_status

# ── Connection ──
client = RemoteAPIClient()
sim = client.require('sim')

# ── Config ──
FLIGHT_HEIGHT = 1.5
REACH_THRESHOLD = 0.5
PAUSE_AT_WP = 0
SPINUP_DELAY = 0
SPEED = 0.005

WAYPOINT_NAMES = ['/pos1', '/pos2', '/pos3', '/pos4', '/pos5', '/pos6']

# ── Handles ──
drone  = sim.getObject('/Quadcopter')
target = sim.getObject('/target')

lidar_handles = {
    'L': sim.getObject('/Quadcopter/base/lidarLeft/body/sensor'),
    'R': sim.getObject('/Quadcopter/base/lidarRight/body/sensor'),
    'F': sim.getObject('/Quadcopter/base/lidarFront/body/sensor'),
    'B': sim.getObject('/Quadcopter/base/lidarBack/body/sensor'),
}


def run_simulation():
    """
    Runs the full waypoint-following simulation.

    Loads waypoints, starts the CoppeliaSim simulation, and
    navigates the drone through each waypoint while
    printing live distance and LiDAR data. After all waypoints
    are reached the drone descends to ground level and the
    simulation is stopped.
    """
    print('Loading waypoint(s)...')
    waypoints = load_waypoints(sim, WAYPOINT_NAMES, FLIGHT_HEIGHT)

    if not waypoints:
        print('No waypoints found, aborting.')
        return

    print(f'\n{len(waypoints)} waypoints loaded. Starting simulation...')
    sim.startSimulation()
    time.sleep(SPINUP_DELAY)

    last_print = 0

    for idx, wp in enumerate(waypoints):
        print(f'\nHeading to waypoint [{idx + 1}/{len(waypoints)}]')

        while True:
            pos = get_drone_pos(sim, drone)
            d = distance(pos, wp)
            move_toward(sim, target, wp, SPEED)

            # Throttled status output — overwrites the same line
            if time.time() - last_print > 0.5:
                lidar_data = read_lidar(sim, lidar_handles)
                lidar_str = format_lidar_status(lidar_data)
                status = f'  Dist: [{d:.3f} m] | LiDAR {lidar_str}'
                print(f'\r{status}', end='', flush=True)
                last_print = time.time()

            if d < REACH_THRESHOLD:
                print(f'\n  Reached waypoint [{idx + 1}]')
                time.sleep(PAUSE_AT_WP)
                break

    print('\nAll waypoints reached: Landing...')
    pos = get_drone_pos(sim, drone)
    set_target(sim, target, pos[0], pos[1], 0.15)
    time.sleep(2)

    print('Simulation Concluded')
    sim.stopSimulation()


run_simulation()