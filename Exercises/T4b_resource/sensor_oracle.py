import numpy as np
from constants import SENSOR_COST, FREE_COST

# START OF MY CODE

def path_evaluation(gallery, path_distribution):
    """
    Method to evaluate the cost of path distribution for each sensor

    Parameters
    ----------
    gallery: Gallery
        Map of the environment with information about sensors of given size
    path_distribution: list(list(double), list(tuple(int, int))
        List of two lists. The paths are in the second list, and in the first list are probabilities corresponding to those paths.
    Returns
    -------
    sensor_counts: list(int)
        List of the value of each sensor against given path distribution
    """

    sensor_counts = np.zeros(gallery.num_sensors)
    for s in range(gallery.num_sensors):
        for p in range(len(path_distribution[1])):
            path = path_distribution[1][p]
            prob = path_distribution[0][p]
            for step in path:
                sensor_counts[s] += prob * FREE_COST
                if step in gallery.sensor_coverage[s]:
                    sensor_counts[s] += prob * SENSOR_COST
    return sensor_counts #higher number means more expensive play for the opponent

def best_sensor(gallery, path_distribution):
    """
    Method to pick the best sensor against the current path distribution

    Parameters
    ----------
    gallery: Gallery
        Map of the environment with information about sensors of given size
    path_distribution: list(list(double), list(tuple(int, int))
        List of two lists. The paths are in the second list, and in the first list are probabilities corresponding to those paths.
    Returns
    -------
    sensor: int
        Index of the sensor, which achieves the best value against the path distribution.
    reward: double
        Value achieved by the best sensor

    """
    sensor_counts = path_evaluation(gallery, path_distribution)
    sensor = np.argmax(sensor_counts)
    reward = sensor_counts[sensor]
    return sensor, reward

def pair_evaluation(gallery, path, sensor):
    """
    Method to evaluate the value of path-sensor pair

    Parameters
    ----------
    gallery: Gallery
        Map of the environment with information about sensors of a given size
    path: list(tuple(int, int))
        List of coordinates forming the path.
    sensor: int
        Index of the sensor.
    Returns
    -------
    cost: double
        Cost of the path assuming the given sensor is active.
    """
    
    cost = 0
    for step in path:
        cost += FREE_COST
        if step in gallery.sensor_coverage[sensor]:
            cost += SENSOR_COST
    return cost

# END OF MY CODE