import numpy as np
import queue
import copy
from constants import FREE_COST,SENSOR_COST

# START OF MY CODE
def cost_function(gallery, sensor_distribution):
    """
    Method to calculate the cost of the cells in the grid given the current sensor distribution

    Parameters
    ----------
    gallery: Gallery
        Map of the environment with information about sensors of given size
    sensor_distribution: list(list(double), list(int))
        List of two lists. In the second list are the sensors, and the first list is the corresponding sensor's probability.
        doubles = probabilities
        int = which sensor group
    Returns
    -------
    cost: Matrix of the same size as the map in the gallery
        Matrix filled by cost for each grid cell in the environment. (Costs are set in constants FREE_COST and SENSOR_COST.
        The cost for accessing a cell is FREE_COST + SENSOR_COST * probability of the sensor being active in the cell.)
    """

    # gallery.sensor_coverage is a list of lists, where the main list has the size num_sensors
    # each sublist contains the coordinates of the cells covered by the sensor
    m = np.full((gallery.y_size, gallery.x_size), FREE_COST)
    for y in range(gallery.y_size):
        for x in range(gallery.x_size):
            for s in range(len(sensor_distribution[1])):
                sensor_id = sensor_distribution[1][s]
                if (y, x) in gallery.sensor_coverage[sensor_id]:
                    prob = sensor_distribution[0][s]
                    m[y, x] += SENSOR_COST * prob
    return m



def best_plan(gallery, cost_matrix):
    """
    Method to calculate the best path to the goal and back given the current cost function

    Parameters
    ----------
    gallery: Gallery
        Map of the environment with information about sensors of a given size
    cost_matrix: Matrix of the same size as the map in the gallery
        Matrix capturing the cost of each cell
    Returns
    -------
    path: list(tuple(int, int))
        List of coordinates visited on the best path
    value: double
        Value of the best path given the cost matrix
    """
    ret_path = [] # path which we choose
    ret_value = np.inf # minimal path value

    for i in range(len(gallery.entrances)):

        # INIT NEW VALUES
        (x,y) = gallery.entrances[i]
        q = queue.PriorityQueue() #we pop the lowest value always the first
        q.put((cost_matrix[(x,y)],(x,y), [(x,y)]))
        closed = set()

        # FIND SHORTEST PATH
        while True:
            value,(x,y), path = q.get()
            if (x,y) in closed:
                continue
            if (x,y) in gallery.goals:
                if value < ret_value:
                    ret_path = path
                    ret_value = value
                break #we know the shortest path because of the priority queue
            closed.add((x,y))
            for neighbour in passable_neighbours((x,y), gallery):
                new_path = copy.copy(path)
                new_path.append(neighbour)
                q.put((value+cost_matrix[(x,y)],neighbour, new_path))

    ret_value = 2*ret_value - cost_matrix[(x,y)] #we don't want to count the goal twice
    return ret_path + list(reversed(ret_path[:-1])), ret_value

# END OF MY CODE


def passable_neighbours(cell, gallery):
    neighbours = []
    for move in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        possible_neighbour = (cell[0] + move[0], cell[1] + move[1])
        if gallery.is_correct_cell(possible_neighbour) and gallery.is_passable(possible_neighbour):
            neighbours.append(possible_neighbour)
    return neighbours
