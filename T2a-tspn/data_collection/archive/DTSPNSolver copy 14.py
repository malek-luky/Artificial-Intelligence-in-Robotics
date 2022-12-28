# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import math
from time import sleep
from itertools import permutations
from itertools import product

#import communication messages
from messages import *

from invoke_LKH import solve_TSP
import dubins

def pose_to_se2(pose):
    return pose.position.x,pose.position.y,pose.orientation.to_Euler()[0]

def se2_to_pose(se2):
    pose = Pose()
    pose.position.x = se2[0]
    pose.position.y = se2[1]
    pose.orientation.from_Euler(se2[2],0,0) 
    return pose


def configurations_to_path(configurations, turning_radius):  
    """
    Compute a closed tour through the given configurations and turning radius, 
    and return densely sampled configurations and length.  

    Parameters
    ----------
    configurations: list Pose
        list of robot configurations in SE3 coordinates (limited to SE2 equivalents), one for each goal
    turning_radius: float
        turning radius for the Dubins vehicle model  

    Returns
    -------
    Path, path_length (float)
        tour as a list of densely sampled robot configurations in SE3 coordinates
    """  
    N = len(configurations)
    path = []
    path_len = 0.
    for a in range(N):
        b = (a+1) % N
        start = pose_to_se2(configurations[a])
        end = pose_to_se2(configurations[b])
        step_size = 0.01 * turning_radius
        dubins_path = dubins.shortest_path(start, end, turning_radius)
        segment_len = dubins_path.path_length()
        step_configurations, _ = dubins_path.sample_many(step_size)
        path = path + [se2_to_pose(sc) for sc in step_configurations]
        path_len += segment_len
    return path, path_len

def create_samples(goals, sensing_radius, position_resolution, heading_resolution):
    """
    Sample the goal regions on the boundary using uniform distribution.

    Parameters
    ----------
    goals: list Vector3
        list of the TSP goal coordinates (x, y, 0)
    sensing_radius: float
        neighborhood of TSP goals  
    position_resolution: int
        number of location at the region's boundary
    heading_resolution: int
        number of heading angles per location
    
    Returns
    -------
    matrix[target_idx][sample_idx] Pose
        2D matrix of configurations in SE3
    """ 
    samples = []
    for idx, g in enumerate(goals):
        samples.append([])
        for sp in range(position_resolution):
            alpha = sp * 2*math.pi / position_resolution
            position = [g.x,g.y] + sensing_radius * np.array([math.cos(alpha), math.sin(alpha)])
            for sh in range(heading_resolution):
                heading = sh * 2*math.pi / heading_resolution
                sample = se2_to_pose((position[0], position[1], heading))
                samples[idx].append(sample)
    return samples

def plan_tour_decoupled(goals, sensing_radius, turning_radius):
    """
    Compute a DTSPN tour using the decoupled approach.  

    Parameters
    ----------
    goals: list Vector3
        list of the TSP goal coordinates (x, y, 0)
    sensing_radius: float
        neighborhood of TSP goals  
    turning_radius: float
        turning radius for the Dubins vehicle model  

    Returns
    -------
    Path, path_length (float)
        tour as a list of robot configurations in SE3 densely sampled
    """

    N = len(goals)

    # find path between each pair of goals (a,b)
    etsp_distances = np.zeros((N,N))	
    for a in range(0,N): 
        for b in range(0,N):
            g1 = goals[a]
            g2 = goals[b]
            etsp_distances[a][b] = (g1-g2).norm() 
        
    # Example how to print a small matrix with fixed precision
    np.set_printoptions(precision=2)
    # print("ETSP distances")
    # print(etsp_distances)

    sequence = solve_TSP(etsp_distances) #returns best possible sequence for visiting the goals
    # print("ETSP sequence")
    # print(sequence)
   
    '''
    TODO - homework (Decoupled)
        1) Sample the configurations ih the goal areas (already prepared)
        2) Find the shortest tour
        3) Return the final tour as the points samples (step = 0.01 * radius)
    '''
    position_resolution = heading_resolution = 8
    samples = create_samples(goals, sensing_radius, position_resolution, heading_resolution)

    # START PF MY CODE
    samples = [samples[x] for x in sequence] # reorder based on the traveling salesman problem (TSP)
    samples_per_point = len(samples[0])
    number_of_points = len(samples)

    #MAP THE SE2 INSTANCES TO MAKE IT FASTER
    se2 = []
    for k in samples:
        se2.append(list(map(lambda x: pose_to_se2(x), k)))

    best_path = 99999
    selected_samples = [0] * N #PROVIDED BY THE TEACHER    

    # CHECK ALL POSSIBLE ROUTES
    # row = number of points
    # col = number of samples per point
    for start in range(samples_per_point):
        length_matrix = 99*np.ones([number_of_points,samples_per_point]) #matrix of length for each point
        previos_node_matrix = 99999*np.zeros([number_of_points,samples_per_point]) #matrix of previous node for each point - using 99999 we get an error if we use wrong node

        # Get the distance for the first row
        for col in range(samples_per_point):
            distance = dubins.shortest_path(se2[0][start], se2[1][col], turning_radius).path_length()
            if distance <= length_matrix[1][col]:
                length_matrix[1][col] = distance
                previos_node_matrix[1][col] = start

        # Get the distance for the middle        
        for row in range (1,number_of_points-1):
            for col in range(samples_per_point):
                for next_col in range(samples_per_point):
                    distance = length_matrix[row][col]+ dubins.shortest_path(se2[row][col], se2[row+1][next_col], turning_radius).path_length()
                    if distance < length_matrix[row+1][next_col]:
                        length_matrix[row+1][next_col] = distance
                        previos_node_matrix[row+1][next_col] = col
        
        # Get the distance for the last row
        for col in range(samples_per_point):
            distance = length_matrix[-1][col] + dubins.shortest_path(se2[-1][col], se2[0][start], turning_radius).path_length()
            if distance < length_matrix[0][start]:
                length_matrix[0][start] = distance
                previos_node_matrix[0][start] = col

        # Save the shortest path and samples
        if length_matrix[0][start] < best_path:
            best_path = length_matrix[0][start]
            # FIND THE END
            index = np.argmin(length_matrix[0])
            index = int(previos_node_matrix[0][index])
            selected_samples[-1] = index
            # ITERATE FROM THE END TO THE START NODE
            for i in range(number_of_points-1,0,-1):
                index = int(previos_node_matrix[i][index])
                selected_samples[i-1] = index
            # for i in range(len(length_matrix)):
            #     index = np.argmin(length_matrix[i]) #TODO: ZDE JE CHYBA MUHAHAHAHAHA, 10h života, necham si tu na vzpominku :)
            #     selected_samples[i-1] = int(previos_node_matrix[i][index])

    # Check the distance
    print("Best path", best_path)
    distance = 0
    for i in range(number_of_points-1):
        #print("ZDE",distance,selected_samples[i])
        distance+=dubins.shortest_path(se2[i][selected_samples[i]], se2[i+1][selected_samples[i+1]], turning_radius).path_length()
    distance+=dubins.shortest_path(se2[-1][selected_samples[-1]], se2[0][selected_samples[0]], turning_radius).path_length()
    #print("FINAL",distance,selected_samples[-1])
    print("SAMPLES:",selected_samples)
    
    configurations = []
    for idx in range(N):
        configurations.append (samples[idx][selected_samples[idx]])
    return configurations_to_path(configurations, turning_radius)

def plan_tour_noon_bean(goals, sensing_radius, turning_radius):
    """
    Compute a DTSPN tour using the NoonBean approach.  

    Parameters
    ----------
    goals: list Vector3
        list of the TSP goal coordinates (x, y, 0)
    sensing_radius: float
        neighborhood of TSP goals  
    turning_radius: float
        turning radius for the Dubins vehicle model  

    Returns
    -------
    Path, path_length (float)
        tour as a list of robot configurations in SE3 densely sampled
    """

    N = len(goals)
    position_resolution = heading_resolution = 8
    samples = create_samples(goals, sensing_radius, position_resolution, heading_resolution)

    # Number of samples per location
    M = position_resolution * heading_resolution
    distances = np.zeros((N*M, N*M))

    # START
    samples_per_point = M
    number_of_points = len(samples)
    distances2 = np.zeros((N*M, N*M)) #TODO: probabbly not necessary, can be optimized
    # HERE WE CREATE AN ALGORITHM TO BE ABLE TO SOLVE THE PROBLEM USING TSP
    # Basically, the main issue is that when we use TSP with neigborhoods, we need to use all edges before we move to another point
    # This is done by settign edges to zeros, and "switching" the edges in a way that we can iterate throughout the point
    # we cannot close the circle, so between the last and first point we use inf distance

    # COMPUTE THE DISTANCES
    # from each point check all other points
    for point1 in range(number_of_points):
        for point2 in range(number_of_points):
            #now from each sample check all otehr samples
            for sample1 in range(len(samples[point1])):
                for sample2 in range(len(samples[point2])):
                    # save the distance between samples
                    distances2[point1*samples_per_point+sample1][point2*samples_per_point+sample2] = dubins.shortest_path(pose_to_se2(samples[point1][sample1]), pose_to_se2(samples[point2][sample2]), turning_radius).path_length()
    print("NOW")                    

    # ADD CONSTANT M TO DISTANCES AND MOVE THEM
    const_M = 99999
    # from each point check all other points
    for point1 in range(number_of_points):
        print(point1,number_of_points)
        for point2 in range(number_of_points):
            #now from each sample check all otehr samples
            for sample1 in range(len(samples[point1])):
                for sample2 in range(len(samples[point2])): 
                    
                    # save the distance between samples
                    # create the circles of zeros
                    if point1 == point2:
                        if sample2 == (sample1 + 1) % samples_per_point:
                            distances[point1*samples_per_point + sample1][point2*samples_per_point + sample2] = 0
                        else:
                            distances[point1*samples_per_point + sample1][point2*samples_per_point + sample2] = const_M
                    else:
                        distances[point1*samples_per_point+sample1][point2*samples_per_point+sample2] = distances2[point1*samples_per_point+sample1][point2*samples_per_point+(sample2-1)%samples_per_point] + const_M


    sequence_TSP = solve_TSP(distances)
    print(sequence_TSP)
    sequence = []
    selected_samples = []

    last_point = 0

    # now we have a list of all samples for each point
    for i in range(len(sequence_TSP)):
        #check which point is it
        point = int(sequence_TSP[i]/samples_per_point) #position/number_of_samples
        #check if we are still looking at the same point (aka samples for the same point)
        #check if havent checked all the points
        if point != last_point and len(sequence) != number_of_points:
            sequence.append(last_point)
            selected_samples.append(sequence_TSP[i-1]%M)
            last_point = point
    
    # END


    # # TODO - goals are used in a random order
    # sequence = np.random.permutation(N)
    # # TODO - this code select the first sample in the set!
    # selected_samples = [0] * N

    configurations = []
    for idx in range(N):
        configurations.append (samples[sequence[idx]][selected_samples[idx]])

    return configurations_to_path(configurations, turning_radius)

####################################################################
# CUSTOM FUNCTIONS
####################################################################
def get_start_node(row,col,previos_node_matrix):
    while row!=0:
        col = previos_node_matrix[int(row)][int(col)]
        row-=1
    return int(col)
