# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import math

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
    # np.set_printoptions(precision=2)
    # print("ETSP distances")
    # print(etsp_distances)

    sequence = solve_TSP(etsp_distances)
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
    samples = [samples[x] for x in sequence]

    se2 = []
    for k in samples:
        se2.append(list(map(lambda x: pose_to_se2(x), k)))

    # TODO - this code select the first sample in the set of each N!

    best_path_length = np.inf

    for starting in range(len(samples[0])):  # starting

        L = len(samples)
        F = np.zeros([L, len(samples[0])])

        #prida do tabulky 99999 vsem bodum v prvnim sloupic kde se nezacina
        for y in range(len(samples[0])):
            if y != starting:
                F[0, y] = 9999999999

        # zacne bublat do hloubky
        for i in range(1, len(samples)):
            for y in range(len(samples[0])):

                mn = np.inf
                for ys in range(len(samples[0])):
                    start = se2[i-1][ys]
                    end = se2[i][y]
                    dubins_path = dubins.shortest_path(start, end, turning_radius)
                    v = F[i-1, ys] + dubins_path.path_length()
                    if i == L-1:
                        start = se2[i][y]
                        end = se2[0][starting]
                        back = dubins.shortest_path(start, end, turning_radius)
                        v += back.path_length()
                    if v < mn:
                        mn = v

                F[i, y] = mn

        yL = None
        mn = np.inf

        for y in range(F.shape[1]):
            if F[L-1, y] < mn:
                mn = F[L-1, y]
                yL = y

        if mn >= best_path_length:
            continue
        best_path_length = mn

        selected_samples = [0] * N
        selected_samples[L-1] = yL

        for s in range(L-1):
            i = L-1-s
            y = selected_samples[i]

            mn = np.inf
            mnYs = None
            for ys in range(F.shape[1]):
                start = se2[i-1][ys]
                end = se2[i][y]
                dubins_path = dubins.shortest_path(start, end, turning_radius)
                v = F[i-1, ys] + dubins_path.path_length()
                if v < mn:
                    mn = v
                    mnYs = ys

            selected_samples[i-1] = mnYs
            print(selected_samples)
            print(best_path_length)

    configurations = []
    for idx in range(N):
        configurations.append (samples[idx][selected_samples[idx]])

    print(configurations_to_path(configurations, turning_radius)[1])
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
    distances2 = np.zeros((N*M, N*M))

    '''
    TODO - homework (Noon-Bean approach)
    '''

    max_dist = -np.inf

    for point1 in range(len(samples)):
        for point2 in range(len(samples)):
            for s1 in range(len(samples[point1])):
                for s2 in range(len(samples[point2])):
                    start = pose_to_se2(samples[point1][s1])
                    end = pose_to_se2(samples[point2][s2])
                    distance = dubins.shortest_path(start, end, turning_radius).path_length()
                    distances2[point1*M + s1][point2*M + s2] = distance
                    max_dist = max(max_dist, distance)

    distances = np.zeros(distances2.shape)

    for point1 in range(len(samples)):
        for point2 in range(len(samples)):
            for s1 in range(len(samples[point1])):
                for s2 in range(len(samples[point2])):
                    if point1 == point2:
                        if s2 == (s1 + 1) % M:
                            distances[point1*M + s1][point2*M + s2] = 0
                        else:
                            distances[point1*M + s1][point2*M + s2] = max_dist
                        continue
                    distances[point1*M + s1][point2*M + s2] = distances2[point1*M + s1][point2*M + (s2-1) % M] + max_dist

    # TODO - goals are used in a random order
    sequence_TSP = solve_TSP(distances)
    print(sequence_TSP)
    sequence = []
    selected_samples = []

    last_point = 0

    for i in range(len(sequence_TSP)):
        point = int(sequence_TSP[i]/M)
        if point != last_point and len(sequence) != N:
            sequence.append(last_point)
            selected_samples.append(sequence_TSP[i-1]%M)
            last_point = point

    configurations = []
    for idx in range(N):
        configurations.append (samples[sequence[idx]][selected_samples[idx]])

    return configurations_to_path(configurations, turning_radius)
