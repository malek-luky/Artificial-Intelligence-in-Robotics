#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
import scipy.spatial

import collections
import heapq

from dijkstar import Graph, find_path

import Environment as env

#import communication messages
from messages import *

from matplotlib import pyplot as plt

# Custom messages
import random

class PRMPlanner():

    def __init__(self, environment, translate_speed, rotate_speed):
        """Initialize the sPRM solver
        Args:  environment: Environment - Map of the environment that provides collision checking
               translate_speed: float - speed of translation motion
               rotate_speed: float - angular speed of rotation motion
        """
        self.environment = environment
        self.translate_speed = translate_speed
        self.rotate_speed = rotate_speed

    def duration(self, p1, p2):
        """Compute duration of movement between two configurations
        Args:  p1: Pose - start pose
               p2: Pose - end pose
        Returns: float - duration in seconds 
        """
        t_translate = (p1.position - p2.position).norm() / self.translate_speed
        t_rotate = p1.orientation.dist(p2.orientation) / self.rotate_speed
        return max(t_translate, t_rotate)

    def create_edge(self, p1, p2, collision_step):
        """Sample an edge between start and goal
        Args:  p1: Pose - start pose
               p2: Pose - end pose
               collision_step: float - minimal time for testing collisions [s]
        Returns: Pose[] - list of sampled poses between the start and goal 
        """
        t = self.duration(p1, p2)
        steps_count = math.ceil(t / collision_step)
        #sample the path
        return slerp_se3(p1, p2, steps_count)   

    def check_collision(self, pth):
        """Check the collision status along a given path
        """
        for se3 in pth:
            if self.environment.check_robot_collision(se3):
                return True
        return False    

    def plan(self, start, goal, space, number_of_samples, neighborhood_radius, collision_step):
        """Plan the path from start to goal configuration
        Args:  start: Pose - start configuration of the robot in SE(3) coordinates
               goal: Pose - goal configuration of the robot in SE(3) coordinates
               space: String - configuration space type
               number_of_samples: int - number of samples to be generated
               neighborhood_radius: float - neighborhood radius to connect samples [s]
               collision_step: float - minimum step to check collisions [s]
        Returns:  Path - the path between the start and the goal Pose in SE(3) coordinates
                  NavGraph - the navigation graph for visualization of the built roadmap
        """
        #R2 - neresime rotaci, jen pozici
        #R3 - to stejne, jen tri dimenze
        #SE2 - pozice a rotace
        #SE3 - pozice a rotace, ale v 3D

        #TODO: t3a-sampl - implement the PRM planner

        # Returned path
        path = Path()
        # Returned graph
        navgraph = NavGraph()
        poses = [start, goal]
        print(start,goal)
        # TODO - generate samples based on 'space'
        print(goal.orientation)

        # START
        i = 0
        while i<number_of_samples:
            x = random.uniform(self.environment.limits_x[0],self.environment.limits_x[1])
            y = random.uniform(self.environment.limits_y[0],self.environment.limits_y[1])
            if space=='R2':
                vertex = Pose(Vector3(x,y,0), Quaternion(0,0,0,1))
            elif space=='R3':
                z = random.uniform(self.environment.limits_z[0],self.environment.limits_z[1])
                vertex = Pose(Vector3(x,y,z), Quaternion(0,0,0,1)) #vsechny poses jsou podprostory SE3, takze to je na phoodu, pro mensi prostory staci hodit veci "navic" na nulu    
            elif space=='SE(3)':
                z = random.uniform(self.environment.limits_z[0],self.environment.limits_z[1])
                yaw = random.uniform(0,2*math.pi)
                pitch = random.uniform(0,2*math.pi)
                roll = random.uniform(0,2*math.pi)
                q = Quaternion()
                q.from_Euler(yaw,pitch,roll)
                vertex = Pose(Vector3(x,y,z), q) #vsechny poses jsou podprostory SE3, takze to je na phoodu, pro mensi prostory staci hodit veci "navic" na nulu
            elif space=='SE(2)':
                z = random.uniform(self.environment.limits_z[0],self.environment.limits_z[1])
                yaw = random.uniform(0,2*math.pi)
                q = Quaternion()
                q.from_Euler(yaw,0,0)
                vertex = Pose(Vector3(x,y,z), q)
            if self.environment.check_robot_collision(vertex)==False:
                poses.append(vertex)
                i += 1
        # END
        navgraph.poses = poses # vytvori to ty body v grafu, jejich struktura co plotuji

        # Graph for finding the shortest path (Dijkstra)
        graph = Graph() # struktura pro grafovy algoritmus dijkstry

        # TODO - generate edges
        # START MY CODE
        for i in range(len(poses)):
            print(i,len(poses))
            for j in range(len(poses)):
                if i<=j:
                    break
                t = self.duration(poses[i], poses[j])
                if t < neighborhood_radius:
                    edge = self.create_edge(poses[i], poses[j], collision_step)
                    if self.check_collision(edge)==False:
                        graph.add_edge(i, j, t)
                        graph.add_edge(j, i, t)
                        navgraph.edges.append([i,j])
        # END MY CODE


        # HELPER FROM THE ORIGINAL CODE
        # edge = self.create_edge(start, goal, collision_step)
        # if self.check_collision(edge):
        #     print("Warning: This edge collides!")
        # t = self.duration(start, goal)
        # graph.add_edge(0, 1, t)
        # graph.add_edge(1, 0, t)
        # navgraph.edges.append([0, 1]) 
           
        try:
            solution = find_path(graph, 0, 1)
        except:
            print("Dijkstra did not find a solution.") 
            return(None, navgraph)
        #print(solution.nodes)
        #add start
        path.poses.append(start)

        # TODO - get solution
        # START OF MY CODE
        edge = self.create_edge(start, goal, collision_step)
        for i in range(len(solution.nodes)-1):
            edge = self.create_edge(poses[solution.nodes[i]], poses[solution.nodes[i+1]], collision_step)
            for e in edge:
                path.poses.append(e)

        # END PF MY CODE

        # HELPER
        # edge = self.create_edge(start, goal, collision_step)
        # for e in edge:
        #     path.poses.append(e)

        #add goal
        path.poses.append(goal)
        return(path, navgraph)

########################################################
# HELPER functions
########################################################
def slerp_se3(start, end, step_no):
    """Method to compute spherical linear interpolation between se3 poses
    Args:  start: Pose - starting pose
           end: Pose - end pose
           step_no: int - numer of interpolation steps
    Returns: steps: Pose[] - list of the interpolated se3 poses, always at least [start,end]
    """
    #extract the translation
    t1 = start.position
    t2 = end.position
    #extract the rotation
    q1 = start.orientation
    q2 = end.orientation
    #extract the minimum rotation angle
    theta = max(0.01, q1.dist(q2)) / 2 
    if (q1.x*q2.x + q1.y*q2.y + q1.z*q2.z + q1.w*q2.w) < 0:
        q2 = q2 * -1

    steps = []
    if step_no < 2:
        steps = [start,end]
    else:
        for a in np.linspace(0.,1.,step_no+1):
            ta = t1 + (t2-t1)*a
            qa = q1*np.sin( (1-a)*theta ) + q2*np.sin( a*theta )
            qa.normalize()
            pose = Pose(ta, qa)
            steps.append(pose)

    return steps   

