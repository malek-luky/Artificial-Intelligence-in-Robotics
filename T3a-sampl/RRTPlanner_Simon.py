#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import sys
import math
import numpy as np
import scipy.spatial

import collections
import heapq

import Environment as env

#import communication messages
from messages import *

from matplotlib import pyplot as plt

class RRRTNode():
    def __init__(self, parent, pose, cost, id):
        self.parent = parent
        self.pose = pose
        self.cost = cost
        self.id = id

class RRTPlanner():

    def __init__(self, environment, translate_speed, rotate_speed):
        """Initialize the sPRM solver
        Args:  environment: Environment - Map of the environment that provides collision checking
               translate_speed: float - speed of translation motion
               rotate_speed: float - angular speed of rotation motion
        """
        self.environment = environment
        self.translate_speed = translate_speed
        self.rotate_speed = rotate_speed
        self.nodes = []

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
        if steps_count <= 1:
            return [p1, p2]
        else:
            parameters = np.linspace(0.,1.,steps_count+1)
            return slerp_se3(p1, p2, parameters)  

    def check_collision(self, pth):
        """Check the collision status along a given path
        """
        for se3 in pth:
            if self.environment.check_robot_collision(se3):
                return True
        return False    

    def Steer(self, x_nearest, x_rand, steer_step):
        """Steer function of RRT algorithm 
        Args:  x_nearest: Pose - pose from which the tree expands
               x_rand: Pose - random pose towards which the tree expands
               steer_step: float - maximum distance from x_nearest [s]
        Returns: Pose - new pose to be inserted into the tree
        """
        t = self.duration(x_nearest, x_rand)
        if t < steer_step:
            return x_rand
        else:
            parameter = steer_step / t
            return slerp_se3(x_nearest, x_rand, [parameter])[0]

    def expand_tree(self, start, space, number_of_samples, neighborhood_radius, collision_step,
            isrrtstar, steer_step):
        """Expand the RRT(*) tree for finding the shortest path
        Args:  start: Pose - start configuration of the robot in SE(3) coordinates
               space: String - configuration space type
               number_of_samples: int - number of samples to be generated
               neighborhood_radius: float - neighborhood radius to connect samples [s]
               collision_step: float - minimum step to check collisions [s]
               isrrtstar: bool - determine RRT* variant
               steer_step: float - step utilized of steering function
        Returns:  NavGraph - the navigation graph for visualization of the built roadmap
        """

        root = RRRTNode(None, start, 0.0, 0)
        self.nodes = [root]

        # TODO: t3a-sampl - implement the RRT/RRT* planner
        # TODO 1) Create your own tree structure
        # TODO 2) Implement the RRT/RRT* algorithm (Steer function is provided)
        # TODO 3) Return the generated tree structure as a NavGrav

        samples = []

        while len(samples) < number_of_samples:
            pose = Pose()
            pose.position.x = random.uniform(self.environment.limits_x[0], self.environment.limits_x[1])
            pose.position.y = random.uniform(self.environment.limits_y[0], self.environment.limits_y[1])
            if space == "R3" or space == "SE(3)":
                pose.position.z = random.uniform(self.environment.limits_z[0], self.environment.limits_z[1])
            s_yaw = random.uniform(0, 2*np.pi)
            s_pitch = random.uniform(0, 2*np.pi)
            s_roll = random.uniform(0, 2*np.pi)
            if space == "SE(2)":
                pose.orientation.from_Euler(s_yaw, 0, 0)
            elif space == "SE(3)":
                pose.orientation.from_Euler(s_yaw, s_pitch, s_roll)
            if not self.environment.check_robot_collision(pose):
                samples.append(pose)

        for x_rand in samples:
            x_nearest = self.nearest(x_rand)
            x_new = self.Steer(x_nearest.pose, x_rand, steer_step)
            if self.obstacleFree(x_nearest.pose, x_new, collision_step):
                if not isrrtstar:
                    self.nodes.append(RRRTNode(x_nearest, x_new, 0.0, len(self.nodes)))
                else:
                    X_near = self.near(x_new, neighborhood_radius)
                    x_min = x_nearest
                    c_min = x_nearest.cost + self.duration(x_nearest.pose, x_new)
                    for x_near in X_near:
                        c = x_near.cost + self.duration(x_near.pose, x_new)
                        if self.obstacleFree(x_near.pose, x_new, collision_step) and c < c_min:
                            x_min = x_near
                            c_min = c
                    newNode = RRRTNode(x_min, x_new, x_min.cost + c_min, len(self.nodes))
                    self.nodes.append(newNode)
                    for x_near in X_near:
                        if self.obstacleFree(x_new, x_near.pose, collision_step) and newNode.cost + self.duration(x_new, x_near.pose) < x_near.cost:
                            x_parent = x_near.parent
                            x_near.parent = newNode

        navgraph = NavGraph()
        for node in self.nodes:
            navgraph.poses.append(node.pose)
            if node.parent is not None:
                navgraph.edges.append((self.nodes.index(node.parent), self.nodes.index(node)))

        return navgraph

    def query(self, goal, neighborhood_radius, collision_step, isrrtstar):
        """Retrieve path for the goal configuration
        Args:  goal: Pose - goal configuration of the robot in SE(3) coordinates
               neighborhood_radius: float - neighborhood radius to connect samples [s]
               collision_step: float - minimum step to check collisions [s]
               isrrtstar: bool - determine RRT* variant
        Returns:  Path - the path between the start and the goal Pose in SE(3) coordinates
        """
        path = Path()

        # TODO: 4) Retrieve path for the goal configuration
        x_prev = goal
        x_next = self.nearest(goal)
        while x_next is not None:
            edge = self.create_edge(x_prev, x_next.pose, collision_step)
            for p in edge:
                path.poses.append(p)
            x_prev = x_next.pose
            x_next = x_next.parent

        path.poses.reverse()

        return path

    def near(self, x_new, neighborhood_radius):
        near = []
        for neig in self.nodes:
            dist = self.duration(x_new, neig.pose)
            if dist <= neighborhood_radius:
                near.append(neig)
        return near

    def nearest(self, x_rand):
        min_dist = np.inf
        x_nearest = None
        for neig in self.nodes:
            dist = self.duration(x_rand, neig.pose)
            if dist < min_dist:
                min_dist = dist
                x_nearest = neig
        return x_nearest

    def obstacleFree(self, x_nearest, x_new, collision_step):
        return not self.check_collision(self.create_edge(x_nearest, x_new, collision_step))

########################################################
# HELPER functions
########################################################
def slerp_se3(start, end, parameters):
    """Method to compute spherical linear interpolation between se3 poses
    Args:  start: Pose - starting pose
           end: Pose - end pose
           parameters: float[] - array of parameter in (0,1), 0-start, 1-end
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
    for a in parameters:
        ta = t1 + (t2-t1)*a
        qa = q1*np.sin( (1-a)*theta ) + q2*np.sin( a*theta )
        qa.normalize()
        pose = Pose(ta, qa)
        steps.append(pose)

    return steps

