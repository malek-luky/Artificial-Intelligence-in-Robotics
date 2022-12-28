#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# MY DEFINES
import random

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
        """
        Returns True if there is a collision
        Check the collision status along a given path
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

        # START OF MY CODE
        # TODO: t3a-sampl - implement the RRT/RRT* planner
        # I am implementing RRT*
        
        
        
        # TODO 1) Create your own tree structure
        # Same as the previous homework
        samples = []
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
                samples.append(vertex)
                i += 1

        # TODO 2) Implement the RRT/RRT* algorithm (Steer function is provided)
        # Following the pseudocode online
        for i in range(len(samples)):
            x_rand = samples[i]
            x_nearest = self.nearest(x_rand)
            x_new = self.Steer(x_nearest.pose, x_rand, steer_step)

            # RRT
            if not isrrtstar:
                if self.CollisionFree(x_nearest.pose, x_new, collision_step): #CollisonFree = ObstacleFree
                    self.nodes.append(RRRTNode(x_nearest, x_new, 0.0, len(self.nodes)))

            # RRT*
            else:
                if self.CollisionFree(x_nearest.pose, x_new, collision_step): #CollisonFree = ObstacleFree
                    X_near = self.inside_radius(x_new, neighborhood_radius)
                    x_min = x_nearest
                    c_min = x_nearest.cost + self.duration(x_nearest.pose, x_new)
                    for x_near in X_near:
                        if self.CollisionFree(x_near.pose, x_new, collision_step) and \
                            x_near.cost + self.duration(x_near.pose, x_new) < c_min:
                            x_min = x_near
                            c_min = x_near.cost + self.duration(x_near.pose, x_new)
                    E = RRRTNode(x_min, x_new, x_min.cost + c_min, len(self.nodes))
                    self.nodes.append(E)
                    for x_near in X_near:
                        if self.CollisionFree(x_new, x_near.pose, collision_step) and \
                            x_near.cost + self.duration(x_new, x_near.pose) < x_near.cost:
                            x_parent = x_near.parent
                            x_near.parent = x_near #kinda confusing from pseudocode, but the works :)
                            # credits to Simon Zvara who helped me with this part, cause I was getting bad results

        # TODO 3) Return the generated tree structure as a NavGrav
        # Similar to previous homework, edges can be done waaay simpler :)
        navgraph = NavGraph()
        poses = []
        edges = []
        for node in self.nodes:
            poses.append(node.pose)
            if node.parent is not None:
                edges.append((self.nodes.index(node.parent), self.nodes.index(node)))
        navgraph.poses = poses
        navgraph.edges = edges
        return navgraph
        # END OF MY CODE

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

        # edge = self.create_edge(self.nodes[0].pose, goal, collision_step) 
        # for p in edge:
        #     path.poses.append(p)

        # START OF MY CODE
        next_ = self.nearest(goal)
        previous = goal
        while next_ != None:
            edge = self.create_edge(previous, next_.pose, collision_step)
            for p in edge:
                path.poses.append(p)
            previous = next_.pose
            next_ = next_.parent #NECCESARY TO HAVE THE CORRECT ORDER
        path.poses.reverse() # reverse
        # END OF MY CODE

        return path

########################################################
# MY OWN FUNCTIONS
########################################################
    def CollisionFree(self, x_nearest, x_new, collision_step):
        return not self.check_collision(self.create_edge(x_nearest, x_new, collision_step))

    def nearest(self, goal):
        """
        Finds the nearest neigbour to the goal
        """
        min_dist = 99999999
        for vertex in self.nodes:
            dist = self.duration(vertex.pose, goal)
            if dist < min_dist:
                min_dist = dist
                nearest = vertex
        return nearest

    def inside_radius(self, vertex, radius):
        neighbours = list()
        for neigbour in self.nodes:
            if self.duration(neigbour.pose,vertex) < radius: #TODO
                neighbours.append(neigbour)
        return neighbours

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

