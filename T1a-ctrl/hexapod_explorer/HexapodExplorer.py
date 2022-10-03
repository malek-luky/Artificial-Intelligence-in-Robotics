#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import numpy as np
import copy

#cpg network
import cpg.oscilator_network as osc

#import messages
from messages import *

import matplotlib.pyplot as plt

import scipy.ndimage as ndimg
from sklearn.cluster import KMeans

import skimage.measure

import collections
import heapq

class HexapodExplorer:

    def __init__(self):
        pass

    ### START OF CUSTOM FUNCTIONS ###
    def bresenham_line(self, start, goal):
        """Bresenham's line algorithm
        Args:
            start: (float64, float64) - start coordinate
            goal: (float64, float64) - goal coordinate
        Returns:
            interlying points between the start and goal coordinate
        """
        (x0, y0) = start
        (x1, y1) = goal
        line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                line.append((x,y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                line.append((x,y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        x = goal[0]
        y = goal[1]
        return line

    def world_to_map(self, point_x_global, point_y_global, grid_map):
        map_origin_x = grid_map.origin.position.x
        map_origin_y = grid_map.origin.position.y
        map_resolution = grid_map.resolution
        point_x_map = round((point_x_global-map_origin_x)/map_resolution)
        point_y_map = round((point_y_global-map_origin_y)/map_resolution)
        return (point_x_map, point_y_map)

    def update_free(self, P_mi):
        """method to calculate the Bayesian update of the free cell with the current occupancy probability value P_mi 
        Args:
            P_mi: float64 - current probability of the cell being occupied
        Returns:
            p_mi: float64 - updated probability of the cell being occupied
        """
        s_z_occ = 0
        s_z_free = 0.95 # to be sure that probability is never 0
        p_z_mi_occ = (1+s_z_occ-s_z_free)/2
        p_z_mi_free = 1-p_z_mi_occ
        p_mi = (p_z_mi_occ*P_mi)/((p_z_mi_occ*P_mi)+(p_z_mi_free*(1-P_mi)))
        return max(0.05, p_mi) #never let p_mi get to 0
    
    def update_occupied(self, P_mi):
        """method to calculate the Bayesian update of the occupied cell with the current occupancy probability value P_mi
        Args:
            P_mi: float64 - current probability of the cell being occupied
        Returns:
            p_mi: float64 - updated probability of the cell being occupied
        """
        s_z_occ = 0.95 # to be sure that probability is never 1
        s_z_free = 0 
        p_z_mi_occ = (1+s_z_occ-s_z_free)/2
        p_z_mi_free = 1-p_z_mi_occ
        p_mi = (p_z_mi_occ*P_mi)/((p_z_mi_occ*P_mi)+(p_z_mi_free*(1-P_mi)))   
        return min(p_mi, 0.95) #never let p_mi get to 1
        ### END OF CUSTOM FUNCTIONS ###

    def fuse_laser_scan(self, grid_map_update, laser_scan, odometry):
        """ Method to fuse the laser scan data sampled by the robot with a given 
            odometry into the probabilistic occupancy grid map
        Args:
            grid_map_update: OccupancyGrid - gridmap to fuse te laser scan to
            laser_scan: LaserScan - laser scan perceived by the robot
            odometry: Odometry - perceived odometry of the robot
        Returns:
            grid_map_update: OccupancyGrid - gridmap updated with the laser scan data
        """
        grid_map_update = copy.deepcopy(grid_map_update)

        # START OF OUR CODE WEEK 3
        # following this task steps on courseware: https://cw.fel.cvut.cz/wiki/courses/uir/hw/t1c-map

        # NOTES
        # filter the data outside of the range of the laser scanner
        # local frame: x = 0, y = 0, theta = alpha_min + i*angle_increment
        # global frame: x = odometry_x, y = odometry_y, theta_global = theta + odometry_angle
        # filtrovat uhly pro kazdou vysec zvlast, kdy je vyhazime na zacatku, ztratime udaje o tom, z jake vysece to bylo
        # dostaneme hodnoty, kdy kazda hodnota reprezentuje jednu vysec
        # alpha_min = minimalni uhel laseru
        # i = kolikata vysec z laser scaneru
        # delta = pocet stupnu jedne vysece scaneru

        if laser_scan is not None:
            '''
            INITIALIZE VALUES
            '''
            laserscan_points = laser_scan.distances
            alpha_min = laser_scan.angle_min
            odometry_R = odometry.pose.orientation.to_R()
            odometry_x = odometry.pose.position.x
            odometry_y = odometry.pose.position.y
            orientation = odometry.pose.orientation.to_Euler()[0]
            free_points = list()
            occupied_points = list()
            print("X: ",odometry_x, "Y: ",odometry_y, "°: ",orientation)

            for laserscan_sector,laserscan_sector_value in enumerate(laserscan_points):
 
                '''
                STEP 1
                Project the laser scan points to x,y plane with respect to the robot heading
                '''
                if laserscan_sector_value<laser_scan.range_min: laserscan_sector_value = laser_scan.range_min 
                if laserscan_sector_value>laser_scan.range_max: laserscan_sector_value = laser_scan.range_max
                theta_i = alpha_min + laserscan_sector*laser_scan.angle_increment
                point_x = laserscan_sector_value*np.cos(theta_i)
                point_y = laserscan_sector_value*np.sin(theta_i)
                #plt.scatter(point_x, point_y)

                '''
                STEP 2
                Compensate for the robot odometry
                '''
                point_normalized = np.transpose(np.array([point_x, point_y, 1]))
                point_x_global, point_y_global, _ = odometry_R @ point_normalized + np.array([odometry_x, odometry_y, 1])
                #plt.scatter(point_x_global, point_y_global)

                '''
                STEP 3
                Transfer the points from the world coordinates to the map coordinates
                '''
                point_x_map, point_y_map = self.world_to_map(point_x_global, point_y_global, grid_map_update)
                point_x_map = max(min(point_x_map, grid_map_update.width),0) #easy version
                point_y_map = max(min(point_y_map, grid_map_update.height),0) #easy version
                #plt.scatter(point_x_map, point_y_map)

                '''
                STEP 4
                Raytrace individual scanned points
                '''
                odometry_x_map, odometry_y_map = self.world_to_map(odometry_x, odometry_y, grid_map_update)
                pts = self.bresenham_line((odometry_x_map, odometry_y_map),(point_x_map, point_y_map))
                free_points.extend(pts) #nejsou free, je to ta line vsech bodu, nvm proc se to jmenuej tak divne
                occupied_points.append((point_x_map, point_y_map))
                #nevim jak to plotnout

            '''
            STEP 5
            Update the occupancy grid using the Bayesian update and the simplified laser scan sensor model
            '''
            data = grid_map_update.data.reshape(grid_map_update.height, grid_map_update.width)
            for (x,y) in occupied_points:
                data[y,x] = self.update_occupied(data[y,x])
            for (x,y) in free_points:
                data[y,x] = self.update_free(data[y,x])
            grid_map_update.data = data.flatten() #easy version, watch out for dimensions in hard one
            


        # END OF OUR CODE WEEK 3

        return grid_map_update


    def find_free_edge_frontiers(self, grid_map_update):
        """Method to find the free-edge frontiers (edge clusters between the free and unknown areas)
        Args:
            grid_map_update: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """

        #TODO:[t1e_expl] find free-adges and cluster the frontiers
        return None 


    def find_inf_frontiers(self, grid_map_update):
        """Method to find the frontiers based on information theory approach
        Args:
            grid_map_update: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """

        #TODO:[t1e_expl] find the information rich points in the environment
        return None


    def grow_obstacles(self, grid_map_update, robot_size):
        """ Method to grow the obstacles to take into account the robot embodiment
        Args:
            grid_map_update: OccupancyGrid - gridmap for obstacle growing
            robot_size: float - size of the robot
        Returns:
            grid_map_update_grow: OccupancyGrid - gridmap with considered robot body embodiment
        """

        grid_map_update_grow = copy.deepcopy(grid_map_update)

        #TODO:[t1d-plan] grow the obstacles for robot_size

        return grid_map_update_grow


    def plan_path(self, grid_map_update, start, goal):
        """ Method to plan the path from start to the goal pose on the grid
        Args:
            grid_map_update: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
        """

        path = Path()
        #add the start pose
        path.poses.append(start)
        
        #TODO:[t1d-plan] plan the path between the start and the goal Pose
        
        #add the goal pose
        path.poses.append(goal)

        return path

    def simplify_path(self, grid_map_update, path):
        """ Method to simplify the found path on the grid
        Args:
            grid_map_update: OccupancyGrid - gridmap for obstacle growing
            path: Path - path to be simplified
        Returns:
            path_simple: Path - simplified path
        """
        if grid_map_update == None or path == None:
            return None
 
        path_simplified = Path()
        #add the start pose
        path_simplified.poses.append(path.poses[0])
        
        #TODO:[t1d-plan] simplifie the planned path
        
        #add the goal pose
        path_simplified.poses.append(path.poses[-1])

        return path_simplified
 
    ###########################################################################
    #INCREMENTAL Planner
    ###########################################################################

    def plan_path_incremental(self, grid_map_update, start, goal):
        """ Method to plan the path from start to the goal pose on the grid
        Args:
            grid_map_update: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
        """

        if not hasattr(self, 'rhs'): #first run of the function
            self.rhs = np.full((grid_map_update.height, grid_map_update.width), np.inf)
            self.g = np.full((grid_map_update.height, grid_map_update.width), np.inf)

        #TODO:[t1x-dstar] plan the incremental path between the start and the goal Pose
 
        return self.plan_path(grid_map_update, start, goal), self.rhs.flatten(), self.g.flatten()
