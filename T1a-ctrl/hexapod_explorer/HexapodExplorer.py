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

# CUSTOM IMPORT
import sys
import heapq
np.set_printoptions(threshold=sys.maxsize) # print full numpy array
# END OF CUSTOM IMPORT

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
    
    def convolution2d(self, image, size):
        y, x = image.shape
        new_image = copy.deepcopy(image)
        for i in range(y-size+1):
            for j in range(x-size+1):
                new_image[i+round(size/2)][j+round(size/2)] = np.max(image[i:i+size, j:j+size]) 
        return new_image

    def distance(self, start, end):
        '''
        Return the distance from the point A to the point B
        '''
        (x1, y1) = start
        (x2, y2) = end
        return (abs(x1 - x2) + abs(y1 - y2))

    def format_path(self, path, came_from, resolution, start, goal):
        '''
        Return the path as requested in assignment as a list
        of positions[(x1, y1), (x2, y2), ...] At the moment the
        path is stored as the dicionary with at the format {(pos1):(pos2),..}
        and if you want to find the final path, you have to go backward,
        in other words start at the goal and than ask, "how I get there" till
        we will not reach the starting point
        '''
        path.poses.insert(0,self.get_pose_from_coordinates(goal, resolution)) # Add goal!!
        coordinates = came_from.get(goal)
        while coordinates != start:
            pose = self.get_pose_from_coordinates(coordinates, resolution)
            coordinates = came_from.get(coordinates)
            path.poses.insert(0,pose)
        pose = self.get_pose_from_coordinates(coordinates, resolution)
        path.poses.insert(0,pose) # Add start!!
        return path

    def neighbors_finder(self, current_pos, grid_map):
        '''
        Return the list of the neighbors of the current position
        '''
        neighbors = []
        (x, y) = current_pos
        for i in range(-1, 2):
            for j in range(-1, 2):
                if (i, j) != (0, 0) and grid_map.data[y +j,x +i]==0: # magic why indexes are switched
                    neighbors.append((x + i, y + j))
        return neighbors

    def get_coordinates_from_pose(self, pose, resolution):
        '''
        Return the coordinates of the pose
        '''
        return (round(pose.position.x/resolution), round(pose.position.y/resolution))

    def get_pose_from_coordinates(self, coordinates, resolution):
        '''
        Return the pose from the coordinates
        '''
        pose = Pose()
        pose.position.x = round(coordinates[0] * resolution,1)
        pose.position.y = round(coordinates[1] * resolution,1) # ROUND TO 1 DECIMAL
        return pose


    def a_star(self, grid_map, path, start_pose, goal_pose):
        '''
        Generates the path. The method returns a list of coordinates for the
        path. It must start with the starting position and end at the goal
        position [(x1, y1), (x2, y2), … ]. If there is no path, it should
        return None.
        '''
        
        print("Starting A* algorithm")
        frontier = list()
        start = self.get_coordinates_from_pose(start_pose, grid_map.resolution)
        goal = self.get_coordinates_from_pose(goal_pose, grid_map.resolution)
        print(start,goal)
        heapq.heappush(frontier, (0, start))
        cost_so_far = dict()    # {(x1,y1):cost1, (x2,y2):cost2, ..}
        cost_so_far[start] = 0
        came_from = dict()      # {(0, 0):None, (1, 2):(0, 1), ...}
        came_from[start] = None

        while frontier:         # while not empty:
            current_pos = heapq.heappop(frontier)[1]

            if current_pos == goal:
                print("goal reached")
                break
            
            # neighbors = [[(x1, y1), [(x2, y2), cost], ... ]
            neighbors = self.neighbors_finder(current_pos, grid_map)
            # print(current_pos)
            # print(neighbors)
            
            for next_pos in neighbors:
                new_cost = cost_so_far.get(current_pos) + 1 # every step costs 1
                if next_pos not in cost_so_far or new_cost < cost_so_far.get(next_pos):
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.distance(next_pos, goal)
                    #print(1, new_cost, self.distance(next_pos, goal), priority)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current_pos
            # self.environment.render()  # show enviroment's GUI
        
        if current_pos == goal:
            return self.format_path(path, came_from, grid_map.resolution, start, goal)
        else:
            return None

    def bresenham_line(self, path_simple, path,grid_map):
        x2, y2 = self.get_coordinates_from_pose(path_simple, grid_map.resolution)
        
        x1, y1 = self.get_coordinates_from_pose(path, grid_map.resolution)
        path_ret = []
        max_diff = max(abs(x2 - x1), abs(y2 - y1))
        x_slope = (x2 - x1)/max_diff
        y_slope = (y2 - y1)/max_diff

        for i in range(max_diff):
            x = round(x2 - x_slope*i)
            y = round(y2 - y_slope*i)
            path_ret.append((x,y))
        return path_ret

    def collision(self, path, grid_map):
        """
        Return True if collision
        Return False if not
        """
        for (x,y) in path:
            if grid_map.data[y,x] != 0:
                return True
        return False

    def plot_graph(self, grid_map):
        fig, ax = plt.subplots()
        grid_map.plot(ax)
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        plt.axis('square')
        plt.show()
    ### END OF CUSTOM FUNCTIONS ###

    def fuse_laser_scan(self, grid_map, laser_scan, odometry):
        """ Method to fuse the laser scan data sampled by the robot with a given 
            odometry into the probabilistic occupancy grid map
        Args:
            grid_map_update: OccupancyGrid - gridmap to fuse te laser scan to
            laser_scan: LaserScan - laser scan perceived by the robot
            odometry: Odometry - perceived odometry of the robot
        Returns:
            grid_map_update: OccupancyGrid - gridmap updated with the laser scan data
        """
        grid_map_update = copy.deepcopy(grid_map)

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
                point_x_map = max(min(point_x_map, grid_map_update.width-1),0) #easy version
                point_y_map = max(min(point_y_map, grid_map_update.height-1),0) #easy version
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
            # data = grid_map_update.data.reshape(grid_map_update.height, grid_map_update.width)
            for (x,y) in occupied_points:
                data[y,x] = self.update_occupied(data[y,x])
            for (x,y) in free_points:
                data[y,x] = self.update_free(data[y,x])
            # grid_map_update.data = data.flatten() #easy version, watch out for dimensions in hard one
            


        # END OF OUR CODE WEEK 3

        return grid_map_update

        

    def grow_obstacles(self, grid_map, robot_size):
        """ Method to grow the obstacles to take into account the robot embodiment
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            robot_size: float - size of the robot
        Returns:
            grid_map_grow: OccupancyGrid - gridmap with considered robot body embodiment
        """
        grid_map_grow = copy.deepcopy(grid_map)

     

        # START OF WEEK 4 CODE PART 1
        # following this task steps on courseware: https://cw.fel.cvut.cz/wiki/courses/uir/hw/t1d-growth

        # Filter all obstacles
        grid_map_grow.data[grid_map_grow.data > 0.5] = 1
        # Filter all unknown arres
        grid_map_grow.data[grid_map_grow.data == 0.5] = 1
        # Filter all free areas
        grid_map_grow.data[grid_map_grow.data < 0.5] = 0
        #Filter cells close to obstacles
        kernel_size =round(robot_size/grid_map_grow.resolution)
        grid_map_grow.data = self.convolution2d(grid_map_grow.data, kernel_size)

        # END OF WEEK 4 CODE PART 1
        #self.plot_graph(grid_map_grow)
        return grid_map_grow


    def plan_path(self, grid_map, start, goal):
        """ Method to plan the path from start to the goal pose on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
        """

        path = Path()

        #add the start pose
        # path.poses.append(start)
        
        # START OF WEEK 4 CODE PART 2
        path = self.a_star(grid_map, path, start, goal)
        # END OF WEEK 4 CODE PART 2
        
        #add the goal pose
        #path.poses.append(goal)
        return path

    def simplify_path(self, grid_map, path):
        """ Method to simplify the found path on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            path: Path - path to be simplified
        Returns:
            path_simple: Path - simplified path
        """
        if grid_map == None or path == None:
            return None
        path_simple = Path()
        #add the start pose
        path_simple.poses.append(path.poses[0])
        
        # START OF WEEK 4 CODE PART 3

        # #ULTIMATE TEST
        # test1 = Pose()
        # test2 = Pose()
        # test1.position.x = 2.9
        # test1.position.y = 4.8
        # test2.position.x = 2.0
        # test2.position.y = 6.1
        # line = self.bresenham_line(test1, test2 ,grid_map)
        # print(line)
        # self.collision(line, grid_map) #there is no collision
               
        #iterate through the path and simplify the path
        i = 1
        while path_simple.poses[-1] != path.poses[-1]: #until the goal is not reached
            #find the connected segment
            previous_pose = path_simple.poses[-1]
            no_col = False
            for pose in path.poses[i::]:
                
                # if pose.position.x==previous_pose.position.x and pose.position.y==previous_pose.position.y:
                #     continue
                result_collision = self.collision(self.bresenham_line(previous_pose, pose ,grid_map), grid_map) #there is no collision
                #print(previous_pose.position.x, previous_pose.position.y, pose.position.x, pose.position.y, result_collision)
                if result_collision == False:
                    temp_pose = pose
                    i+=1
                    no_col = True
                    #the goal is reached
                    if pose == path.poses[-1]:  
                        path_simple.poses.append(pose)
                        break
            
                else:#if no_col==True: #there is collision
                    path_simple.poses.append(temp_pose)
                    break

        
        
        # END OF WEEK 4 CODE PART 3
        
        #add the goal pose
        path_simple.poses.append(path.poses[-1])

        return path_simple

    def find_free_edge_frontiers(self, grid_map):
        """Method to find the free-edge frontiers (edge clusters between the free and unknown areas)
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """

        #TODO:[t1e_expl] find free-adges and cluster the frontiers
        return None 


    def find_inf_frontiers(self, grid_map):
        """Method to find the frontiers based on information theory approach
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """

        #TODO:[t1e_expl] find the information rich points in the environment
        return None
 
    ###########################################################################
    #INCREMENTAL Planner
    ###########################################################################

    def plan_path_incremental(self, grid_map, start, goal):
        """ Method to plan the path from start to the goal pose on the grid
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            start: Pose - robot start pose
            goal: Pose - robot goal pose
        Returns:
            path: Path - path between the start and goal Pose on the map
        """

        if not hasattr(self, 'rhs'): #first run of the function
            self.rhs = np.full((grid_map.height, grid_map.width), np.inf)
            self.g = np.full((grid_map.height, grid_map.width), np.inf)

        #TODO:[t1x-dstar] plan the incremental path between the start and the goal Pose
 
        return self.plan_path(grid_map, start, goal), self.rhs.flatten(), self.g.flatten()
