#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import time
import numpy as np
import copy

# cpg network
import cpg.oscilator_network as osc

#import messages
from messages import *

import matplotlib.pyplot as plt

import scipy.ndimage as ndimg
from sklearn.cluster import KMeans

import collections
import heapq

# CUSTOM IMPORT
import sys
import heapq
import skimage.measure as skm
#from queue import PriorityQueue
np.set_printoptions(threshold=sys.maxsize)  # print full numpy array

MAGIC_NUMBER = 50 #TODO

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def pop(self):
        return heapq.heappop(self.elements)[1]

    def top(self):
        u = self.elements[0]
        return u[1]

    def topKey(self):
        u = self.elements[0]
        return u[0]

    def contains(self, element):
        ret = False
        for item in self.elements:
            if element == item[1]:
                ret = True
                break
        return ret

    def print_elements(self):
        return self.elements

    def remove(self, element):
        i = 0
        for item in self.elements:
            if element == item[1]:
                self.elements[i] = self.elements[-1]
                self.elements.pop()
                heapq.heapify(self.elements)
                break
            i += 1

    def arg(self,value):
        for item in self.elements:
            for subitem in item[0]:
                if value == subitem:
                    return item[1]

# END OF CUSTOM IMPORT


class HexapodExplorer:

    def __init__(self):
        pass

    ### START OF CUSTOM FUNCTIONS ###

    #### START OF MY CODE SEMESTER PROJECT ####
    def print_path(self, path):
        if path is not None:
            if len(path.poses)==0:
                print("Path is empty")
            for i, pose in enumerate(path.poses):
                print("Point",i,":",pose.position.x,pose.position.y)
        else:
            print("Path is None")

    #### END OF MY CODE SEMESTER PROJECT ####

    def world_to_map(self, point_x_global, point_y_global, grid_map):
        map_origin = np.array([grid_map.origin.position.x, grid_map.origin.position.y])
        point = np.array([point_x_global, point_y_global])
        res = tuple((point - map_origin) / grid_map.resolution)
        return tuple((round(x) for x in res))

    def update_free(self, P_mi):
        """method to calculate the Bayesian update of the free cell with the current occupancy probability value P_mi 
        Args:
            P_mi: float64 - current probability of the cell being occupied
        Returns:
            p_mi: float64 - updated probability of the cell being occupied
        """
        s_z_occ = 0
        s_z_free = 0.95  # to be sure that probability is never 0
        p_z_mi_occ = (1+s_z_occ-s_z_free)/2
        p_z_mi_free = 1-p_z_mi_occ
        p_mi = (p_z_mi_occ*P_mi)/((p_z_mi_occ*P_mi)+(p_z_mi_free*(1-P_mi)))
        return max(0.05, p_mi)  # never let p_mi get to 0

    def update_occupied(self, P_mi):
        """method to calculate the Bayesian update of the occupied cell with the current occupancy probability value P_mi
        Args:
            P_mi: float64 - current probability of the cell being occupied
        Returns:
            p_mi: float64 - updated probability of the cell being occupied
        """
        s_z_occ = 0.95  # to be sure that probability is never 1
        s_z_free = 0
        p_z_mi_occ = (1+s_z_occ-s_z_free)/2
        p_z_mi_free = 1-p_z_mi_occ
        p_mi = (p_z_mi_occ*P_mi)/((p_z_mi_occ*P_mi)+(p_z_mi_free*(1-P_mi)))
        return min(p_mi, 0.95)  # never let p_mi get to 1

    def convolution2d(self, grid_map, kernel_size):
        """
        Return the convolution of the image with the kernel
        NOT WORKING, MIGHT BE USEFUL LATER
        """
        y_size, x_size = grid_map.shape
        new_grid_map = copy.deepcopy(grid_map)
        offset = int(kernel_size/2)  # kernel size are only odd numbers
        for x in range(offset, y_size-offset+1):
            for y in range(offset, x_size-offset+1):
                new_grid_map[x][y] = np.max(
                    grid_map[x-offset:x+offset, y-offset:y+offset])
        return new_grid_map

    def distance(self, start, end):
        '''
        Return the distance from the point A to the point B
        '''
        (x1, y1) = start
        (x2, y2) = end
        return math.sqrt(abs(x1 - x2)**2 + abs(y1 - y2)**2)

    def format_path(self, path, came_from, resolution, start, goal):
        '''
        Return the path as requested in assignment as a list
        of positions[(x1, y1), (x2, y2), ...] At the moment the
        path is stored as the dicionary with at the format {(pos1):(pos2),..}
        and if you want to find the final path, you have to go backward,
        in other words start at the goal and than ask, "how I get there" till
        we will not reach the starting point
        '''
        path.poses.insert(0, self.get_pose_from_coordinates(
            goal, resolution))  # Add goal!!
        coordinates = came_from.get(goal)
        while coordinates != start:
            pose = self.get_pose_from_coordinates(coordinates, resolution)
            coordinates = came_from.get(coordinates)
            path.poses.insert(0, pose)
        pose = self.get_pose_from_coordinates(coordinates, resolution)
        path.poses.insert(0, pose)  # Add start!!
        return path

    def neighbors_finder(self, current_pos, grid_map):
        '''
        Return the list of the neighbors of the current position
        '''
        neighbors = []
        (x, y) = current_pos
        for i in range(-1, 2):
            for j in range(-1, 2):
                # magic why indexes are switched
                if (i, j) != (0, 0) and grid_map.data[y + j, x + i] == 0:
                    neighbors.append((x + i, y + j))
        return neighbors

    def get_coordinates_from_pose(self, pose, resolution):
        '''
        Return the coordinates of the pose
        '''
        return (round(pose.position.x/resolution+MAGIC_NUMBER), round(pose.position.y/resolution+MAGIC_NUMBER))

    def get_pose_from_coordinates(self, coordinates, resolution):
        '''
        Return the pose from the coordinates
        '''
        pose = Pose()
        pose.position.x = np.round(coordinates[0] * resolution-MAGIC_NUMBER/10,decimals=3)
        pose.position.y = np.round(coordinates[1] * resolution-MAGIC_NUMBER/10,decimals=3)
        return pose

    def a_star(self, grid_map, path, start_pose, goal_pose):
        '''
        Generates the path. The method returns a list of coordinates for the
        path. It must start with the starting position and end at the goal
        position [(x1, y1), (x2, y2), … ]. If there is no path, it should
        return None.
        '''
        frontier = list()
        start = self.world_to_map(start_pose.position.x,start_pose.position.y, grid_map)
        goal = self.world_to_map(goal_pose.position.x,goal_pose.position.y, grid_map)
        heapq.heappush(frontier, (0, start))
        cost_so_far = dict()    # {(x1,y1):cost1, (x2,y2):cost2, ..}
        cost_so_far[start] = 0
        came_from = dict()      # {(0, 0):None, (1, 2):(0, 1), ...}
        came_from[start] = None

        while frontier:         # while not empty:
            current_pos = heapq.heappop(frontier)[1]

            if current_pos == goal:
                break

            # neighbors = [[(x1, y1), [(x2, y2), cost], ... ]
            neighbors = self.neighbors_finder(current_pos, grid_map)

            for next_pos in neighbors:
                new_cost = cost_so_far.get(
                    current_pos) + self.distance(next_pos, current_pos)
                if next_pos not in cost_so_far or new_cost < cost_so_far.get(next_pos):
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.distance(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current_pos

        if current_pos == goal:
            ret = self.format_path(path, came_from, grid_map.resolution, start, goal)
            return ret
        else:
            return None

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

    def collision(self, path, grid_map):
        """
        Return True if collision on the bresenham line, return False if not
        """
        for (x, y) in path:
            if grid_map.data[y, x] != 0:
                return True
        return False

    def plot_graph(self, grid_map):
        """
        Plot the graphs in case error occurs and we want to see the map
        """
        #fig, ax = plt.subplots()
        data = grid_map.data.reshape(grid_map.height, grid_map.width)

        # creating a plot
        fig, ax = plt.subplots()
        plt.imshow(data, cmap='viridis')
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.colorbar()

        # plotting a plot
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        plt.title("pixel_plot")

        # show plot
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

        if laser_scan is not None and odometry is not None:
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
            #print("X: ", odometry_x, "Y: ", odometry_y, "°: ", orientation)

            for laserscan_sector, laserscan_sector_value in enumerate(laserscan_points):

                '''
                STEP 1
                Project the laser scan points to x,y plane with respect to the robot heading
                '''
                if laserscan_sector_value < laser_scan.range_min:
                    laserscan_sector_value = laser_scan.range_min
                if laserscan_sector_value > laser_scan.range_max:
                    laserscan_sector_value = laser_scan.range_max
                theta_i = alpha_min + laserscan_sector*laser_scan.angle_increment
                point_x = laserscan_sector_value*np.cos(theta_i)
                point_y = laserscan_sector_value*np.sin(theta_i)
                #plt.scatter(point_x, point_y)

                '''
                STEP 2
                Compensate for the robot odometry
                '''
                point_normalized = np.transpose(
                    np.array([point_x, point_y, 1]))
                point_x_global, point_y_global, _ = odometry_R @ point_normalized + \
                    np.array([odometry_x, odometry_y, 1])
                #plt.scatter(point_x_global, point_y_global)

                '''
                STEP 3
                Transfer the points from the world coordinates to the map coordinates
                '''
                point_x_map, point_y_map = self.world_to_map(
                    point_x_global, point_y_global, grid_map_update)
                point_x_map = max(
                    min(point_x_map, grid_map_update.width-1), 0)  # easy version
                point_y_map = max(
                    min(point_y_map, grid_map_update.height-1), 0)  # easy version
                #plt.scatter(point_x_map, point_y_map)

                '''
                STEP 4
                Raytrace individual scanned points
                '''
                odometry_x_map, odometry_y_map = self.world_to_map(
                    odometry_x, odometry_y, grid_map_update)
                pts = self.bresenham_line(
                    (odometry_x_map, odometry_y_map), (point_x_map, point_y_map))
                # nejsou free, je to ta line vsech bodu, nvm proc se to jmenuej tak divne
                free_points.extend(pts)
                occupied_points.append((point_x_map, point_y_map))
                # nevim jak to plotnout

            '''
            STEP 5
            Update the occupancy grid using the Bayesian update and the simplified laser scan sensor model
            '''
            data = grid_map_update.data.reshape(grid_map_update.height, grid_map_update.width)
            for (x, y) in occupied_points:
                data[y, x] = self.update_occupied(data[y, x])
            for (x, y) in free_points:
                data[y, x] = self.update_free(data[y, x])
            grid_map_update.data = data.flatten() #easy version, watch out for dimensions in hard one

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

        """
        UPDATE
        * use obstacle growing only on obstacles, not uknown areas
        -> filter all unknown areas moved to the end of the function
        """
        grid_map_grow = copy.deepcopy(grid_map)

        # START OF WEEK 4 CODE PART 1
        # following this task steps on courseware: https://cw.fel.cvut.cz/wiki/courses/uir/hw/t1d-growth

        # Filter all obstacles
        grid_map_grow.data[grid_map.data > 0.5] = 1  # obstacles
        # Filter all free areas
        grid_map_grow.data[grid_map.data <= 0.5] = 0  # free area
        # Filter cells close to obstacle
        kernel_size = round(
            robot_size/grid_map_grow.resolution)  # must be even
        r = round(kernel_size)
        kernel = np.fromfunction(lambda x, y: (
            (x-r)**2 + (y-r)**2 < r**2)*1, (2*r+1, 2*r+1), dtype=int).astype(np.uint8)
        grid_map_grow.data = ndimg.convolve(grid_map_grow.data, kernel)
        grid_map_grow.data[grid_map_grow.data > 1] = 1

        # Filter all unknown arres
        grid_map_grow.data[grid_map.data == 0.5] = 1  # unknown area
        # self.plot_graph(grid_map_grow)
        # END OF WEEK 4 CODE PART 1
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

        # add the start pose
        # path.poses.append(start)

        # START OF WEEK 4 CODE PART 2
        path = self.a_star(grid_map, path, start, goal)
        # adding start and end point is implemented inside the function
        # END OF WEEK 4 CODE PART 2

        # add the goal pose
        # path.poses.append(goal)
        return path

    def simplify_path(self, grid_map, path):
        """ Method to simplify the found path on the grid
            Founds the connected segments and remove unnecessary points
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            path: Path - path to be simplified
        Returns:
            path_simple: Path - simplified path
        """
        if grid_map == None or path == None or len(path.poses)==0:
            return None
        path_simple = Path()
        path_simple.poses.append(path.poses[0]) # add the start pose

        # START OF WEEK 4 CODE PART 3
        i = 1
        while path_simple.poses[-1] != path.poses[-1]: #while goal not reached
            last_pose = path_simple.poses[-1]
            for pose in path.poses[i::]:   
                end = path_simple.poses[-1]            
                bres_line = self.bresenham_line(self.world_to_map(end.position.x,end.position.y,grid_map),
                                                self.world_to_map(pose.position.x, pose.position.y,grid_map))                
                result_collision = self.collision(bres_line,grid_map)
                if result_collision == False:
                    last_pose = pose
                    i += 1
                    if pose == path.poses[-1]:  # goal is reached
                        path_simple.poses.append(pose)
                        break
                else:
                    path_simple.poses.append(last_pose)
                    break
        # END OF WEEK 4 CODE PART 3

        # add the goal pose - already addded!!
        # path_simple.poses.append(path.poses[-1])
        return path_simple

    def find_free_edge_frontiers(self, grid_map):
        """Method to find the free-edge frontiers (edge clusters between the free and unknown areas)
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """

        # START OF MY CODE WEEK 5
        data = copy.deepcopy(grid_map.data)#.reshape(grid_map.height, grid_map.width))
        data[data == 0.5] = 10

        mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        data_c = ndimg.convolve(data, mask, mode='constant', cval=0.0)

        # Format result
        frontiers = []
        for (y, val1) in enumerate(data_c):
            for (x, val2) in enumerate(val1):
                if grid_map.data.reshape(grid_map.height, grid_map.width)[(y, x)] < 0.5 and val2 < 50 and val2 >= 10:
                    data[(y, x)] = 1
                else:
                    data[(y, x)] = 0
        labeled_image, num_labels = skm.label(
            data, connectivity=2, return_num=True)

        # Final Labeling
        cluster = {}
        for label in range(1, num_labels+1):
            cluster[label] = []

        for x in range(0, labeled_image.shape[1]):
            for y in range(0, labeled_image.shape[0]):

                label = labeled_image[y, x]
                if label != 0:
                    cluster[label].append((x, y))

        pose_list = []

        for label, items in cluster.items():
            centroid = (0, 0)
            for item in items:
                centroid = (centroid[0]+item[0], centroid[1]+item[1])
            centroid = (centroid[0]/len(items), centroid[1]/len(items))
            #print(centroid)
            pose = self.get_pose_from_coordinates(
                [centroid[0], centroid[1]], grid_map.resolution)
            pose_list.append(pose)

        # print(pose_list)
        # self.plot_graph(grid_map)
        # self.plot_graph(grid_convolved)
        # self.plot_graph(grid_frontiers)

        return pose_list
        # END OF MY CODE WEEK 5

    def find_inf_frontiers(self, grid_map):
        """Method to find the frontiers based on information theory approach
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """
        # TODO:[t1e_expl] find the information rich points in the environment
        return None

    ###########################################################################
    # INCREMENTAL Planner
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
        start = tuple(self.world_to_map(start.position.x, start.position.y, grid_map))
        goal = tuple(self.world_to_map(goal.position.x, goal.position.y, grid_map))

        if not hasattr(self, 'rhs'):  # first run of the function
            self.rhs = np.full((grid_map.height, grid_map.width), np.inf)
            self.g = np.full((grid_map.height, grid_map.width), np.inf)
            self.gridmap_data = copy.deepcopy(grid_map.data.reshape(
                grid_map.height, grid_map.width).transpose())

            # Main init
            self.initialize(goal) # only in the first cycle
            self.compute_shortest_path(start, goal)

        return self.plan_path_Dstar(grid_map, start, goal), self.rhs.flatten(), self.g.flatten()

# START OF MY CODE WEEK 5 D-Star

#### HELP FUNCTIONS #####
    def c(self, From, To):
        """
        returns the value of an edge from From to To
        """
        if self.gridmap_data[To] == 1:
            return np.inf
        else:
            return np.sqrt((From[0] - To[0]) ** 2 + (From[1] - To[1]) ** 2)

    def min_succ(self, u):
        """
        returns the successor with the lowest rhs value
        u(int,int) = current coordinate
        """
        ret = np.inf
        for s_ in self.pred(u):
            ret = min(ret, self.c(u, s_) + self.g[s_])
        return ret

    def pred(self, coord):
        """
        returns list of all 8 neighbors around the coord
        """
        ret = []
        coord = np.array(coord)
        neighbor_list = [coord+[1,0], coord+[0,1], coord+[-1,0], coord+[0,-1], coord+[1,1], coord+[-1,-1], coord+[1,-1], coord+[-1,+1]]
        for neighbor in neighbor_list:
            neighbor = tuple(neighbor)
            if 0 <= neighbor[0] < len(self.gridmap_data) and 0 <= neighbor[1] < len(self.gridmap_data[1]) and self.gridmap_data[neighbor] == 0:
                ret.append(neighbor)
        return ret

    def scan_graph(self, grid_map, data):
        ret_list = list()
        for x in range(0, grid_map.width):
            for y in range(0, grid_map.height):
                if data[x, y] != self.gridmap_data[x, y]:
                    ret_list.append((x, y))
        return ret_list

    def format_path_Dstar(self, start, goal):
        path = Path()
        pose = Pose()
        pose.position.x = start[0]+0.5
        pose.position.y = start[1]+0.5
        path.poses.append(pose)

        u = start
        while u != goal:
            for s in self.pred(u):
                if self.g[s] < self.g[u]:
                    u = s
            pose = Pose()
            pose.position.x = u[0]+0.5
            pose.position.y = u[1]+0.5
            path.poses.append(pose)
        return path

#### ALGORITHM #####
    def calc_key(self, s):
        """
        s = coordinate
        g(s) = np.full((grid_map.height, grid_map.width), np.inf)
        rhs(s) = np.full((grid_map.height, grid_map.width), np.inf)
        np.full(shape, fill_value) = new array of given shape and type, filled with fill_value
        """
        # +h(start_s,s) + k_m is chosen heuristic (we use 0)
        return [min(self.g[s], self.rhs[s]), min(self.g[s], self.rhs[s])]

    def initialize(self, goal):
        """
        goal(int, int) = goal position
        """
        self.U = PriorityQueue()
        #self.km= 0 #useless
        # s is already inf (elaborate g(s) rhs() with np.full np.inf)
        self.rhs[goal] = 0
        self.U.put(goal, self.calc_key(goal))

    def update_vertex(self, u, goal):
        """
        u(int, int) = current position
        s_(int,int) = neighbor of u
        """
        if u != goal:
            self.rhs[u] = self.min_succ(u)
        self.U.remove(u) #removes only if u in U
        if self.g[u] != self.rhs[u]:
            self.U.put(u, self.calc_key(u))

    def compute_shortest_path(self, start, goal):
        """
        find the shortest path
        """
        while not self.U.empty() and self.U.topKey() < self.calc_key(start) or self.rhs[start] != self.g[start]:
            k_old = self.U.topKey()
            u = self.U.pop()
            if k_old < self.calc_key(u):
                self.U.put(u, self.calc_key(u))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in self.pred(u):
                    self.update_vertex(s, goal)
            else:
                self.g[u] = np.inf
                for s in self.pred(u):
                    self.update_vertex(s, goal)
                self.update_vertex(u, goal)

    def plan_path_Dstar(self, grid_map, start, goal):
        """
        Do the "main" part of the following algorithm
        https://www.cs.cmu.edu/~maxim/files/dlite_tro05.pdf
        """   
        # Init (we need to have "original" data every iteration)
        data = grid_map.data.reshape(
            grid_map.height, grid_map.width).transpose()    

        #start = self.U.arg(self.min_succ(start))
        changed_edges = self.scan_graph(grid_map, data)
        if changed_edges != []:
            for edge in changed_edges:
                x = edge[0]
                y = edge[1]
                self.gridmap_data[x, y] = data[x, y] #Update the edge cost
                for s in self.pred([x, y]):
                    self.update_vertex(s, goal) #Update vertex U
            for element in self.U.print_elements():
                self.U.remove(element[1])
                self.U.put(element[1], self.calc_key(element[1])) #U.Update(s,calckey)
            self.compute_shortest_path(start, goal)
        
        if self.g[start] == np.inf:
            print("No path found")
            return None

        return self.format_path_Dstar(start, goal)
# END OF MY CODE WEEK 5 D-Star