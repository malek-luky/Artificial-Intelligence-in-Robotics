"""
Project: Artificial Intelligence for Robotics
Author: Lukas Malek
Email: malek.luky@gmail.com
Date: February 2023
"""

# STANDARD LIBRARIES
import math
import numpy as np
import copy
from messages import *
import matplotlib.pyplot as plt
import scipy.ndimage as ndimg
import heapq

# CUSTOM IMPORT
import sys
import heapq
import skimage.measure as skm
from sklearn.cluster import KMeans
np.set_printoptions(threshold=sys.maxsize)  # print full numpy array
from hexapod_robot.HexapodRobotConst import *

# PRIORITY QUEUE
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

###################################################################
# CLASS HEXAPOD EXPLORER
###################################################################
class HexapodExplorer:

    def __init__(self):
        pass

    def print_path(self, path):
        """
        Method to print the path point by point
        """
        if path is not None:
            if len(path.poses)==0:
                print("Path is empty")
            for i, pose in enumerate(path.poses):
                print("Point",i,":",pose.position.x,pose.position.y)
        else:
            print("Path is None")

    def update_free(self, P_mi):
        """
        Method to calculate the Bayesian update of the free cell with the current occupancy probability value P_mi 

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
        """
        Method to calculate the Bayesian update of the occupied cell with the current occupancy probability value P_mi

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

    def distance(self, start, end):
        '''
        Return the distance from the point A to the point B
        '''
        (x1, y1) = start
        (x2, y2) = end
        return math.sqrt(abs(x1 - x2)**2 + abs(y1 - y2)**2)

    def format_path(self, path, came_from, grid_map, start, goal):
        '''
        Return the path as requested in assignment as a list
        of positions[(x1, y1), (x2, y2), ...] At the moment the
        path is stored as the dicionary with at the format {(pos1):(pos2),..}
        and if you want to find the final path, you have to go backward,
        in other words start at the goal and than ask, "how I get there" till
        we will not reach the starting point
        '''
        path.poses.insert(0, self.map_to_world(
            goal, grid_map))  # Add goal!!
        coordinates = came_from.get(goal)
        while coordinates != start:
            pose = self.map_to_world(coordinates, grid_map)
            coordinates = came_from.get(coordinates)
            path.poses.insert(0, pose)
        pose = self.map_to_world(coordinates, grid_map)
        path.poses.insert(0, pose)  # Add start!!
        return path

    def neighbors_finder(self, current_pos, grid_map):
        '''
        Return the list of the neighbors of the current position
        return neighbors = [[(x1, y1), [(x2, y2), cost], ... ]
        '''
        neighbors = []
        (x, y) = current_pos
        for i in range(-1, 2):
            for j in range(-1, 2):
                # magic why indexes are switched
                if (i, j) != (0, 0) and grid_map.data[y + j, x + i] == 0:
                    neighbors.append((x + i, y + j))
        return neighbors
    
    def world_to_map(self, point, grid_map):
        """
        Return the coordinates of the point in the map from the pose
        """
        if type(point) == Vector3:
            point = (point.x, point.y)
        else:
            point = (point[0], point[1]) #possible floating point error
        map_origin = np.array([grid_map.origin.position.x, grid_map.origin.position.y])
        res = (point - map_origin) / grid_map.resolution
        return tuple(np.round(res).astype(int))   

    def map_to_world(self, coordinates, grid_map):
        '''
        Return the pose from the map coordinates
        '''
        pose = Pose()
        pose.position.x = np.round(coordinates[0] * grid_map.resolution+grid_map.origin.position.x,decimals=3)
        pose.position.y = np.round(coordinates[1] * grid_map.resolution+grid_map.origin.position.y,decimals=3)
        return pose

    def a_star(self, grid_map, start_pose, goal_pose):
        '''
        Generates the path. The method returns a list of coordinates for the
        path. It must start with the starting position and end at the goal
        position [(x1, y1), (x2, y2), … ]. If there is no path, it should
        return None.
        '''
        path = Path()
        frontier = list()
        start = self.world_to_map(start_pose.position, grid_map)
        goal = self.world_to_map(goal_pose.position, grid_map)
        heapq.heappush(frontier, (0, start))
        cost_so_far = dict() # {(x1,y1):cost1, (x2,y2):cost2, ..}
        cost_so_far[start] = 0
        came_from = dict()   # {(0, 0):None, (1, 2):(0, 1), ...}
        came_from[start] = None
        while frontier: # while not empty:
            current_pos = heapq.heappop(frontier)[1]
            if current_pos == goal:
                break
            neighbors = self.neighbors_finder(current_pos, grid_map)

            for next_pos in neighbors:
                new_cost = cost_so_far.get(
                    current_pos) + self.distance(next_pos, current_pos)
                if next_pos not in cost_so_far or new_cost < cost_so_far.get(next_pos):
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.distance(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current_pos
        if current_pos == goal and start is not None and goal is not None:
            ret = self.format_path(path, came_from, grid_map, start, goal)
            return ret
        else:
            return None

    def closest_frontier(self, start, frontiers,gridmap_processed):
        '''
        Return the closest frontier from the list of frontiers
        '''
        shortest_path = np.inf
        ret_path = None
        ret_frontier = None
        for frontier_tuple in frontiers:
            frontier = frontier_tuple[0]
            path = self.a_star(gridmap_processed, start, frontier)
            if path is not None and len(path.poses) < shortest_path:
                ret_path = path
                shortest_path = len(path.poses)
                ret_frontier = frontier
        return ret_path, ret_frontier


    def bresenham_line(self, start, goal):
        """
        Bresenham's line algorithm

        Args:
            start: (float64, float64) - start coordinate
            goal: (float64, float64) - goal coordinate
        Returns:
            (float64, float64) - interlying points between the start and goal coordinate
        """
        (x0, y0) = start
        (x1, y1) = goal
        x0 = int(x0)
        y0 = int(y0)
        x1 = int(x1)
        y1 = int(y1)
        line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while np.round(x) != np.round(x1): #floating point error?
                line.append((x,y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while np.round(y) != np.round(y1): #floating point error?
                line.append((x,y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        x = goal[0]
        y = goal[1]
        return line

    def plot_graph(self, grid_map):
        """
        Plot the graphs in case error occurs and we want to see the map
        """
        data = grid_map.data.reshape(grid_map.height, grid_map.width)
        _, ax = plt.subplots()
        plt.imshow(data, cmap='viridis')
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.colorbar()
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        plt.title("pixel_plot")
        plt.show()

    def fuse_laser_scan(self, grid_map, laser_scan, odometry):
        """
        Method to fuse the laser scan data sampled by the robot with a given 
        odometry into the probabilistic occupancy grid map

        Args:
            grid_map_update: OccupancyGrid - gridmap to fuse te laser scan to
            laser_scan: LaserScan - laser scan perceived by the robot
            odometry: Odometry - perceived odometry of the robot
        Returns:
            grid_map_update: OccupancyGrid - gridmap updated with the laser scan data
        """
        grid_map_update = copy.deepcopy(grid_map)
        if laser_scan is not None and odometry is not None:
            laserscan_points = laser_scan.distances
            alpha_min = laser_scan.angle_min
            odometry_R = odometry.pose.orientation.to_R()
            odometry_x = odometry.pose.position.x
            odometry_y = odometry.pose.position.y
            free_points = list()
            occupied_points = list()
            for laserscan_sector, laserscan_sector_value in enumerate(laserscan_points):
                '''
                STEP 1: Project the laser scan points to x,y plane with respect to the robot heading
                '''
                if laserscan_sector_value < laser_scan.range_min or laserscan_sector_value > laser_scan.range_max:
                    continue # ignore invalid laserscan values
                theta_i = alpha_min + laserscan_sector*laser_scan.angle_increment
                point_x = laserscan_sector_value*np.cos(theta_i)
                point_y = laserscan_sector_value*np.sin(theta_i)

                '''
                STEP 2: Compensate for the robot odometry
                '''
                point_normalized = np.transpose(
                    np.array([point_x, point_y, 1]))
                point_x_global, point_y_global, _ = odometry_R @ point_normalized + \
                    np.array([odometry_x, odometry_y, 1])

                '''
                STEP 3: Transfer the points from the world coordinates to the map coordinates
                '''
                point_x_map, point_y_map = self.world_to_map(
                    [point_x_global, point_y_global], grid_map_update)

                '''
                STEP 4: Raytrace individual scanned points
                '''
                odometry_x_map, odometry_y_map = self.world_to_map(
                    [odometry_x, odometry_y], grid_map_update)
                pts = self.bresenham_line(
                    (odometry_x_map, odometry_y_map), (point_x_map, point_y_map))
                free_points.extend(pts)
                occupied_points.append((point_x_map, point_y_map))

            '''
            STEP 5: Dynamically resize the map
            '''
            data = grid_map_update.data.reshape(grid_map_update.height, grid_map_update.width)
            for (x, y) in [(odometry_x_map,odometry_y_map)] + occupied_points: # for the first loop we need to resize the map based on robot
                if x < 0 or y < 0 or data is None or x >= grid_map.width or y >= grid_map.height:
                    x_shift = min(0, x)     # negative coordinate -> we need to shift the origin
                    y_shift = min(0, y)
                    new_height = max(grid_map.height if data is not None else 0, y+1) - y_shift
                    new_width = max(grid_map.width if data is not None else 0, x+1) - x_shift
                    new_origin = np.array([grid_map.origin.position.x, grid_map.origin.position.y]) + np.array([x_shift, y_shift]) * grid_map.resolution
                    new_data = 0.5 * np.ones((new_height, new_width))
                    if data is not None:
                        new_data[-y_shift:-y_shift+grid_map.height, -x_shift:-x_shift+grid_map.width] = data
                    grid_map_update.width = new_width
                    grid_map_update.height = new_height
                    grid_map_update.origin = Pose(Vector3(new_origin[0], new_origin[1], 0.0), Quaternion(1, 0, 0, 0))
                    grid_map_update.data = new_data.flatten()
                    return self.fuse_laser_scan(grid_map_update, laser_scan, odometry)

            '''
            STEP 6: Update the occupancy grid using the Bayesian update and the simplified laser scan sensor model
            '''
            for (x, y) in occupied_points:
                data[y, x] = self.update_occupied(data[y, x])
            for (x, y) in free_points:
                data[y, x] = self.update_free(data[y, x])
            grid_map_update.data = data.flatten() 
        return grid_map_update

    def grow_obstacles(self, grid_map, robot_size):
        """ Method to grow the obstacles to take into account the robot embodiment
        Args:
            grid_map: OccupancyGrid - gridmap for obstacle growing
            robot_size: float - size of the robot
        Returns:
            grid_map_grow: OccupancyGrid - gridmap with considered robot body embodiment
        Notes:
            use obstacle growing only on obstacles, not uknown areas
            filter all unknown areas moved to the end of the function
        """

        grid_map_grow = copy.deepcopy(grid_map)
        grid_map_grow.data[grid_map.data > 0.5] = 1  # obstacles
        grid_map_grow.data[grid_map.data <= 0.5] = 0  # free area
        kernel_size = round(robot_size/grid_map_grow.resolution)  # must be even
        r = round(kernel_size) # filter cells close to obstacles
        kernel = np.fromfunction(lambda x, y: ((x-r)**2 + (y-r)**2 < r**2)*1, (2*r+1, 2*r+1), dtype=int).astype(np.uint8)
        grid_map_grow.data = ndimg.convolve(grid_map_grow.data, kernel)
        grid_map_grow.data[grid_map_grow.data > 1] = 1
        grid_map_grow.data[grid_map.data == 0.5] = 1  # unknown area
        return grid_map_grow

    def simplify_path(self, grid_map, path):
        """
        Method to simplify the found path on the grid
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
        i = 1
        while path_simple.poses[-1] != path.poses[-1]: #while goal not reached
            last_pose = path_simple.poses[-1]
            for pose in path.poses[i::]:   
                end = path_simple.poses[-1]            
                bres_line = self.bresenham_line(self.world_to_map(end.position,grid_map),
                                                self.world_to_map(pose.position,grid_map))                
                collision = False
                for (x, y) in bres_line:
                    if grid_map.data[y,x] != 0: # this is correct!
                        collision = True
                if collision == False:
                    last_pose = pose
                    i += 1
                    if pose == path.poses[-1]:  # goal is reached
                        path_simple.poses.append(pose)
                        break
                else:
                    path_simple.poses.append(last_pose)
                    break
        path_simple.poses.pop(0) # remove the start pose
        return path_simple

    def find_free_edge_frontiers(self, grid_map):
        """
        Method to find the free-edge frontiers (edge clusters between the free and unknown areas)

        Args:
            grid_map: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """

        data = copy.deepcopy(grid_map.data)
        data[data == 0.5] = 10
        mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        data_c = ndimg.convolve(data, mask, mode='constant', cval=0.0)

        # Format result
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
            """
            f1 Implementation
            """
            # centroid = (0, 0)
            # for item in items:
            #     centroid = (centroid[0]+item[0], centroid[1]+item[1])
            # centroid = (centroid[0]/len(items), centroid[1]/len(items))
            # pose = self.map_to_world(
            #     [centroid[0], centroid[1]], grid_map)
            # pose_list.append(pose)

            """
            f2 Implementation
            """
            f = len(items)
            D = LASER_MAX_RANGE / grid_map.resolution
            n_r = np.floor(f/D + 0.5) +1
            kmeans = KMeans(n_clusters=int(n_r), max_iter=70, tol=1e-2, n_init=1).fit(items)
            for centroid in kmeans.cluster_centers_:
                pose_list.append(self.map_to_world(centroid, grid_map))
        return pose_list

    def find_inf_frontiers(self, grid_map):
        """
        f3 Implementation
        Method to find the information frontiers (frontiers with the highest information gain)
        Args:
            grid_map: OccupancyGrid - gridmap of the environment
        Returns:
            pose_list: Pose[] - list of selected frontiers
        """

        # Calculate entropy of each cell in the grid
        H = grid_map.data.copy()
        for row,row_val in enumerate(H):
            for col,col_val in enumerate(row_val): 
                p = H[row][col]
                if p == 0 or p == 1:
                    H[row][col] = 0
                else:
                    H[row][col] = -p * np.log(p) - (1-p) * np.log(1-p)

        # Find free frontiers
        frontiers = self.find_free_edge_frontiers(grid_map)

        # Calculate information gain of each frontier
        frontiers_weighted = []
        rays = 8
        for frontier in frontiers:
            frontier_cell = self.world_to_map(frontier.position, grid_map)

            # Calculate information gain along 8 rays from the frontier
            I_action = 0.0
            for i in range(rays):
                ray_end_pose = Pose()
                ray_end_pose.position.x = frontier.position.x + math.cos(i * math.pi/rays) * LASER_MAX_RANGE
                ray_end_pose.position.y = frontier.position.y + math.sin(i * math.pi/rays) * LASER_MAX_RANGE
                ray_end_cell = self.world_to_map(ray_end_pose.position, grid_map)
                ray = self.bresenham_line(frontier_cell, ray_end_cell)

                # Accumulate information gain along the ray
                for x, y in ray:
                    if x < 0 or x >= grid_map.width or y < 0 or y >= grid_map.height:
                        break    # ray reaches map bounds
                    if grid_map.data[y, x] == 1:
                        break    # ray reaches obstacle
                    I_action += H[y, x]
            frontiers_weighted.append((frontier, I_action))

        return frontiers_weighted