"""
Project: Artificial Intelligence for Robotics
Author: Lukas Malek
Email: malek.luky@gmail.com
Date: February 2023
"""

# STANDARD LIBRARIES
import sys
import time
import threading as thread
import numpy as np
import matplotlib.pyplot as plt
import copy
 
# IMPORT EHAXPOD CLASSES
sys.path.append('../')
sys.path.append('hexapod_robot')
sys.path.append('hexapod_explorer')
import HexapodRobot 
import HexapodExplorer
from hexapod_robot.HexapodRobotConst import *
from messages import *
import pretty_errors
from lkh.invoke_LKH import solve_TSP

# CHOSE THE SETUP
planning = "p2" #p2/p3


 
###################################################################
# CLASS EXPLORER
###################################################################
class Explorer:
    """
    Class to represent an exploration agent
    """
    def __init__(self, robotID = 0):
        
        """
        Variables
        """
        print(time.strftime("%H:%M:%S"),"Initializing Explorer class...")
        self.gridmap = OccupancyGrid()
        self.gridmap.resolution = 0.1
        self.gridmap.width = 2 #m2 - otherwise default is 100
        self.gridmap.height = 2 #m2 - for m1 we need origin self.gridmap.origin = Pose(Vector3(-5.0,-5.0,0.0), Quaternion(1,0,0,0))
        self.gridmap.data = 0.5*np.ones((self.gridmap.height,self.gridmap.width)) #unknown (grey) area

        """
        Initialize values
        """
        self.frontiers = None
        self.frontier = None
        self.path = None
        self.path_simple = None
        self.stop = False # stop condition for the threads
        self.nav_goal = None
        self.collision = False

        """
        Connecting the simulator
        """
        self.robot = HexapodRobot.HexapodRobot(robotID)
        self.explor = HexapodExplorer.HexapodExplorer()
 
    def start(self):
        """
        Connect to the robot
        """
        print(time.strftime("%H:%M:%S"),"Starting threads...")
        self.robot.turn_on()
 
        """
        Start navigation thread
        """
        self.robot.start_navigation()
 
        """
        Start navigation thread
        """
        try:
            mapping_thread = thread.Thread(target=self.mapping)
            mapping_thread.start() 
        except:
            print(time.strftime("%H:%M:%S"),"Error: unable to start mapping thread")
            sys.exit(1)
 
        """
        Start planning thread
        """
        try:
            planning_thread = thread.Thread(target=self.planning)
            planning_thread.start() 
        except:
            print(time.strftime("%H:%M:%S"),"Error: unable to start planning thread")
            sys.exit(1)
 
        """
        Start trajectory following thread
        """
        try:
            traj_follow_thread = thread.Thread(target=self.trajectory_following)
            traj_follow_thread.start() 
        except:
            print(time.strftime("%H:%M:%S"),"Error: unable to start planning thread")
            sys.exit(1)
 
    def __del__(self):
        """
        Turn off the robot
        """
        self.robot.stop_navigation()
        self.robot.turn_off()
 
    def mapping(self):
        """
        Mapping thread for fusing the laser scans into the grid map
        """
        while not self.stop:
            """
            Obstacle growing: p1
            """
            time.sleep(THREAD_SLEEP)
            odometry = self.robot.odometry_.pose
            robot_pos = self.explor.world_to_map(odometry.position,self.gridmap)
            radius = int((ROBOT_SIZE+EXTRA_PADDING)/self.gridmap.resolution)
            self.gridmap = self.explor.fuse_laser_scan(self.gridmap, self.robot.laser_scan_, self.robot.odometry_)
            self.gridmap.data = self.gridmap.data.reshape(self.gridmap.height, self.gridmap.width)
            self.gridmap_processed = self.explor.grow_obstacles(self.gridmap, ROBOT_SIZE, robot_pos, radius)
            self.gridmap_astar = self.explor.grow_obstacles(self.gridmap, ROBOT_SIZE+EXTRA_PADDING, robot_pos, radius) # extra safety margin so we dont recalculate the path too often
        print(time.strftime("%H:%M:%S"),"Mapping thread terminated successfully!")
            

    def planning(self):
        """
        Find frontiers, select the next goal and path  
        """
        time.sleep(4*THREAD_SLEEP) #wait for map init
        timeout = 0 # timeout counter
        while not self.stop:
            time.sleep(THREAD_SLEEP) #wait for propper map init           
            """
            Check for collision from growing obstacles
            """
            self.collision = False
            if self.path is not None:
                for point in self.path.poses:
                    (x,y) = self.explor.world_to_map(point.position,self.gridmap)
                    if self.gridmap_processed.data[y,x] == 1:
                        self.robot.stop() #stop robot
                        self.collision = True
                        self.path_simple = None
                        self.path = None
                        self.frontier = None
                        print(time.strftime("%H:%M:%S"),"Collision detected! Rerouting...")
                        break

            """
            Check for timeout
            """
            if timeout > 10:
                self.stop = True
                print(time.strftime("%H:%M:%S"),"No frontiers found 10 times in a row, stopping...")

            """
            Reroute only if collision is detected or if the goal is reached
            """
            self.frontiers = self.explor.remove_frontiers(self.gridmap_astar, self.frontiers) # remove frontiers using newly found obstacles
            if not self.collision and self.path_simple is not None:# and (self.robot.navigation_goal is not None or len(self.path_simple.poses) == 0):
                continue # look for frontiers only if collision or reached goal, don't look for frontiers last step before the goal
            
            """
            Find frontiers based on the chosen strategy
            p1: only find the frontiers
            p2: find frontiers and compute heuristic
            p3: only find the frontiers
            f1: only frontiers
            f2: fronteirs + KMeans approach (default for p1 and p3)
            f3: KMeans + information gain from each frontier
            """
            if planning == "p1" or planning=="p3":
                self.frontiers = self.explor.find_free_edge_frontiers(self.gridmap)
            elif planning == "p2":
                self.frontiers = self.explor.find_inf_frontiers(self.gridmap)
            self.frontiers = self.explor.remove_frontiers(self.gridmap_astar, self.frontiers) # remove frontiers in obstacles
            if len(self.frontiers) ==0:
                timeout+=1
                print(time.strftime("%H:%M:%S"),"No frontiers found. Timeout: ", timeout,"/10")
                continue #skip the frontier selection
            start = self.robot.odometry_.pose

            """
            p1: Select closest frontier and find the path
            """
            if planning == "p1":
                self.frontier = self.explor.closest_frontier(start, self.frontiers, self.gridmap_astar)
                self.path = self.explor.a_star(self.gridmap_astar, start, self.frontier) #we know that frontier is reachable
                if self.path is not None:
                    print(time.strftime("%H:%M:%S"),"New shortest path found!")
                    self.path_simple = self.explor.simplify_path(self.gridmap_astar, self.path)
                    timeout = 0
                else:
                    self.path_simple = None
                    timeout+=1
                    print(time.strftime("%H:%M:%S"),"No new path without collision! Timeout: ", timeout,"/10")
            """
            p2: Select the frontier with highest inforation gain
            """
            if planning == "p2":
                self.frontiers = sorted(self.frontiers, key=lambda x: x[1], reverse=True) #TODO: might be faster to just find the max
                for frontier in self.frontiers:
                    self.path = self.explor.a_star(self.gridmap_astar, start, frontier[0])
                    if self.path is not None:
                        self.frontier = frontier[0]
                        self.path_simple = self.explor.simplify_path(self.gridmap_astar, self.path)
                        timeout = 0
                        print(time.strftime("%H:%M:%S"),"New path with highest information gain found!")
                        break
                if self.path is None:
                    self.path_simple = None
                    timeout+=1
                    print(time.strftime("%H:%M:%S"),"No new path without collision! Timeout: ", timeout,"/10")
                    
            """
            p3: Select the best route using Travel Salesman Problem and distance matrix
            """
            if planning == "p3":
                if len(self.frontiers) == 1: # TSP cannot have dim <3
                    goal = self.frontiers[0]
                else:
                    TSP_matrix = copy.deepcopy(self.frontiers)
                    TSP_matrix.insert(0, start)
                    distance_matrix = np.zeros((len(TSP_matrix), len(TSP_matrix)))
                    for row in range(len(TSP_matrix)): #rows are normqal
                        for col in range(1, len(TSP_matrix)): #column must be zero as we do not loop back to the start (so called open-ended)
                            path = self.explor.a_star(self.gridmap_astar, TSP_matrix[row], TSP_matrix[col])
                            if path is None:
                                distance_matrix[row][col] = 999999
                            else:
                                distance_matrix[row][col] = len(path.poses)
                    TSP_result = solve_TSP(distance_matrix) #[0, 7, 6, 3, 5, 4, 1, 2] order hot to follow frontiers
                    goal = TSP_matrix[TSP_result[1]] # first index so wwe skip the zero index, - because the frontiers are withotu domoetry positio
                self.path = self.explor.a_star(self.gridmap_astar, start, goal)
                if self.path is not None:
                    self.frontier = goal
                    self.path_simple = self.explor.simplify_path(self.gridmap_astar, self.path)
                    timeout = 0
                    print(time.strftime("%H:%M:%S"),"New best TSP path found!")
                else:
                    self.path_simple = None
                    timeout+=1
                    print(time.strftime("%H:%M:%S"),"No new path without collision! Timeout: ", timeout,"/10")


        print(time.strftime("%H:%M:%S"),"Planning thread terminated successfully!")


 
    def trajectory_following(self):
        """
        Assigns new goals to the robot when the previous one is reached
        """ 
        time.sleep(6*THREAD_SLEEP) #wait for plan init
        while not self.stop: 
            time.sleep(THREAD_SLEEP)
            if self.collision: # collision - new route
                self.robot.stop() #stop robot
                continue          
            if self.robot.navigation_goal is None and self.path_simple is not None:
                if len(self.path_simple.poses)==0: #reached the goal
                    self.robot.stop() #stop robot
                    self.path_simple = None
                    self.path = None
                    print(time.strftime("%H:%M:%S"), "Goal reached, waiting for new route")
                else: #cotninue with the old route
                    self.nav_goal = self.path_simple.poses.pop(0)
                    self.robot.goto(self.nav_goal)
                    print(time.strftime("%H:%M:%S"),"Goto: ", self.nav_goal.position.x, self.nav_goal.position.y)        
        print(time.strftime("%H:%M:%S"),"Trajectory following thread terminated successfully!")
            
 
 
if __name__ == "__main__":
    """
    Initiaite robot and start threads
    """
    expl = Explorer()
    expl.start()
    time.sleep(16*THREAD_SLEEP) #wait for everything to init

    """
    Initiate plotting
    """
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(5,10))
    plt.ion()
    ax0.set_xlabel('x[m]')
    ax0.set_ylabel('y[m]')
    ax1.set_xlabel('x[m]')
    ax1.set_ylabel('y[m]')
    ax2.set_xlabel('x[m]')
    ax2.set_ylabel('y[m]')
    ax0.set_aspect('equal', 'box')
    ax1.set_aspect('equal', 'box')
    ax2.set_aspect('equal', 'box')

    while not expl.stop:
        plt.pause(THREAD_SLEEP)
        ax0.cla() #clear the points from the previous iteration
        ax1.cla()
        ax2.cla()
        if expl.gridmap.data is not None and expl.gridmap_processed.data is not None and expl.gridmap_astar.data is not None: #grid
            expl.gridmap.plot(ax0)
            expl.gridmap_processed.plot(ax1)
            expl.gridmap_astar.plot(ax2)
        for frontier in expl.frontiers: #frontiers
            if type(frontier) != Pose:
                frontier = frontier[0]
            ax0.scatter(frontier.position.x, frontier.position.y,c='red')
            ax1.scatter(frontier.position.x, frontier.position.y,c='red')
            ax2.scatter(frontier.position.x, frontier.position.y,c='red')
        if expl.path is not None: #path
            expl.path.plot(ax0, style = 'point')
            expl.path.plot(ax1, style = 'point')
            expl.path.plot(ax2, style = 'point')
        if expl.path_simple is not None: #print simple path points
            for pose in expl.path_simple.poses:
                ax0.scatter(pose.position.x, pose.position.y,c='red', s=150, marker='x')
                ax1.scatter(pose.position.x, pose.position.y,c='red', s=150, marker='x')
                ax2.scatter(pose.position.x, pose.position.y,c='red', s=150, marker='x')
        if expl.robot.odometry_.pose is not None: #robot
            ax0.scatter(expl.robot.odometry_.pose.position.x, expl.robot.odometry_.pose.position.y,c='grey', s=150, marker='o')
            ax1.scatter(expl.robot.odometry_.pose.position.x, expl.robot.odometry_.pose.position.y,c='grey', s=150, marker='o')
            ax2.scatter(expl.robot.odometry_.pose.position.x, expl.robot.odometry_.pose.position.y,c='grey', s=150, marker='o')
        if expl.nav_goal is not None: #frontier
            ax0.scatter(expl.nav_goal.position.x, expl.nav_goal.position.y,c='black', s=150, marker='x')
            ax1.scatter(expl.nav_goal.position.x, expl.nav_goal.position.y,c='black', s=150, marker='x')
            ax2.scatter(expl.nav_goal.position.x, expl.nav_goal.position.y,c='black', s=150, marker='x')

        plt.show()
    expl.__del__() #turn it off
        