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
 
# IMPORT EHAXPOD CLASSES
sys.path.append('../')
sys.path.append('hexapod_robot')
sys.path.append('hexapod_explorer')
import HexapodRobot 
import HexapodExplorer
import HexapodRobotConst as Constants
from messages import *
 
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
        
        """
        Task m1
        """
        self.gridmap.width = 1
        self.gridmap.height = 1
        self.gridmap.origin = Pose(Vector3(-5.0,-5.0,0.0), Quaternion(1,0,0,0))
        self.gridmap.data = 0.5*np.ones((self.gridmap.height,self.gridmap.width)) #unknown

        """
        Initialize values
        """
        self.frontiers = None
        self.path = None
        self.path_simple = None
        self.stop = False # stop condition for the threads
        self.goal_reached = True
        self.rerouting = False
 
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
            time.sleep(Constants.THREAD_SLEEP)
            print(time.strftime("%H:%M:%S"),"Mapping...")
            self.gridmap = self.explor.fuse_laser_scan(self.gridmap, self.robot.laser_scan_, self.robot.odometry_)
            self.gridmap.data = self.gridmap.data.reshape(self.gridmap.height, self.gridmap.width)
            self.gridmap_processed = self.explor.grow_obstacles(self.gridmap, Constants.ROBOT_SIZE)
            

    def planning(self):
        """
        Find frontiers, select the next goal and path  
        """
        time.sleep(Constants.THREAD_SLEEP+1) #wait for map init
        
        # START OF MY CODE f1 + p1
        while not self.stop:
            time.sleep(Constants.THREAD_SLEEP)
            """
            Find frontiers: f1
            """
            if self.goal_reached: #wait until previous goal is reached
                self.frontiers = self.explor.find_free_edge_frontiers_f1(self.gridmap)
                for frontier in self.frontiers: #remove frontiers which are in obstacle
                    (x,y) = self.explor.world_to_map(frontier.position,self.gridmap)
                    if self.gridmap_processed.data[y,x] == 1:
                        self.frontiers.remove(frontier)
            """
            Select closest frontier and find the path: p1
            """
            if self.goal_reached: #wait until previous goal is reached
                if len(self.frontiers) !=0 and self.robot.odometry_ is not None:
                    shortest_path = np.inf
                    start = self.robot.odometry_.pose
                    for frontier in self.frontiers:
                        path = self.explor.a_star(self.gridmap_processed, start, frontier)
                        path_simple = self.explor.simplify_path(self.gridmap_processed, path)
                        if path_simple is not None and len(path_simple.poses) < shortest_path:
                            self.path_simple = path_simple
                            self.path = path
                            shortest_path = len(path_simple.poses)
                            self.closest_frontier = frontier
                    if self.path_simple is not None:
                        print(time.strftime("%H:%M:%S"),"New shortest path found!")
                        self.goal_reached = False
                    else:
                        print(time.strftime("%H:%M:%S"),"No new path without collision!")
                else:
                    print(time.strftime("%H:%M:%S"),"No frontiers found")
            """
            Check for collision from growing obstacles
            """
            if not self.goal_reached and self.path.poses is not None:
                collision = False
                for point in self.path.poses: # use the old route!!
                    (x,y) = self.explor.world_to_map(point.position,self.gridmap)
                    if self.gridmap_processed.data[y,x] == 1:
                        collision = True
                        break
                if collision == True: # if collision or unreachable, find new path
                    print(time.strftime("%H:%M:%S"),"Goal unreachable or collision... Finding new frontiers!")
                    self.path_simple = None
                    self.path = None
                    self.goal_reached = True
                    self.rerouting = True
                else: # option 3: no collision
                    print(time.strftime("%H:%M:%S"),"Reaching previous goal")
 
    def trajectory_following(self):
        """
        Assigns new goals to the robot when the previous one is reached
        """ 
        time.sleep(3*Constants.THREAD_SLEEP) #wait for plan init
        while not self.stop: 
            time.sleep(Constants.THREAD_SLEEP)
            if self.rerouting and self.path_simple is not None:
                time.sleep(Constants.THREAD_SLEEP) #wait for a new route    
                self.rerouting = False
                self.nav_goal = self.path_simple.poses.pop(0)
                print(time.strftime("%H:%M:%S"),"Replanning, new route: ", self.nav_goal.position.x, self.nav_goal.position.y)
                self.robot.goto(self.nav_goal)            
            if self.robot.navigation_goal is None and self.path_simple is not None:
                if self.goal_reached == False and len(self.path_simple.poses)==0:
                    print(time.strftime("%H:%M:%S"), "Goal reached, waiting for new route")
                    time.sleep(Constants.THREAD_SLEEP+1) #wait for a new route
                    self.goal_reached = True
                elif self.goal_reached == False and len(self.path_simple.poses) != 0: #cotninue with the old route
                    self.nav_goal = self.path_simple.poses.pop(0)
                    print(time.strftime("%H:%M:%S"),"Goto: ", self.nav_goal.position.x, self.nav_goal.position.y)
                    self.robot.goto(self.nav_goal)
            elif self.robot.navigation_goal is not None:
                odom = self.robot.odometry_.pose
                odom.position.z = 0 
                dist = self.nav_goal.dist(odom) # print(dist)
            
 
 
if __name__ == "__main__":
    """
    Initiaite robot and start threads
    """
    expl = Explorer()
    expl.start()
    time.sleep(10*Constants.THREAD_SLEEP) #wait for everything to init

    """
    Initiate plotting
    """
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(5,10))
    plt.ion()
    ax0.set_xlabel('x[m]')
    ax0.set_ylabel('y[m]')
    ax1.set_xlabel('x[m]')
    ax1.set_ylabel('y[m]')
    ax0.set_aspect('equal', 'box')
    ax1.set_aspect('equal', 'box')

    """
    Contrinuously plot the map
    """
    while(1):
        plt.pause(Constants.THREAD_SLEEP)
        print(time.strftime("%H:%M:%S"),"Plotting...")
        ax0.cla() #clear the points from the previous iteration
        ax1.cla()
        if expl.gridmap.data is not None: #grid
            expl.gridmap.plot(ax0)
        if expl.gridmap_processed.data is not None:
            expl.gridmap_processed.plot(ax1)
        if expl.path_simple is not None: #path
            expl.path_simple.plot(ax0)
            expl.path.plot(ax1)
            ax0.scatter(expl.nav_goal.position.x, expl.nav_goal.position.y,c='red', s=150, marker='x')
            ax1.scatter(expl.nav_goal.position.x, expl.nav_goal.position.y,c='red', s=150, marker='x')
            ax0.scatter (expl.closest_frontier.position.x, expl.closest_frontier.position.y,c='green', s=100, marker='x')
            ax1.scatter (expl.closest_frontier.position.x, expl.closest_frontier.position.y,c='green', s=100, marker='x')
        # if expl.robot.odometry_.pose is not None: #robot
        #     ax0.scatter(expl.robot.odometry_.pose.position.x, expl.robot.odometry_.pose.position.y,c='black', s=150, marker='x')
        #     ax1.scatter(expl.robot.odometry_.pose.position.x, expl.robot.odometry_.pose.position.y,c='black', s=150, marker='x')
        for frontier in expl.frontiers: #frontiers
            ax0.scatter(frontier.position.x, frontier.position.y)
            ax1.scatter(frontier.position.x, frontier.position.y)
        plt.show()
        