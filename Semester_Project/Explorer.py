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
from hexapod_robot.HexapodRobotConst import *
from messages import *

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
            time.sleep(THREAD_SLEEP)
            self.gridmap = self.explor.fuse_laser_scan(self.gridmap, self.robot.laser_scan_, self.robot.odometry_)
            self.gridmap.data = self.gridmap.data.reshape(self.gridmap.height, self.gridmap.width)
            self.gridmap_processed = self.explor.grow_obstacles(self.gridmap, ROBOT_SIZE)
            

    def planning(self):
        """
        Find frontiers, select the next goal and path  
        """
        time.sleep(THREAD_SLEEP+1) #wait for map init
        
        # START OF MY CODE f1 + p1
        while not self.stop:
            time.sleep(3*THREAD_SLEEP) #wait for propper map init
            """
            f1+f2+f3: Find using heurestic and KMeans approach, each part is build on top of the other one: 
            """
            if self.goal_reached: #wait until previous goal is reached
                self.frontiers = self.explor.find_inf_frontiers(self.gridmap)
                for frontier_tuple in self.frontiers: #remove frontiers which are in obstacle
                    frontier = frontier_tuple[0]
                    (x,y) = self.explor.world_to_map(frontier.position,self.gridmap_processed)
                    if self.gridmap_processed.data[y,x] == 1:
                        self.frontiers.remove(frontier_tuple)

            if self.goal_reached: # plan a new goal (aka wait until previous goal is reached)
                if len(self.frontiers) !=0 and self.robot.odometry_ is not None:
                    shortest_path = np.inf
                    start = self.robot.odometry_.pose
                    """
                    p1: Select closest frontier and find the path
                    """
                    if planning == "p1":
                        for frontier_tuple in self.frontiers:
                            frontier = frontier_tuple[0]
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
                    """
                    p2: Select closest frontier and find the path
                    """
                    if planning == "p2":
                        while self.path_simple is None: # rerun if the chosen frontier is unreachable
                            self.frontiers = sorted(self.frontiers, key=lambda x: x[1], reverse=True)
                            max_entrophy = self.frontiers[0][1]
                            filtered_lst = [item for item in self.frontiers if item[1] == max_entrophy] #filter frontiers with same entrophy
                            if len(filtered_lst) > 1:
                                min_distance = np.inf
                                for frontier_tuple in filtered_lst:
                                    frontier = frontier_tuple[0]
                                    distance = self.explor.distance((frontier.position.x, frontier.position.y), (start.position.x, start.position.y))
                                    if distance < min_distance:
                                        min_distance = distance
                                        self.closest_frontier = frontier # highest heuristic + closest
                            else:
                                self.closest_frontier = filtered_lst[0][0]
                            self.path = self.explor.a_star(self.gridmap_processed, start, self.closest_frontier)
                            self.path_simple = self.explor.simplify_path(self.gridmap_processed, self.path)
                            if self.path_simple is None:
                                print("MISTAKE!")
                                self.frontiers.pop(0) #delete the unreachable frontier #TODO: useless right now... while loop?
                            if len(self.frontiers) == 0:
                                self.path_simple = None
                                self.path = None
                                print(time.strftime("%H:%M:%S"),"No new path without collision!") #TODO
                                break
                        print(time.strftime("%H:%M:%S"),"New shortest path found!") #TODO
                        self.goal_reached = False #TODO
                    """
                    p3:
                    """
                    

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
                    self.rerouting = True # option 3: no collision - do nothing
 
    def trajectory_following(self):
        """
        Assigns new goals to the robot when the previous one is reached
        """ 
        time.sleep(3*THREAD_SLEEP) #wait for plan init
        while not self.stop: 
            time.sleep(THREAD_SLEEP)
            if self.rerouting:
                time.sleep(10*THREAD_SLEEP) #wait for a new route    
                if self.path_simple is not None:
                    self.rerouting = False
                    self.nav_goal = self.path_simple.poses.pop(0)
                    print(time.strftime("%H:%M:%S"),"Replanning, new route: ", self.nav_goal.position.x, self.nav_goal.position.y)
                    self.robot.goto(self.nav_goal)            
            if self.robot.navigation_goal is None and self.path_simple is not None:
                if self.goal_reached == False and len(self.path_simple.poses)==0:
                    print(time.strftime("%H:%M:%S"), "Goal reached, waiting for new route")
                    print("Frontiers: ", len(self.path_simple.poses))
                    time.sleep(10*THREAD_SLEEP) #wait for a new route
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
    time.sleep(10*THREAD_SLEEP) #wait for everything to init

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
        plt.pause(THREAD_SLEEP)
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
        if expl.robot.odometry_.pose is not None: #robot
            ax0.scatter(expl.robot.odometry_.pose.position.x, expl.robot.odometry_.pose.position.y,c='black', s=150, marker='x')
            ax1.scatter(expl.robot.odometry_.pose.position.x, expl.robot.odometry_.pose.position.y,c='black', s=150, marker='x')
        for frontier_tuple in expl.frontiers: #frontiers
            frontier = frontier_tuple[0]
            ax0.scatter(frontier.position.x, frontier.position.y)
            ax1.scatter(frontier.position.x, frontier.position.y)
        plt.show()
        