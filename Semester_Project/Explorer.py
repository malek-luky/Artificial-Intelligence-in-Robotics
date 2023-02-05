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
import pretty_errors

# CHOSE THE SETUP
planning = "p1" #p2/p3


 
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
        self.gridmap.width = 1 #m2 - otherwise default is 100
        self.gridmap.height = 1 #m2 - for m1 we need origin self.gridmap.origin = Pose(Vector3(-5.0,-5.0,0.0), Quaternion(1,0,0,0))
        self.gridmap.data = 0.5*np.ones((self.gridmap.height,self.gridmap.width)) #unknown (grey) area

        """
        Initialize values
        """
        self.frontiers = None
        self.path = None
        self.path_simple = None
        self.stop = False # stop condition for the threads
        self.nav_goal = None

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
        print(time.strftime("%H:%M:%S"),"Mapping thread terminated successfully!")
            

    def planning(self):
        """
        Find frontiers, select the next goal and path  
        """
        time.sleep(2*THREAD_SLEEP) #wait for map init
        timeout = 0 # timeout counter
        
        # START OF MY CODE f1 + p1
        while not self.stop:
            time.sleep(3*THREAD_SLEEP) #wait for propper map init
            """
            Remove frontiers in obstacles
            """
            
            
            """
            Check for collision from growing obstacles
            """
            collision = False
            if self.path is not None:
                for point in self.path.poses:
                    (x,y) = self.explor.world_to_map(point.position,self.gridmap)
                    if self.gridmap_processed.data[y,x] == 1:
                        self.robot.stop() #stop robot
                        collision = True
                        self.path_simple = None
                        self.path = None
                        print(time.strftime("%H:%M:%S"),"Collision detected! Rerouting...")
                        break
            """
            Reroute only if collision is detected or if the goal is reached
            """
            self.frontiers = self.explor.remove_frontiers(self.gridmap_processed, self.frontiers) # remove frontiers in newly found obstacles
            if not collision and self.path_simple is not None and self.robot.navigation_goal is not None:
                continue # look for frontiers only if collision or reached goal

            """
            Find frontiers
            """
            
            """
            f1+f2+f3: Find using heurestic and KMeans approach, each part is build on top of the other one: 
            """
            self.frontiers = self.explor.find_inf_frontiers(self.gridmap) #find frontiers
            self.frontiers = self.explor.remove_frontiers(self.gridmap_processed, self.frontiers) # remove frontiers in obstacles
            if len(self.frontiers) ==0:
                timeout+=1
                print(time.strftime("%H:%M:%S"),"No frontiers found")
                if timeout > 10:
                    self.stop = True
                    print(time.strftime("%H:%M:%S"),"No frontiers found 10 times in a row, stopping...")
                continue
            timeout = 0
            start = self.robot.odometry_.pose
            """
            p1: Select closest frontier and find the path
            """
            if planning == "p1":
                self.path, self.frontier = self.explor.closest_frontier(start, self.frontiers, self.gridmap_processed)
                if self.path is not None:
                    print(time.strftime("%H:%M:%S"),"New shortest path found!")
                    self.path_simple = self.explor.simplify_path(self.gridmap_processed, self.path)
                else:
                    self.path_simple = None
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
                                self.frontier = frontier # highest heuristic + closest
                    else:
                        self.frontier = filtered_lst[0][0]
                    self.path = self.explor.a_star(self.gridmap_processed, start, self.frontier)
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
        print(time.strftime("%H:%M:%S"),"Planning thread terminated successfully!")


 
    def trajectory_following(self):
        """
        Assigns new goals to the robot when the previous one is reached
        """ 
        time.sleep(3*THREAD_SLEEP) #wait for plan init
        while not self.stop: 
            time.sleep(THREAD_SLEEP)
            if self.path is None: # collision - new route
                self.robot.stop() #stop robot
                continue          
            if self.robot.navigation_goal is None and self.path_simple is not None:
                if len(self.path_simple.poses)==0: #reached the goal
                    self.robot.stop() #stop robot
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
    while not expl.stop:
        plt.pause(THREAD_SLEEP)
        ax0.cla() #clear the points from the previous iteration
        ax1.cla()
        if expl.gridmap.data is not None and expl.gridmap_processed.data is not None: #grid
            expl.gridmap.plot(ax0)
            expl.gridmap_processed.plot(ax1)
        if expl.robot.odometry_.pose is not None: #robot
            ax0.scatter(expl.robot.odometry_.pose.position.x, expl.robot.odometry_.pose.position.y,c='black', s=150, marker='x')
            ax1.scatter(expl.robot.odometry_.pose.position.x, expl.robot.odometry_.pose.position.y,c='black', s=150, marker='x')
        if expl.path is not None and expl.path_simple is not None: #path
            expl.path.plot(ax0, style = 'point')
            expl.path.plot(ax1, style = 'point')
        if expl.nav_goal is not None: #frontier
            ax0.scatter(expl.nav_goal.position.x, expl.nav_goal.position.y,c='red', s=150, marker='x')
            ax1.scatter(expl.nav_goal.position.x, expl.nav_goal.position.y,c='red', s=150, marker='x')

        for frontier_tuple in expl.frontiers: #frontiers
            frontier = frontier_tuple[0]
            ax0.scatter(frontier.position.x, frontier.position.y,c='red')
            ax1.scatter(frontier.position.x, frontier.position.y,c='red')
        plt.show()
    expl.__del__() #turn it off
        