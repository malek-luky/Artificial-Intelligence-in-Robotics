#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import threading as thread
 
import numpy as np
import matplotlib.pyplot as plt
 
 
sys.path.append('../')
sys.path.append('hexapod_robot')
sys.path.append('hexapod_explorer')
 
#import hexapod robot and explorer
import HexapodRobot 
import HexapodExplorer
import HexapodRobotConst as Constants
 
#import communication messages
from messages import *

#define sleep time
TRAJECTORY_SLEEP_TIME = 8
MAPPING_SLEEP_TIME = 4
PLANNING_SLEEP_TIME = 10
PLOTTING_SLEEP_TIME = 5
 
class Explorer:
    """ Class to represent an exploration agent
    """
    def __init__(self, robotID = 0):
        
 
        """ VARIABLES
        """
        print(time.strftime("%H:%M:%S"),"Initializing Explorer class...")
        #occupancy grid map of the robot ... possibly extended initialization needed in case of 'm1' assignment
        self.gridmap = OccupancyGrid()
        self.gridmap.resolution = 0.1
        # START OF MY CODE m1
        #TODO: m2
        self.gridmap.width = 100
        self.gridmap.height = 100
        self.gridmap.origin = Pose(Vector3(-5.0,-5.0,0.0), Quaternion(1,0,0,0))
        self.gridmap.data = 0.5*np.ones((self.gridmap.height,self.gridmap.width)) #unknown
        # END OF MY CODE m1
 
        """ Initialize values
        """
        #current frontiers
        self.frontiers = None
 
        #current path
        self.path = None
 
        #stopping condition
        self.stop = False
 
        """Connecting the simulator
        """
        #instantiate the robot
        self.robot = HexapodRobot.HexapodRobot(robotID)
        #...and the explorer used in task t1c-t1e
        self.explor = HexapodExplorer.HexapodExplorer()
 
    def start(self):
        """ method to connect to the simulated robot and start the navigation, localization, mapping and planning
        """
        print(time.strftime("%H:%M:%S"),"Starting threads...")

        #turn on the robot 
        self.robot.turn_on()
 
        #start navigation thread
        self.robot.start_navigation()
 
        #start the mapping thread
        try:
            mapping_thread = thread.Thread(target=self.mapping)
            mapping_thread.start() 
        except:
            print(time.strftime("%H:%M:%S"),"Error: unable to start mapping thread")
            sys.exit(1)
 
        #start planning thread
        try:
            planning_thread = thread.Thread(target=self.planning)
            planning_thread.start() 
        except:
            print(time.strftime("%H:%M:%S"),"Error: unable to start planning thread")
            sys.exit(1)
 
        #start trajectory following
        try:
            traj_follow_thread = thread.Thread(target=self.trajectory_following)
            traj_follow_thread.start() 
        except:
            print(time.strftime("%H:%M:%S"),"Error: unable to start planning thread")
            sys.exit(1)
 
    def __del__(self):
        #turn off the robot
        self.robot.stop_navigation()
        self.robot.turn_off()
 
    def mapping(self):
        
        """ Mapping thread for fusing the laser scans into the grid map
        """
        # START OF MY CODE necessary init
        # fuse the laser scan   
        while not self.stop:
            print(time.strftime("%H:%M:%S"),"Mapping...")
            self.gridmap = self.explor.fuse_laser_scan(self.gridmap, self.robot.laser_scan_, self.robot.odometry_)
            self.gridmap.data = self.gridmap.data.reshape(self.gridmap.height, self.gridmap.width)
            
            #### OBSTACLE GROWING #### #p1
            self.gridmap_processed = self.explor.grow_obstacles(self.gridmap, Constants.ROBOT_SIZE)
            
            time.sleep(MAPPING_SLEEP_TIME)
        # END OF MY CODE necessary init

    def planning(self):
        time.sleep(MAPPING_SLEEP_TIME*1+1) #wait for map init
        """ Planning thread that takes the constructed gridmap, find frontiers, and select the next goal with the navigation path  
        """
        # START OF MY CODE f1 + p1
        self.goal_reached = True
        while not self.stop:
            
            
 
            #### PATH PLANNING ####
            #p1 - select the closest frontier
            #wait until the previous goal is reached
            print("planning",self.goal_reached)
            if self.goal_reached:
                #### FRONTIERS ####
                #f1 - select the frontiers
                self.frontiers = self.explor.find_free_edge_frontiers(self.gridmap)
                if len(self.frontiers) !=0 and self.robot.odometry_ is not None:
                    self.path_final = None
                    shortest_path = np.inf
                    self.start = self.robot.odometry_.pose
                    for frontier in self.frontiers:
                        self.path = self.explor.plan_path(self.gridmap_processed, self.start, frontier)
                        #path simplification
                        self.path_simple = self.explor.simplify_path(self.gridmap_processed, self.path)
                        if self.path_simple is not None and len(self.path_simple.poses) < shortest_path:
                            self.path_final = self.path_simple
                            shortest_path = len(self.path_simple.poses)
                    if self.path_final is not None:
                        print(time.strftime("%H:%M:%S"),"New shortest path found!")
                        self.goal_reached = False
                    else:
                        print(time.strftime("%H:%M:%S"),"No new path without collision!")
                    # if self.path_final is not None:
                    #     self.explor.print_path(self.path_final)
                else:
                    print(time.strftime("%H:%M:%S"),"No frontiers found")
            elif self.goal_reached == False:
                print(time.strftime("%H:%M:%S"),"Reaching previous goal")
            
            time.sleep(PLANNING_SLEEP_TIME)
        # END OF MY CODE f1 + p1
 
    def trajectory_following(self):
        time.sleep(MAPPING_SLEEP_TIME*1+PLANNING_SLEEP_TIME+1) #wait for plan init
        """trajectory following thread that assigns new goals to the robot navigation thread
        """ 
        # START OF MY CODE necessary init
        while not self.stop:             
            print("trajectory",self.goal_reached)   
            if self.goal_reached == False and len(self.path_final.poses)==0:
                print("changed to true")
                self.goal_reached = True
                time.sleep(PLANNING_SLEEP_TIME) #wait for a new route
            if self.robot.navigation_goal is None and self.path_final is not None and len(self.path_final.poses) != 0:
                nav_goal = self.path_final.poses.pop(0)
                print(time.strftime("%H:%M:%S"),"Goto: ", nav_goal.position.x, nav_goal.position.y)
                self.robot.goto(nav_goal)
            elif self.robot.navigation_goal is not None:
                odom = self.robot.odometry_.pose
                odom.position.z = 0 
                dist = nav_goal.dist(odom)
                print(time.strftime("%H:%M:%S"),"Vzdalenost: ", dist)
            time.sleep(TRAJECTORY_SLEEP_TIME)
        # END OF MY CODE necessary init
 
 
if __name__ == "__main__":
    #instantiate the robot
    ex0 = Explorer()
    #start the threads (locomotion)
    ex0.start()
 
    #continuously plot the map, targets and plan (once per second)
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(5,10))
    plt.ion()
    time.sleep(MAPPING_SLEEP_TIME*1+PLANNING_SLEEP_TIME+TRAJECTORY_SLEEP_TIME+1) #wait for everything to init
    ax0.set_xlabel('x[m]')
    ax0.set_ylabel('y[m]')
    ax1.set_xlabel('x[m]')
    ax1.set_ylabel('y[m]')
    ax0.set_aspect('equal', 'box')
    ax1.set_aspect('equal', 'box')
    while(1):
        ax0.cla()
        ax1.cla()
        #plot the gridmap
        if ex0.gridmap.data is not None:
            ex0.gridmap.plot(ax0)
        if ex0.gridmap_processed.data is not None:
            ex0.gridmap_processed.plot(ax1)
        #plot the navigation path
        if ex0.path_final is not None:
            ex0.path_final.plot(ax0)
            ex0.path_final.plot(ax1)
        for frontier in ex0.frontiers:
            ax0.scatter(frontier.position.x, frontier.position.y)
            ax1.scatter(frontier.position.x, frontier.position.y)
        plt.show()
        plt.pause(PLOTTING_SLEEP_TIME)#pause for 1s
        #time.sleep(2)
        print(time.strftime("%H:%M:%S"),"Plotting...")