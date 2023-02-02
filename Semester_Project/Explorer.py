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
 
class Explorer:
    """ Class to represent an exploration agent
    """
    def __init__(self, robotID = 0):
        print("Init")
 
        """ VARIABLES
        """
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
        #turn on the robot 
        self.robot.turn_on()
 
        #start navigation thread
        self.robot.start_navigation()
 
        #start the mapping thread
        try:
            mapping_thread = thread.Thread(target=self.mapping)
            mapping_thread.start() 
        except:
            print("Error: unable to start mapping thread")
            sys.exit(1)
 
        #start planning thread
        try:
            planning_thread = thread.Thread(target=self.planning)
            planning_thread.start() 
        except:
            print("Error: unable to start planning thread")
            sys.exit(1)
 
        #start trajectory following
        try:
            traj_follow_thread = thread.Thread(target=self.trajectory_following)
            traj_follow_thread.start() 
        except:
            print("Error: unable to start planning thread")
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
            self.gridmap = self.explor.fuse_laser_scan(self.gridmap, self.robot.laser_scan_, self.robot.odometry_)
            self.gridmap.data = self.gridmap.data.reshape(self.gridmap.height, self.gridmap.width)
        # END OF MY CODE necessary init

    def planning(self):
        """ Planning thread that takes the constructed gridmap, find frontiers, and select the next goal with the navigation path  
        """
        # START OF MY CODE f1 + p1
        
        while not self.stop:
            self.path_final = None

            #### OBSTACLE GROWING ####
            #p1
            self.gridmap_processed = self.explor.grow_obstacles(self.gridmap, Constants.ROBOT_SIZE)
            
            #### FRONTIERS ####
            #f1 - select the frontiers
            self.frontiers = self.explor.find_free_edge_frontiers(self.gridmap)
 
            #### PATH PLANNING ####
            #p1 - select the closest frontier
            if len(self.frontiers) !=0 and self.robot.odometry_ is not None:
                shortest_path = np.inf
                self.start = self.robot.odometry_.pose
                for frontier in self.frontiers:
                    self.path = self.explor.plan_path(self.gridmap_processed, self.start, frontier)
                    #path simplification
                    self.path_simple = self.explor.simplify_path(self.gridmap_processed, self.path)
                    if self.path_simple is not None and len(self.path_simple.poses) < shortest_path:
                        self.path_final = self.path_simple
                        shortest_path = len(self.path_simple.poses)
                # if self.path_final is not None:
                #     self.explor.print_path(self.path_final)
            else:
                print("No frontiers found")
        # END OF MY CODE f1 + p1
 
    def trajectory_following(self):
        """trajectory following thread that assigns new goals to the robot navigation thread
        """ 
        # START OF MY CODE necessary init
        while not self.stop:
            if self.path_final is not None and len(self.path_final.poses) != 0:
                print("trajectory_following")
                nav_goal = self.path_final.poses.pop(0)
                print(nav_goal)
                print("Goto: ", nav_goal.position.x, nav_goal.position.y)
                self.robot.goto(nav_goal)
                while self.robot.navigation_goal is not None:
                    odom = self.robot.odometry_.pose
                    odom.position.z = 0 
                    dist = nav_goal.dist(odom)
                    print("Vzdaľenosť: ", dist)
                    time.sleep(1)
                print("Arrived")
        # END OF MY CODE necessary init
 
 
if __name__ == "__main__":
    #instantiate the robot
    ex0 = Explorer()
    #start the threads (locomotion)
    ex0.start()
 
    #continuously plot the map, targets and plan (once per second)
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(10,5))
    plt.ion()
    while(1):
        plt.cla()
        #plot the gridmap
        if ex0.gridmap.data is not None:
            ex0.gridmap.plot(ax0)
        if ex0.gridmap_processed.data is not None:
            ex0.gridmap_processed.plot(ax1)
        #plot the navigation path
        if ex0.path is not None:
            ex0.path.plot(ax0)
            ex0.path.plot(ax1)
        for frontier in ex0.frontiers:
            ax0.scatter(frontier.position.x, frontier.position.y)
            ax1.scatter(frontier.position.x, frontier.position.y)
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        ax0.set_aspect('equal', 'box')
        ax1.set_aspect('equal', 'box')
        plt.show()
        plt.pause(1)#pause for 1s