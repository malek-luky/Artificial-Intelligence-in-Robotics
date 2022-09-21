#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

#import messages
from messages import *

DELTA_DISTANCE = 0.12
C_TURNING_SPEED = 5
C_AVOID_SPEED = 10

class HexapodController:
    def __init__(self):
        pass


    def goto(self, goal, odometry, collision):
        """Method to steer the robot towards the goal position given its current 
           odometry and collision status
        Args:
            goal: Pose of the robot goal
            odometry: Perceived odometry of the robot
            collision: bool of the robot collision status
        Returns:
            cmd: Twist steering command
        """
        # zero velocity steering command
        cmd_msg = Twist()
        
        """
        gx, gy = goal.x, goal.y
        cx, cy = odometry.position.x, odometry.position.y
        target_to_goal = np.linalg.norm(np.array([gx,cx,gy-cy])) #((gx-cx)**2-(gy-cy)**2)**(1/2)
        is_in_goal = target_to_goal <DELTA_DISTANCE
        """

        # START OF OUR CODE
        if collision:
            return None
        if (goal is not None) and (odometry is not None):
            diff = goal.position-odometry.pose.position
            target_to_goal = (diff).norm()
            is_in_goal = target_to_goal <DELTA_DISTANCE
            if is_in_goal:
                return None
            targ_h = np.arctan2(diff.y,diff.x)
            cur_h = odometry.pose.orientation.to_Euler()[0]
            diff_h = targ_h - cur_h
            diff_h = (diff_h + math.pi) % (2*math.pi) - math.pi
            cmd_msg.linear.x = target_to_goal
            cmd_msg.angular.z = C_TURNING_SPEED*diff_h
            print("Dst to target:",target_to_goal)
        return cmd_msg     
        # END OF OUR CODE       


    def goto_reactive(self, goal, odometry, collision, laser_scan):
        """Method to steer the robot towards the goal position while avoiding 
           contact with the obstacles given its current odometry, collision 
           status and laser scan data
        Args:
            goal: Pose of the robot goal
            odometry: Perceived odometry of the robot
            collision: bool of the robot collision status
            laser_scan: LaserScan data perceived by the robot
        Returns:
            cmd: Twist steering command
        """
        #zero velocity steering command
        cmd_msg = Twist()

        #TODO: if input data are valid, calculate the steering command
        
        return cmd_msg
