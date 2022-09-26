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
        dst_to_target = np.linalg.norm(np.array([gx,cx,gy-cy])) #((gx-cx)**2-(gy-cy)**2)**(1/2)
        is_in_goal = dst_to_target <DELTA_DISTANCE
        """

        # START OF OUR CODE WEEK 1
        if collision:
            return None
        if (goal is not None) and (odometry is not None):
            diff = goal.position-odometry.pose.position
            dst_to_target = (diff).norm()
            is_in_goal = dst_to_target <DELTA_DISTANCE
            if is_in_goal:
                return None
            targ_h = np.arctan2(diff.y,diff.x)
            cur_h = odometry.pose.orientation.to_Euler()[0]
            diff_h = targ_h - cur_h
            diff_h = (diff_h + math.pi) % (2*math.pi) - math.pi
            cmd_msg.linear.x = dst_to_target
            cmd_msg.angular.z = C_TURNING_SPEED*diff_h
            #print("\n\033[F\033[F\033[F")
            print("Dst to target: ",dst_to_target)#, "\nGoal Position: ", goal.position[0], goal.position[1], goal.position[2], "\nCurrent Position: ", odometry.pose.position[0], odometry.pose.position[1], odometry.pose.position[2])
        return cmd_msg     
        # END OF OUR CODE WEEK 1    


    def goto_reactive(self, goal, odometry, collision, laser_scan):
        """Method to steer the robot towards the goal position while avoiding 
           contact with the obstacles given its current odometry, collision 
           status and laser scan data
        Args:
            goal: Pose of the robot goal
            odometry: Perceived odometry of the robot
            collision: bool of the robot collision status
            laser_scan: LaserScan data perceived by the robot
                Attributes
                header - Header - timestamp and reference frame id of the Odometry message
                angle_min - float - start angle of the scan [rad]
                angle_max - float - end angle of the scan [rad]
                angle_increment - float - angular distance between measurements [rad]
                range_min - float - minimum measurable distance [m]
                range_max - float - maximum measurable distance [m]
                distances - float[] - distance data [m] (note: distance data out of range should be discarded) 
        Returns:
            cmd: Twist steering command
        """
        #zero velocity steering command
        cmd_msg = Twist()

        # START OF OUR CODE WEEK 2
        if laser_scan is not None:
            cmd_msg = self.goto(goal=goal, odometry = odometry, collision = collision)
            if cmd_msg is None:
                return cmd_msg
            length = len(laser_scan.distances)
            left_points = laser_scan.distances[:(length//2)] # Seperate the points into left and right
            right_points = laser_scan.distances[(length//2):]
            left_points = [left_points[i] for i in range(len(left_points)) if laser_scan.range_min < left_points[i] < laser_scan.range_max]
            right_points = [right_points[i] for i in range(len(right_points)) if laser_scan.range_min < right_points[i] < laser_scan.range_max]
            min_left = min(left_points) if len(left_points) > 0 else laser_scan.range_max
            min_right = min(right_points) if len(right_points) > 0 else laser_scan.range_max
            repulsive_force = 1/min_left - 1/min_right
            cmd_msg.angular.z += C_AVOID_SPEED*repulsive_force
            #print("\n\n\n\n\033[F\033[F\033[F\033[F\033[F\033[F\033[F\033[F\033[F") # Jump 5 lines up
            print("Number of meassured data: ", length, "\nLeft_min_value: ", min_left, "\nRight_min_value: ", min_right, "\nRepulsive_force: ", repulsive_force)
            
        return cmd_msg
        # END OF OUR CODE WEEK 2
