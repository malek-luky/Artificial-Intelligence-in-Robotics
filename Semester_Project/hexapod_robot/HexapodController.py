"""
Project: Artificial Intelligence for Robotics
Author: Lukas Malek
Email: malek.luky@gmail.com
Date: February 2023
"""

import numpy as np
from messages import *
import sys
sys.path.append('../')
sys.path.append('hexapod_robot')
import HexapodRobotConst as Constants

class HexapodController:
    def __init__(self):
        pass

    def goto(self, goal, odometry, collision):
        """
        Method to steer the robot towards the goal position given its current 
        odometry and collision status
        
        Args:
            goal: Pose of the robot goal
            odometry: Perceived odometry of the robot
            collision: bool of the robot collision status
        Returns:
            cmd: Twist steering command
        Notes:
            gx, gy = goal.x, goal.y
            cx, cy = odometry.position.x, odometry.position.y
            dst_to_target = np.linalg.norm(np.array([gx,cx,gy-cy])) #((gx-cx)**2-(gy-cy)**2)**(1/2)
            is_in_goal = dst_to_target <Constants.DELTA_DISTANCE
        """
        cmd_msg = Twist() # zero velocity steering command
        if collision:
            cmd_msg.linear.x = 0
            cmd_msg.angular.z = 0
            return None
        if (goal is not None) and (odometry is not None):
            diff = goal.position-odometry.pose.position
            dst_to_target = (diff).norm()
            is_in_goal = dst_to_target <Constants.DELTA_DISTANCE
            if is_in_goal:
                return None
            targ_h = np.arctan2(diff.y,diff.x)
            cur_h = odometry.pose.orientation.to_Euler()[0]
            diff_h = targ_h - cur_h
            diff_h = (diff_h + math.pi) % (2*math.pi) - math.pi
            if abs(diff_h) > np.pi/6:
                cmd_msg.linear.x = 0
                cmd_msg.angular.z = 10*Constants.C_TURNING_SPEED*diff_h
            else:
                cmd_msg.linear.x = dst_to_target
                cmd_msg.angular.z = Constants.C_TURNING_SPEED*diff_h
        return cmd_msg     

    def stop(self):
        cmd_msg = Twist()
        cmd_msg.linear.x = 0
        cmd_msg.angular.z = 0