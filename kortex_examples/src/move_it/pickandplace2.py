#!/usr/bin/env python3

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, SRI International
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of SRI International nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Acorn Pooley, Mike Lautman

# Inspired from http://docs.ros.org/kinetic/api/moveit_tutorials/html/doc/move_group_python_interface/move_group_python_interface_tutorial.html
# Modified by Alexandre Vannobel to test the FollowJointTrajectory Action Server for the Kinova Gen3 robot

# To run this node in a given namespace with rosrun (for example 'my_gen3'), start a Kortex driver and then run : 
# rosrun kortex_examples example_move_it_trajectories.py __ns:=my_gen3

import sys
import time
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
from math import pi
from std_srvs.srv import Empty

sys.path.insert(0, '/home/heisenkong/catkin_workspace/src/ros_kortex-noetic-devel/cv_module/src')
import shape_from_hand

class pickandplace(object):
    """pickandplace"""
    def __init__(self):

        # Initialize the node
        super(pickandplace, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('pickandplace')

        try:
            self.is_gripper_present = rospy.get_param(rospy.get_namespace() + "is_gripper_present", False)
            if self.is_gripper_present:
                gripper_joint_names = rospy.get_param(rospy.get_namespace() + "gripper_joint_names", [])
                self.gripper_joint_name = gripper_joint_names[0]
            else:
                self.gripper_joint_name = ""
            self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)

            # Create the MoveItInterface necessary objects
            arm_group_name = "arm"
            self.robot = moveit_commander.RobotCommander("robot_description")
            self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
            self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())
            self.display_trajectory_publisher = rospy.Publisher(rospy.get_namespace() + 'move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)

            if self.is_gripper_present:
                gripper_group_name = "gripper"
                self.gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, ns=rospy.get_namespace())

            rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())
        except Exception as e:
            print (e)
            self.is_init_success = False
        else:
            self.is_init_success = True


    def reach_named_position(self, target):
        arm_group = self.arm_group
        
        # Going to one of those targets
        rospy.loginfo("Going to named target " + target)
        # Set the target
        arm_group.set_named_target(target)
        # Plan the trajectory
        (success_flag, trajectory_message, planning_time, error_code) = arm_group.plan()
        # Execute the trajectory and block while it's not finished
        return arm_group.execute(trajectory_message, wait=True)

    def reach_joint_angles(self, tolerance):
        arm_group = self.arm_group
        success = True

        # Get the current joint positions
        joint_positions = arm_group.get_current_joint_values()
        rospy.loginfo("Printing current joint positions before movement :")
        for p in joint_positions: rospy.loginfo(p)

        # Set the goal joint tolerance
        self.arm_group.set_goal_joint_tolerance(tolerance)

        # Set the joint target configuration
        if self.degrees_of_freedom == 7:
            joint_positions[0] = pi/2
            joint_positions[1] = 0
            joint_positions[2] = pi/4
            joint_positions[3] = -pi/4
            joint_positions[4] = 0
            joint_positions[5] = pi/2
            joint_positions[6] = 0.2
        elif self.degrees_of_freedom == 6:
            joint_positions[0] = 0
            joint_positions[1] = 0
            joint_positions[2] = pi/2
            joint_positions[3] = pi/4
            joint_positions[4] = 0
            joint_positions[5] = pi/2
            arm_group.set_joint_value_target(joint_positions)
            
        # Plan and execute in one command
        success &= arm_group.go(wait=True)

        # Show joint positions after movement
        new_joint_positions = arm_group.get_current_joint_values()
        rospy.loginfo("Printing current joint positions after movement :")
        for p in new_joint_positions: rospy.loginfo(p)
        return success

    def reach_joint_angles_custom(self, tolerance):
        arm_group = self.arm_group
        success = True

        # Get the current joint positions
        joint_positions = arm_group.get_current_joint_values()
        rospy.loginfo("Printing current joint positions before movement :")
        for p in joint_positions: rospy.loginfo(p)

        # Set the goal joint tolerance
        self.arm_group.set_goal_joint_tolerance(tolerance)

        # Set the joint target configuration
        if self.degrees_of_freedom == 7:
            joint_positions[0] = 0
            joint_positions[1] = pi/2
            joint_positions[2] = 0
            joint_positions[3] = 0
            joint_positions[4] = 0
            joint_positions[5] = 0
            joint_positions[6] = 0.2
        elif self.degrees_of_freedom == 6:
            joint_positions[0] = 0
            joint_positions[1] = pi/2
            joint_positions[2] = pi/4
            joint_positions[3] = pi/4
            joint_positions[4] = pi
            joint_positions[5] = pi/2
        arm_group.set_joint_value_target(joint_positions)
        
        # Plan and execute in one command
        success &= arm_group.go(wait=True)

        # Show joint positions after movement
        new_joint_positions = arm_group.get_current_joint_values()
        rospy.loginfo("Printing current joint positions after movement :")
        for p in new_joint_positions: rospy.loginfo(p)
        return success
    
    

    def reach_joint_angles_gripperfacedown(self, tolerance):
        arm_group = self.arm_group
        success = True

        # Get the current joint positions
        joint_positions = arm_group.get_current_joint_values()
        rospy.loginfo("Printing current joint positions before movement :")
        for p in joint_positions: rospy.loginfo(p)

        # Set the goal joint tolerance
        self.arm_group.set_goal_joint_tolerance(tolerance)

        # Set the joint target configuration
        if self.degrees_of_freedom == 7:
            joint_positions[5] = -pi/5.5
            
        elif self.degrees_of_freedom == 6:
            joint_positions[0] = 0
            joint_positions[1] = pi/2
            joint_positions[2] = pi/4
            joint_positions[3] = pi/4
            joint_positions[4] = pi
            joint_positions[5] = pi/2
        arm_group.set_joint_value_target(joint_positions)
        
        # Plan and execute in one command
        success &= arm_group.go(wait=True)

        # Show joint positions after movement
        new_joint_positions = arm_group.get_current_joint_values()
        rospy.loginfo("Printing current joint positions after movement :")
        for p in new_joint_positions: rospy.loginfo(p)
        return success

    def reach_joint_angles_orientation(self, tolerance, angle):
        arm_group = self.arm_group
        success = True

        # Get the current joint positions
        joint_positions = arm_group.get_current_joint_values()
        rospy.loginfo("Printing current joint positions before movement :")
        for p in joint_positions: rospy.loginfo(p)

        # Set the goal joint tolerance
        self.arm_group.set_goal_joint_tolerance(tolerance)

        # Set the joint target configuration
        if self.degrees_of_freedom == 7:
            joint_positions[6] += angle
            
        elif self.degrees_of_freedom == 6:
            joint_positions[0] = 0
            joint_positions[1] = pi/2
            joint_positions[2] = pi/4
            joint_positions[3] = pi/4
            joint_positions[4] = pi
            joint_positions[5] = pi/2
        arm_group.set_joint_value_target(joint_positions)
        
        # Plan and execute in one command
        success &= arm_group.go(wait=True)

        # Show joint positions after movement
        new_joint_positions = arm_group.get_current_joint_values()
        rospy.loginfo("Printing current joint positions after movement :")
        for p in new_joint_positions: rospy.loginfo(p)
        return success

    def reach_joint_angles_unorientation(self, tolerance, angle):
        arm_group = self.arm_group
        success = True

        # Get the current joint positions
        joint_positions = arm_group.get_current_joint_values()
        rospy.loginfo("Printing current joint positions before movement :")
        for p in joint_positions: rospy.loginfo(p)

        # Set the goal joint tolerance
        self.arm_group.set_goal_joint_tolerance(tolerance)

        # Set the joint target configuration
        if self.degrees_of_freedom == 7:
            joint_positions[6] -= angle
            
        elif self.degrees_of_freedom == 6:
            joint_positions[0] = 0
            joint_positions[1] = pi/2
            joint_positions[2] = pi/4
            joint_positions[3] = pi/4
            joint_positions[4] = pi
            joint_positions[5] = pi/2
        arm_group.set_joint_value_target(joint_positions)
        
        # Plan and execute in one command
        success &= arm_group.go(wait=True)

        # Show joint positions after movement
        new_joint_positions = arm_group.get_current_joint_values()
        rospy.loginfo("Printing current joint positions after movement :")
        for p in new_joint_positions: rospy.loginfo(p)
        return success

    def get_cartesian_pose(self):
        arm_group = self.arm_group

        # Get the current pose and display it
        pose = arm_group.get_current_pose()
        rospy.loginfo("Actual cartesian pose is : ")
        rospy.loginfo(pose.pose)

        return pose.pose

    def reach_cartesian_pose(self, pose, tolerance, constraints):
        arm_group = self.arm_group
        
        # Set the tolerance
        arm_group.set_goal_position_tolerance(tolerance)

        # Set the trajectory constraint if one is specified
        if constraints is not None:
            arm_group.set_path_constraints(constraints)

        # Get the current Cartesian Position
        arm_group.set_pose_target(pose)

        # Plan and execute
        rospy.loginfo("Planning and going to the Cartesian Pose")
        return arm_group.go(wait=True)

    def reach_gripper_position(self, relative_position):
        gripper_group = self.gripper_group
        
        # We only have to move this joint because all others are mimic!
        gripper_joint = self.robot.get_joint(self.gripper_joint_name)
        gripper_max_absolute_pos = gripper_joint.max_bound()
        gripper_min_absolute_pos = gripper_joint.min_bound()
        try:
            val = gripper_joint.move(relative_position * (gripper_max_absolute_pos - gripper_min_absolute_pos) + gripper_min_absolute_pos, True)
            return val
        except:
            return False 

    def pick(self, pick_position, pick_orientation, success):
    #Pick_position an 3x1 array [x,y,z]
        #opening gripper
        if self.is_gripper_present and success:
            rospy.loginfo("Opening the gripper...")
            success &= self.reach_gripper_position(0.4)
            print (success)
    
        
        #point A
        if self.degrees_of_freedom == 7 and success:
            rospy.loginfo("Reach Cartesian Pose with constraints...")
            # Get actual pose
            actual_pose = self.get_cartesian_pose()
            actual_pose.position.x = pick_position[0]
            actual_pose.position.y = pick_position[1]
            actual_pose.position.z = 0.2
            print("Actual pose: ", actual_pose)
            constraints = None
            # Orientation constraint (we want the end effector to stay the same orientation)
            #constraints = moveit_msgs.msg.Constraints()
            #orientation_constraint = moveit_msgs.msg.OrientationConstraint()
            #constraints.orientation_constraints.append(orientation_constraint)
            #print("Orientation constraints: ", constraints.orientation_constraints)
            # Send the goal
            success &= self.reach_cartesian_pose(pose=actual_pose, tolerance=0.01, constraints=constraints)
            
        #Turn tool to orientation
        if success:
            rospy.loginfo("Turn tool to orientation")
            success &= self.reach_joint_angles_orientation(tolerance=0.01,angle=pick_orientation)
            print(success)
            

        #Go down to surface
        if self.degrees_of_freedom == 7 and success:
            rospy.loginfo("Reach Cartesian Pose with constraints...")
            # Get actual pose
            actual_pose = self.get_cartesian_pose()
            actual_pose.position.z = pick_position[2]
            print("Actual pose: ", actual_pose)
            # Orientation constraint (we want the end effector to stay the same orientation)
            #constraints = moveit_msgs.msg.Constraints()
            #orientation_constraint = moveit_msgs.msg.OrientationConstraint()
            #constraints.orientation_constraints.append(orientation_constraint)
            #print("Orientation constraints: ", constraints.orientation_constraints)
            # Send the goal
            success &= self.reach_cartesian_pose(pose=actual_pose, tolerance=0.01, constraints=None)
           

        #closing gripper
        if self.is_gripper_present and success:
            rospy.loginfo("Closing the gripper 75%...")
            success &= self.reach_gripper_position(0.75)
            print (success)
          

        #Go up to clear board
        if self.degrees_of_freedom == 7 and success:
            rospy.loginfo("Reach Cartesian Pose with constraints...")
            # Get actual pose
            actual_pose = self.get_cartesian_pose()
            actual_pose.position.z = 0.2
            print("Actual pose: ", actual_pose)
            # Orientation constraint (we want the end effector to stay the same orientation)
            #constraints = moveit_msgs.msg.Constraints()
            #orientation_constraint = moveit_msgs.msg.OrientationConstraint()
            #constraints.orientation_constraints.append(orientation_constraint)
            #print("Orientation constraints: ", constraints.orientation_constraints)
            # Send the goal
            success &= self.reach_cartesian_pose(pose=actual_pose, tolerance=0.01, constraints=None)
        
        print("preparing to unturn tool")
        #unturn tool
        if success:
            print("Unturning tool_in")
            rospy.loginfo("UNTurn tool")
            success &= self.reach_joint_angles_unorientation(tolerance=0.01,angle=pick_orientation)
            print(success)
        
        return success

    def place(self, place_position, place_orientation,success):
    #Place_position an 3x1 array [x,y,z]  
        #point B
        if self.degrees_of_freedom == 7 and success:
            rospy.loginfo("Reach Cartesian Pose with constraints...")
            # Get actual pose
            actual_pose = self.get_cartesian_pose()
            actual_pose.position.x = place_position[0]
            actual_pose.position.y = place_position[1]
            print("Actual pose: ", actual_pose)
            constraints = None
            # Send the goal
            success &= self.reach_cartesian_pose(pose=actual_pose, tolerance=0.01, constraints=constraints)

        #Turn tool to orientation
        if success:
            rospy.loginfo("Turn tool to orientation")
            success &= self.reach_joint_angles_orientation(tolerance=0.01,angle=place_orientation)
            print(success)

        #Go down to surface
        if self.degrees_of_freedom == 7 and success:
            rospy.loginfo("Reach Cartesian Pose with constraints...")
            # Get actual pose
            actual_pose = self.get_cartesian_pose()
            actual_pose.position.z = place_position[2]
            print("Actual pose: ", actual_pose)
            constraints = None
            #print("Orientation constraints: ", constraints.orientation_constraints)
            # Send the goal
            success &= self.reach_cartesian_pose(pose=actual_pose, tolerance=0.01, constraints=constraints)
            
        #opening gripper
        if self.is_gripper_present and success:
            rospy.loginfo("Opening the gripper...")
            success &= self.reach_gripper_position(0.4)
            print (success)

        #Go up to clear board
        if self.degrees_of_freedom == 7 and success:
            rospy.loginfo("Reach Cartesian Pose with constraints...")
            # Get actual pose
            actual_pose = self.get_cartesian_pose()
            actual_pose.position.z = 0.2
            print("Actual pose: ", actual_pose)
            # Orientation constraint (we want the end effector to stay the same orientation)
            constraints = moveit_msgs.msg.Constraints()
            orientation_constraint = moveit_msgs.msg.OrientationConstraint()
            constraints.orientation_constraints.append(orientation_constraint)
            #print(constraints.orientation_constraints)
            # Send the goal
            success &= self.reach_cartesian_pose(pose=actual_pose, tolerance=0.01, constraints=constraints)
            
        
        #unturn tool
        if success:
            rospy.loginfo("Turn tool")
            success &= self.reach_joint_angles_unorientation(tolerance=0.01,angle=place_orientation)
            print(success)

        return success


    def pick_and_place(self, pick_position, pick_orientation, place_position, place_orientation, success):
        if success:
            rospy.loginfo('Running Pick function')
            success &= self.pick(pick_position,pick_orientation,success)

        if success:
            rospy.loginfo('Running Place function')
            success &= self.place(place_position, place_orientation, success)
        return success



def main():
    example = pickandplace()

    # For testing purposes
    success = example.is_init_success
    try:
        rospy.delete_param("/kortex_examples_test_results/moveit_general_python")
    except:
        pass

    
    if success:
        rospy.loginfo("Reaching Named Target Home...")
        success &= example.reach_named_position("home")
        print (success)

    if success:
        rospy.loginfo("Facing gripper down")
        success &= example.reach_joint_angles_gripperfacedown(tolerance=0.01)
        print(success)

    pick_positions = np.array([[0.7,-0.15,0.04],[0.7,-0.10,0.04],[0.7,-0.05,0.04],[0.7,0,0.04],[0.7,0.05,0.04],[0.7,0.10,0.4],[0.7,0.15,0.04]])
    pick_orientations = np.array([0,0,0,0,0,0,0])
    

    
    place_positions,place_orientations = shape_from_hand.main()
    place_positions = np.hstack((place_positions, np.full((place_positions.shape[0], 1), 0.04)))
    print("place positions", place_positions)
    print("place orientations", place_orientations)
    #place_positions = np.array(place_positions)
    #place_orientations = np.array(place_orientations)

    #Testing wrapper pick and place function (object 1)
    if success:
        for i in range(min(len(pick_positions),len(place_positions))):
            rospy.loginfo("Pick & Place object no. %d",i+1)
            success &= example.pick_and_place(pick_positions[i],pick_orientations[i], place_positions[i],place_orientations[i],success)
            print(success)
    '''
    if success:
        rospy.loginfo("Pick & Place first object")
        pick_position_1 = np.array([0.7,-0.18,0.03])
        place_position_1 = np.array([0.4,-0.15,0.03])
        pick_orientation = pi/2
        place_orientation = 0
        success &= example.pick_and_place(pick_position_1,pick_orientation, place_position_1,place_orientation,success)
        print(success)
    
    #Testing wrapper pick and place function (object 2)
    if success:
        rospy.loginfo("Pick & Place first object")
        pick_position_1 = np.array([0.7,0,0.03])
        place_position_1 = np.array([0.4,0,0.03])
        pick_orientation = pi/2
        place_orientation = 0
        success &= example.pick_and_place(pick_position_1,pick_orientation, place_position_1,place_orientation,success)
        print(success)
    '''

    if success:
        rospy.loginfo("Reaching Named Target Home...")
        success &= example.reach_named_position("home")
        print (success) 

 
    # For testing purposes
    rospy.set_param("/kortex_examples_test_results/moveit_general_python", success)

    if not success:
        rospy.logerr("The example encountered an error.")

if __name__ == '__main__':
    main()
