#!/usr/bin/env python
import numpy as np
import rospy
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped

from interfaces.controller_interface import ControllerInterface

from function_modules.adapt_orca import AdaptORCA
from function_modules.utils import convert_to_homo_matrix

from function_modules.dwa_recover import DeadlockDetector, DWA_control


use_localization = rospy.get_param("use_localization", False) 

goal_inRef_dict = rospy.get_param("final_goal")
goal_inRef = [goal_inRef_dict['x'], goal_inRef_dict['y']]

controller_config = rospy.get_param("controller_cfgs")
adapt_orca_controller = AdaptORCA(controller_config)

deadlock_detector = DeadlockDetector(window_size=15, min_progress=0.5, velocity_threshold=0.05)
dwa_controller = DWA_control()


def compute_xy_and_heading_inRef(ref_trans_homo, curr_trans_homo):
    """
    Compute the heading (yaw) angle from homogenous matrix, assuming motion in the XY plane.
    
    robot2head_homo: robot's head pose in ego frame
        transform between "robot's head" and "robot's ego", since default moving-forward direction might not
        align with +x-axis in robot's ego frame;
        column-vectors give the coordinates of the new axes in the old frame
    """
    robot2head_homo = np.array([[1., 0., 0., 0.], 
                                [0., 1., 0., 0.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]])
    
    curr_in_ref = np.dot(np.linalg.inv(ref_trans_homo), curr_trans_homo)

    Rot = curr_in_ref[0:3, 0:3]
    x_rel, y_rel, _ = curr_in_ref[:3, 3]

    heading, _, _ = Rotation.from_matrix(Rot).as_euler('zyx', degrees=False)

    # robot's "head" heading direction
    head_heading, _, _ = Rotation.from_matrix(np.dot(curr_in_ref, robot2head_homo)[0:3, 0:3]).as_euler('zyx', degrees=False)

    return x_rel, y_rel, head_heading


class ControllerNode:
    def __init__(self):
        rospy.init_node("controller_node")

        self.controller_interface = ControllerInterface()

        self.pub_controller = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        
        self.goal_inRef = np.array(goal_inRef)

        if use_localization:
            rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.rviz_goal_callback)
    
    def rviz_goal_callback(self, msg):
        # if use localization, goal is absolute coordinate
        self.goal_inRef = np.array([msg.pose.position.x, msg.pose.position.y])
        
        _homo_ref = convert_to_homo_matrix(
            translation=self.controller_interface.robot_init.position, rot_vec=self.controller_interface.robot_init.orientation)
        ext_goal = np.append(self.goal_inRef, [0, 1]).reshape(-1, 1)
        self.goal_inRef = np.transpose(np.dot(np.linalg.inv(_homo_ref), ext_goal))[:, :2]
        rospy.loginfo("Reset goal from RViz for controller")


    def compute_robot_state(self):
        """
        Return:
            robot_state: (5,) array with order of [x, y, theta, v, omega]
        """
        robot_state = np.zeros((5,))

        _homo_ref = convert_to_homo_matrix(
            translation=self.controller_interface.robot_init.position, rot_vec=self.controller_interface.robot_init.orientation)
        _homo_ego = convert_to_homo_matrix(
            translation=self.controller_interface.robot.position, rot_vec=self.controller_interface.robot.orientation)

        robot_state[0], robot_state[1], robot_state[2] = compute_xy_and_heading_inRef(_homo_ref, _homo_ego)

        robot_state[3] = self.controller_interface.robot.linear_velocity[0]   # linear_vel.x
        robot_state[4] = self.controller_interface.robot.angular_velocity[2]  # angular_vel.z
        
        return robot_state
    

    def start(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            path_to_follow = self.controller_interface.request_path()
            
            # update goal with planned path if available
            if path_to_follow is not None:
                adapt_orca_controller.set_robot_goal(path_to_follow[-1, :2])
                rospy.loginfo(f"Path-to-follow as goal updated: {path_to_follow[-1, :2]}")
            
            # # wait until the first goal derived from path_to_follow arrives
            if adapt_orca_controller.robot_goal is None:
                rate.sleep()
                continue

            robot_state = self.compute_robot_state()
            
            if np.linalg.norm(robot_state[:2] - self.goal_inRef) < 0.2:
                rospy.loginfo("Success: very close to the goal. Note as reached.")
                rate.sleep()
                continue

            action = adapt_orca_controller.predict(
                robot_state, 
                self.controller_interface.human_states, 
                self.controller_interface.static_humans)
            
            # convert action tuple (vx, vy) to twist and publish
            linear_vel, angular_vel = adapt_orca_controller.velocity_xy_to_unicycle(action[0], action[1], robot_state[2])
            
            # if orca makes robot keeping rotating (hardly happens):
            if path_to_follow is not None:
                deadlock = deadlock_detector.update(position=robot_state[:2], linear_velocity=linear_vel, subgoal=self.goal_inRef)
                if deadlock:
                    rospy.loginfo("ORCA falls into deadlock, change to DWA.")
                    human_states = None
                    if len(self.controller_interface.human_states) > 0:
                        human_states = np.asarray(self.controller_interface.human_states)[:, :2]
                    if len(self.controller_interface.static_humans) > 0:
                        statics = np.asarray(self.controller_interface.static_humans)[:, :2]
                        human_states = np.concatenate((human_states, statics), axis=0)
                    
                    linear_vel, angular_vel = dwa_controller.predict(robot_state, human_states, path_to_follow[-1, :2])
            
            
            MAX_ANGULAR_VEL = 0.7
            angular_vel = np.clip(angular_vel, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)

            msg = Twist()
            msg.linear.x = linear_vel
            msg.angular.z = angular_vel
            self.pub_controller.publish(msg)

            rate.sleep()


if __name__ == "__main__":
    controller = ControllerNode()
    controller.start()
