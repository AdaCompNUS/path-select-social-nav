#!/usr/bin/env python
import numpy as np
import os
import cv2

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped

from sensor_infos.camera import ImageListener
from sensor_infos.lidar import LidarListener
from sensor_infos.odom import OdomListener, OdomListener_TF

from path_select.msg import AgentStates
from path_select.msg import CostMap

from function_modules.human_detect_track import HumanDetTracker
from function_modules.human_predict import HumanPredictor
from function_modules.costmap_generate import CostMapping
from function_modules.utils import convert_to_homo_matrix, find_nearest_free_point

script_path = os.path.dirname(os.path.realpath(__file__))


use_localization = rospy.get_param("use_localization", False) 

goal_inRef_dict = rospy.get_param("final_goal")
goal_inRef = [goal_inRef_dict['x'], goal_inRef_dict['y']]

camera_intrinsics = np.array(rospy.get_param("camera_intrinsics"))
camera_distortion = np.array(rospy.get_param("camera_distortion"))

extrinsics = np.array(rospy.get_param("extrinsics"))

image_size = tuple(rospy.get_param("image_size"))

# as dicts
detect_cfg = rospy.get_param("detect_cfg")
track_cfg = rospy.get_param("track_cfg")

filter_range = rospy.get_param("filter_range")
axis_specs = rospy.get_param("axis_specs")



def stack_pose_and_velocity(position: np.ndarray, orientation: np.ndarray, linear_vel: np.ndarray, angular_vel: np.ndarray):
    stacked_matrix = np.zeros((4, 4))
    stacked_matrix[0, :3] = position
    stacked_matrix[1, :] = orientation
    stacked_matrix[2, :3] = linear_vel
    stacked_matrix[3, :3] = angular_vel

    return stacked_matrix  # (4, 4) matrix, row-order is position/orient/linear-vel/angular-vel

def rviz_goal_callback(msg):
    global goal_inRef
    goal_inRef = [msg.pose.position.x, msg.pose.position.y]
    rospy.loginfo("Reset goal from RViz")


if __name__ == "__main__":
    rospy.init_node("human_env_info_node")
    ns = rospy.resolve_name("human_env_info")

    image_listener = ImageListener(rgb_image_topic="/usb_cam1/image_raw")
    lidar_listener = LidarListener(lidar_topic="/lidar_points")

    if not use_localization:
        odom_listener = OdomListener(odom_topic="/spot/odometry")
    else:
        odom_listener = OdomListener_TF()
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, rviz_goal_callback)
    
    
    ## NOTE: you can also choose and download larger size yolo model
    human_detect_tracker = HumanDetTracker(
        detection_model_path=os.path.join(script_path, 'function_modules/yolo_model/yolov10b.pt'),
        detection_model_config=detect_cfg,
        byte_tracker_config=track_cfg,
        max_frames=40,
        camera_intrinsics=camera_intrinsics,
        camera_distorsion=camera_distortion,
        lidar_extrinsics=extrinsics)

    human_predictor = HumanPredictor(
        pred_model_path=os.path.join(script_path,'function_modules/nmrf_predict/ckp_prediction.p'),
        hyper_config_path=os.path.join(script_path,'function_modules/nmrf_predict/config.yaml'))

    cost_mapper = CostMapping(
        filter_range=filter_range, 
        axis_specs=axis_specs, 
        resolution=0.15, 
        spread_radius=1)
    

    ## comment if not needed
    pub_annotate_img = rospy.Publisher(f"{ns}/annotated_detection", Image, queue_size=1)
    pub_costmap_img = rospy.Publisher(f"{ns}/colorized_costmap", Image, queue_size=1)

    pub_peds_states = rospy.Publisher(f"{ns}/pedestrian_robot_states", AgentStates, queue_size=1)
    pub_costmap = rospy.Publisher(f"{ns}/costmap", CostMap, queue_size=1)

    cv_bridge = CvBridge()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        frame_id, image_data = image_listener.rgb_time, image_listener.rgb_image
        lidar_data = lidar_listener.pcl_array

        depth_data, lidar_data_inImg = human_detect_tracker.get_depth_from_lidar(lidar_data, axis_specs, filter_range, image_size=image_size)

        if use_localization:
            odom_listener.get_transform(parent_frame="map", child_frame="body")
        
        ego_position, ego_orient = odom_listener.position, odom_listener.orientation

        # detect and track for human/robot states
        _homo_ref = convert_to_homo_matrix(translation=odom_listener.initial_pose[0], rot_vec=odom_listener.initial_pose[1])
        _homo_ego = convert_to_homo_matrix(translation=ego_position, rot_vec=ego_orient)

        human_detect_tracker.process_frame(frame_id=frame_id, frame=image_data, depth_data=depth_data,
                                           ref_homo_trans=_homo_ref,
                                           ego_homo_trans=_homo_ego)
        
        ego_inRef_coords = human_detect_tracker.recent_robot_locs[-1][1]  # shape (1, 3)
        
        human_state_buffer, static_buffer = human_detect_tracker.convert_tracks_to_human_states(
            speed_max=10., window_size=3, std_thresh=4.0, interp_interval=0.1)   # interp_interval represents frequency of human_states
        
        ## comment if don't need annotated image of detections (it will cost long time)
        # annotated_image = human_detect_tracker.annotate_with_detections(image_data)
        # human_detect_tracker.draw_lidar_points_depth(annotated_image, lidar_data_inImg)
        # pub_annotate_img.publish(cv_bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8"))
        

        # obtain human/robot states (positions and velocities)
        msg = AgentStates()
        msg.dynamic_human = np.array(human_state_buffer).flatten()
        msg.static_human = np.array(static_buffer).flatten()
        msg.robot = stack_pose_and_velocity(
            ego_position, ego_orient, odom_listener.linear_velocity, odom_listener.angular_velocity).flatten()
        msg.robot_ref = stack_pose_and_velocity(
            odom_listener.initial_pose[0], odom_listener.initial_pose[1], odom_listener.initial_pose[2], odom_listener.initial_pose[3]).flatten()
        msg.image = cv_bridge.cv2_to_imgmsg(image_data, encoding="bgr8")
        pub_peds_states.publish(msg)

        
        # accumulate for history trajectories
        # NOTICE: downsample_scale is determined by human states frequency (interp_interval) v.s. prediction sequence default frequency (2.5Hz)
        pred_samples = human_predictor.predict_future_trajectory(
            human_detect_tracker.interpolate_online_trajs, human_detect_tracker.robot_trajs, downsample_scale=4)
        
        cost_map_with_preds = cost_mapper.generate_costmap(raw_lidar_data=lidar_data, pred_locs=pred_samples,
                                                            ref_homo_trans=_homo_ref, 
                                                            ego_homo_trans=_homo_ego)
        
        goal_arr = np.array(goal_inRef).reshape(1, -1)
        if use_localization:
            ext_goal = np.append(goal_arr, [0, 1]).reshape(-1, 1)
            # if use localization, goal is absolute coordinate
            goal_arr = np.transpose(np.dot(np.linalg.inv(_homo_ref), ext_goal))[:, :2]
        
        goal_inMap = cost_mapper.transform_meter2map(goal_arr)  # returns tuple of int array

        if cost_map_with_preds[goal_inMap] >= 1:
            rospy.logwarn("Goal is in an obstacle! Searching for nearest free space...")
            new_point = find_nearest_free_point(cost_map_with_preds, goal_inMap[0][0], goal_inMap[1][0], max_radius=20)
            if new_point is not None:
                goal_inMap[0][0], goal_inMap[1][0] = new_point[0], new_point[1]
                rospy.loginfo("Nearest free space found.")
        
        
        map_msg = CostMap()
        map_msg.row = cost_map_with_preds.shape[0]
        map_msg.col = cost_map_with_preds.shape[1]
        map_msg.map_preds = cost_map_with_preds.flatten()
        map_msg.map_instant = cost_mapper.instant_occ_map.flatten()
        # robot's current position / goal in map
        map_msg.ego_inMap = np.array(cost_mapper.transform_meter2map(ego_inRef_coords)).flatten()
        map_msg.goal_inMap = np.array(goal_inMap).flatten()
        pub_costmap.publish(map_msg)


        ## comment if don't need to display costmap
        cost_map_8bit = (cost_map_with_preds * 255).astype(np.uint8)
        cost_map_image = 255 - cv2.applyColorMap(cost_map_8bit, cv2.COLORMAP_HOT)      # in BGR format
        cost_map_image =  cv2.flip(cv2.cvtColor(cost_map_image, cv2.COLOR_BGR2RGB), 0) # make origin in the bottom
        pub_costmap_img.publish(cv_bridge.cv2_to_imgmsg(cost_map_image, encoding="bgr8"))

        rate.sleep()
