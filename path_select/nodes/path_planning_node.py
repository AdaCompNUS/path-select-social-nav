#!/usr/bin/env python
import rospy
import numpy as np
from PIL import Image as PILImage
from enum import Enum
import requests
from types import SimpleNamespace
import cv2

from function_modules.costmap_generate import CostMapping
from function_modules.path_generate_select import PathPlanner, GPT_Query, Qwen_Query

from function_modules.utils import convert_to_homo_matrix, points2ref_frame, points_lidar_to_camera
from function_modules.utils import draw_paths_and_labels, show_paths_inMap

from std_msgs.msg import String
from path_select.srv import PlanningService, PlanningServiceResponse
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


extrinsics = np.array(rospy.get_param("extrinsics"))
camera_intrinsics = np.array(rospy.get_param("camera_intrinsics"))
camera_distortion = np.array(rospy.get_param("camera_distortion"))

image_size = tuple(rospy.get_param("image_size"))

# instantiate only for meter->map coordinates transform function
filter_range = rospy.get_param("filter_range")
axis_specs = rospy.get_param("axis_specs")
cost_mapper = CostMapping(filter_range=filter_range, axis_specs=axis_specs, resolution=0.15, spread_radius=1)

anchor_cfgs = rospy.get_param("anchor_cfgs")
path_pseudo_height = rospy.get_param("path_pseudo_height")
path_planner = PathPlanner()

qwen_client = Qwen_Query(server_url="https://your-service-website")
prompt_temp = "Choose one of the {num_of_path} paths to align with social norms. Each projected path is shown as a red curve on the RGB image, labeled by number. " \
"Answer format: {{'Descriptions for each path':<text>, 'Chosen Path':<number>, 'Analysis and reason to choose':<text>, 'Second choice and reason':<text>}}\n"

## IF: you cannot setup qwen model, use GPT, but it is much slower
# no prompt about depth here
gpt_client = GPT_Query()
prompt_gpt_temp = "Assume you are navigating a robot toward the goal. An RGB image from the robot's frontal camera is provided.  " \
     "Choose one of the {num_of_path} path to follow common social conventions that most people obey daily. " \
     "- Analyze each path along the red line and its labelled number, considering the current social context and human behaviors.\n " \
     "- Various norms may apply, but always prioritize the most essential ones.\n" \
     "- Double-check whether the selected path will interrupt pedestrians' activities near the path or enter an improper area.\n" \
     "Answer format: {{'Descriptions for each path':<text>, 'Chosen Path':<number>, 'Analysis and reason to choose':<text>}} with top-1 choice." \
     "Restrict descriptions for each path and the final reason in one sentence."





def convert_state_matrix_to_namespace(state_mat: np.ndarray):
    state_dict = {
        "position": state_mat[0, :3],
        "orientation": state_mat[1, :],
        "linear_velocity": state_mat[2, :3],
        "angular_velocity": state_mat[3, :3],
    }
    return SimpleNamespace(**state_dict)


def lidar_to_pixel(points_inLidar: np.ndarray, lidar_extrinsics: np.ndarray, image_size) -> np.ndarray:
    points_inCam = points_lidar_to_camera(points_inLidar, lidar_extrinsics)

    points_3d = points_inCam.reshape(-1, 1, 3).astype(np.float32)
    # NOTE: change the projection here if you are not using fisheye camera
    points_2d, _ = cv2.fisheye.projectPoints(points_3d, np.zeros(3), np.zeros(3), camera_intrinsics, camera_distortion[:, :4])
    points_2d = points_2d.reshape(-1, 2)

    valid_mask = (0 <= points_2d[:, 0]) & (points_2d[:, 0] < image_size[0]) & \
                 (0 <= points_2d[:, 1]) & (points_2d[:, 1] < image_size[1])

    points_2d = points_2d[valid_mask, :]

    points_inImg = np.hstack((points_2d, points_inCam[:, 2][valid_mask].reshape(-1, 1)))
    # shape: N * 3, (width, height, depth)

    return points_inImg


def update_path_with_ego(planned_path_origin, ego):
    path_arr = np.array(planned_path_origin)
    row_values = path_arr[:, 0]

    mask = row_values > ego[0]
    filtered_path = path_arr[mask]

    updated_path = list(map(tuple, filtered_path))
    return updated_path



class PathPlanningNodeStatus(str, Enum):
    idle = 'idle'
    finish = 'finish'
    busy = 'busy'


class PathPlanningNode:
    def __init__(self):
        rospy.init_node("path_planning_node")
        ns = rospy.resolve_name("path_planning")

        self.pub_planner_state = rospy.Publisher(f"{ns}/planning_node_state", String, queue_size=1)
        self.pub_path_img = rospy.Publisher(f"{ns}/path_image", Image, queue_size=1)
        self.pub_retrieve_map = rospy.Publisher(f"{ns}/retrieve_map", Image, queue_size=1)

        self.plan_service = rospy.Service("/execute_planning", PlanningService, self.plan_handler)
        self.retrieve_service = rospy.Service("/retrieve_path", PlanningService, self.retrieve_handler)

        self.current_state = PathPlanningNodeStatus.idle

        self.path_candidates = None
        self.planned_path = None          # resampled + truncated path, last waypoint is the goal for controller
        self.planned_path_origin = None   # only has {start, anchor, goal}

        self.bridge = CvBridge()
        self.path_image = None
        self.retrieve_map = None

        self.agent_states_dict = None
        self.cost_map_dict = None

        rospy.Timer(rospy.Duration(0.1), self.publish_state)
        rospy.Timer(rospy.Duration(0.1), self.publish_path_img)
        rospy.Timer(rospy.Duration(0.1), self.publish_retrieve_map)

    def publish_state(self, event):
        self.pub_planner_state.publish(self.current_state.value)
    
    def publish_path_img(self, event):
        if self.path_image is not None:
            self.pub_path_img.publish(self.bridge.cv2_to_imgmsg(self.path_image, encoding="bgr8"))
    
    def publish_retrieve_map(self, event):
        if self.retrieve_map is not None:
            self.pub_retrieve_map.publish(self.bridge.cv2_to_imgmsg(self.retrieve_map, encoding="bgr8"))
    

    def plan_handler(self, req):
        self.agent_states_dict = self.preprocess_agent_states_request(req.agent_states)
        self.cost_map_dict = self.preprocess_cost_map_request(req.cost_map)

        if self.current_state != PathPlanningNodeStatus.idle:
            return PlanningServiceResponse(path=[], success=False, message="Node status is not idle, cannot start the next planning.")
        
        self.current_state = PathPlanningNodeStatus.busy
        return PlanningServiceResponse(path=[], success=True, message="Start path planning.") 

    def retrieve_handler(self, req):
        # self.agent_states_dict = self.preprocess_agent_states_request(req.agent_states)
        self.cost_map_dict = self.preprocess_cost_map_request(req.cost_map)
    
        if self.current_state != PathPlanningNodeStatus.finish:
            return PlanningServiceResponse(path=[], success=False, message="Planning is still ongoing, cannot retrieve path.")
        if self.planned_path is None:
            return PlanningServiceResponse(path=[], success=False, message="No path as reference.")
        
        self.planned_path = self.retrieve_path_with_costmap()
        path_inMeter = cost_mapper.transform_map2meter(self.planned_path, pseudo_height=path_pseudo_height).flatten()    # original shape:(num, 3)
        return PlanningServiceResponse(path=path_inMeter, success=True, message="Path retrieved successfully.")
    
    def preprocess_agent_states_request(self, msg):
        as_dict = {}
        d_human_info = np.array(msg.dynamic_human, dtype=np.float32).reshape(-1, 4)
        s_human_info = np.array(msg.static_human, dtype=np.float32).reshape(-1, 2)
        as_dict['hasHuman'] = False if len(d_human_info) + len(s_human_info) < 1 else True
        #as_dict['min_Humandist'] = min(min(d_human_info[:, 0]), min(s_human_info[:, 0]))
        
        # stacked vector for robot state, (4, 4) matrix, row-order is position/orient/linear-vel/angular-vel
        as_dict['robot'] = convert_state_matrix_to_namespace(np.array(msg.robot, dtype=np.float32).reshape(4, 4))
        as_dict['robot_init'] = convert_state_matrix_to_namespace(np.array(msg.robot_ref, dtype=np.float32).reshape(4, 4))
        as_dict['rgb_image'] = np.array(self.bridge.imgmsg_to_cv2(msg.image, desired_encoding='bgr8'))
        return as_dict
    
    def preprocess_cost_map_request(self, msg):
        cm_dict = {}
        row = msg.row
        col = msg.col
        cm_dict['costmap'] = np.array(msg.map_preds, dtype=np.float32).reshape(row, col)
        cm_dict['instant_map'] = np.array(msg.map_instant, dtype=np.int8).reshape(row, col)
        # robot current position & goal
        cm_dict['ego'] = tuple(np.array(msg.ego_inMap, dtype=np.int8))
        cm_dict['goal'] = tuple(np.array(msg.goal_inMap, dtype=np.int8))
        # NOTE: ego itself could be obstacle since robot has size
        cm_dict['costmap'][cm_dict['ego'][0]-2:cm_dict['ego'][0]+3, cm_dict['ego'][1]-2:cm_dict['ego'][1]+3] = 0.
        return cm_dict

        
    def execute_planning(self):
        if not self.agent_states_dict['hasHuman']: # or as_dict['min_Humandist'] > 5.:
            self.generate_direct_to_goal_path()
            pth_resample_trunc = path_planner.resample_and_truncate_path(self.path_candidates[0], resample_num_pts=20, pixel_threshold=4./0.15)
            return pth_resample_trunc
    
        self.generate_path_candidates()
        self.path_image, valid_paths = self.draw_path_image()
        rospy.loginfo(f"Valid Paths: {valid_paths}")
        
        if len(valid_paths) < 1:
            return self.planned_path_origin

        query_image = PILImage.fromarray(self.path_image)

        try:
            num_of_path = len(valid_paths)
            prompt = prompt_temp.format(num_of_path=num_of_path)

            ## in case use qwen-2.5
            response = qwen_client.chat(prompt, query_image)
            vlm_answer = response.choices[0].message.content
            rospy.loginfo(f"VLM query success. Answer: {vlm_answer}")
            path_id = qwen_client.str2path_num(vlm_answer) - 1
            # # hallucinate more paths than having
            if path_id > len(valid_paths) - 1:
                path_id = len(valid_paths) - 1
            self.update_reference_path_origin(new_ref_path=valid_paths[path_id][1])
            return valid_paths[path_id][2]

            
            ## in case use GPT-4o
            # prompt_gpt = prompt_gpt_temp.format(num_of_path=num_of_path)
            # response = gpt_client.chat(prompt_gpt, query_image)
            # if response.status_code == 200:
            #    vlm_answer = response.json()['choices'][0]['message']['content']
            #    rospy.loginfo(f"GPT query success. Answer: {vlm_answer}")
            #    path_id = gpt_client.str2path_num(vlm_answer) - 1
            #    # hallucinate more paths than having
            #    if path_id > len(valid_paths) - 1:
            #        path_id = len(valid_paths) - 1
            #    self.update_reference_path_origin(new_ref_path=valid_paths[path_id][1])
            #    return valid_paths[path_id][2]

            # else:
            #    rospy.loginfo("VLM failed to send response.")
            #    return None
            

        except requests.exceptions.RequestException:
            # planning failed
            rospy.loginfo("Network Errors.")
            return None
    
    def retrieve_path_with_costmap(self):
        if not self.agent_states_dict['hasHuman']: # or as_dict['min_Humandist'] > 5.:
            self.generate_direct_to_goal_path()
            pth_resample_trunc = path_planner.resample_and_truncate_path(self.path_candidates[0], resample_num_pts=20, pixel_threshold=4./0.15)
        
        ## generate multiple paths only if human is in scenario
        else:
            self.generate_path_candidates()
            # update reference's start point with ego position
            self.planned_path_origin = update_path_with_ego(self.planned_path_origin, self.cost_map_dict['ego'])

            # get the planned path
            if len(self.path_candidates) < 1 or self.path_candidates is None:  # if no new path generated
                closest_new_path = self.planned_path_origin
            
            else:
                closest_new_path = path_planner.find_closest_path_to_reference(
                    self.path_candidates, self.planned_path_origin, resample_num_pts=20, trunc_dist_thresh=3./0.15)
            
            pth_resample_trunc = path_planner.resample_and_truncate_path(closest_new_path, resample_num_pts=20, pixel_threshold=5./0.15)

        self.current_state = PathPlanningNodeStatus.idle
        rospy.loginfo("Path retrieve succeeded.")
        # rospy.loginfo(f"Path: {pth_resample_trunc}")
        map_img = show_paths_inMap(self.cost_map_dict['costmap'].copy(), [pth_resample_trunc])
        self.retrieve_map = map_img

        # return the path to the client
        return pth_resample_trunc
    

    def update_reference_path_origin(self, new_ref_path):
        self.planned_path_origin = new_ref_path

    def generate_path_candidates(self):
        ego_pos = self.cost_map_dict['ego']
        goal_pos = self.cost_map_dict['goal']
        
        anchors = path_planner.generate_anchors(map_grid=self.cost_map_dict['instant_map'], 
                                                start_point=ego_pos, goal_point=goal_pos,
                                                anchor_cfgs=anchor_cfgs)
        rospy.loginfo("Anchors generation success.")
        rospy.loginfo(f"From start point {ego_pos} to goal {goal_pos}, with anchors: {anchors}.")
        
        paths = path_planner.find_paths_through_anchors(grid=self.cost_map_dict['costmap'], 
                                                        start=ego_pos, goal=goal_pos, 
                                                        anchors=anchors)
        
        # rospy.loginfo(f"Paths: {paths}")
        rospy.loginfo("Path candidates sampling success.")
        self.path_candidates = paths
    

    def generate_direct_to_goal_path(self):
        ## if no person detected, direct heading to goal position
        ego_pos = self.cost_map_dict['ego']
        goal_pos = self.cost_map_dict['goal']

        paths = path_planner.find_paths_through_anchors(grid=self.cost_map_dict['costmap'], 
                                                        start=ego_pos, goal=goal_pos, 
                                                        anchors=[goal_pos])  # anchors should be list
        rospy.loginfo("No person detected. Direct to-goal path generated.")
        # rospy.loginfo(f"From start point {ego_pos} to goal {goal_pos}, paths: {paths}.")
        self.path_candidates = paths
    
    
    def draw_path_image(self):
        clustered_paths, clustered_resample_paths = path_planner.multiple_paths_clustering(
            self.path_candidates, cluster_threshold=8, resample_num_pts=20, trunc_dist_thresh=4./0.15)
        
        _homo_init = convert_to_homo_matrix(self.agent_states_dict['robot_init'].position, self.agent_states_dict['robot_init'].orientation)
        _homo_ego = convert_to_homo_matrix(self.agent_states_dict['robot'].position, self.agent_states_dict['robot'].orientation)

        valid_points_2d = []
        valid_paths = []
        for idx in range(len(clustered_resample_paths)):
            pth_resample_trunc = path_planner.truncate_path(clustered_resample_paths[idx], pixel_threshold=4./0.15)
            # map unit to meter
            path_coord = cost_mapper.transform_map2meter(pth_resample_trunc, pseudo_height=path_pseudo_height)
            # world coordinate to ego lidar coordinate
            path_coord_ego = points2ref_frame(_homo_ego, _homo_init, path_coord)
            # ego lidar coordinate to camera to image
            path_coord_ego[:, 2] = path_pseudo_height
            path_coord_ego[0][0], path_coord_ego[0][1] = 0., 0.
            points_2d = lidar_to_pixel(path_coord_ego, extrinsics, image_size)[:, :2]

            if points_2d.shape[0] < 1:
                rospy.loginfo("Out of image range, discard path.")
                continue

            path_tuple = (idx, clustered_paths[idx], pth_resample_trunc)
            rospy.loginfo(f"Path {idx}: {clustered_paths[idx]}")
            valid_paths.append(path_tuple)
            valid_points_2d.append(points_2d)
        
        image_data = self.agent_states_dict['rgb_image'].copy()
        draw_paths_and_labels(valid_points_2d, image_data)
        
        return image_data, valid_paths


    def start(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.current_state == PathPlanningNodeStatus.busy:
                self.planned_path = self.execute_planning()
                self.current_state = PathPlanningNodeStatus.finish
            rate.sleep()



if __name__ == "__main__":
    planning_node = PathPlanningNode()
    planning_node.start()

