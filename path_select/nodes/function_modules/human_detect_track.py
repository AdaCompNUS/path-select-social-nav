import numpy as np
from ultralytics import YOLOv10
## clone the repo and directly install with: pip install .
from .byte_track.tracker.byte_tracker import BYTETracker
## remember to install: cython_bbox + lap

from sklearn.cluster import KMeans

import time

import copy
import cv2
from collections import defaultdict
from scipy.interpolate import interp1d

from .utils import points2ref_frame, points_camera_to_lidar, points_lidar_to_camera
from .utils import filter_lidar_data

import warnings
warnings.filterwarnings('ignore')
#warnings.simplefilter("ignore")

class HumanDetTracker:

    def __init__(self, 
                 detection_model_path: str,
                 detection_model_config: dict,
                 byte_tracker_config: dict,
                 max_frames: int,
                 camera_intrinsics: np.ndarray = None,
                 camera_distorsion: np.ndarray = None,
                 lidar_extrinsics: np.ndarray = None):
        """
        Args:
            detection_model_path (str): Path to the YOLOv10 model file.
            byte_tracker_config (dict): Configuration dictionary for BYTETracker.
            max_frames: aximum number of frames to consider, e.g., around 4s

            camera_intrinsics (np.ndarray): 3x3 matrix of camera intrinsics, e.g.:
                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
            lidar_extrinsics (np.ndarray): 4x4 matrix transform lidar points into camera frame: P_cam=Ext*P_lidar.
                column-vectors give the coordinates of the new x/y/z axis in the old frame
        """
        # Load detection model
        self.detection_model = YOLOv10(detection_model_path)

        self.imgsz = detection_model_config['imgsz']  # default: imgsz=640
        self.detect_conf_thresh = detection_model_config['confidence_thresh']

        # Initialize tracker
        self.tracker = BYTETracker(track_thresh=byte_tracker_config['track_thresh'],
                                   track_buffer=byte_tracker_config['track_buffer'], 
                                   match_thresh=byte_tracker_config['match_thresh'])
        '''
        track_thresh: tracking confidence threshold, 0.6
        track_buffer: the frames for keep lost tracks
        match_thresh: matching threshold for tracking
        '''

        self.max_frames = max_frames

        # Calibration matrices
        self.camera_intrinsics = camera_intrinsics
        self.camera_distorsion = camera_distorsion
        self.lidar_extrinsics = lidar_extrinsics

        # Buffer to store the online targets for recent frames
        self.recent_tracks = []
        self.recent_3d_coord_tracks = [] # depth registered

        # Buffer to store the robot ego position for recent frames
        self.recent_robot_locs = []

        # Dict of tuple: (trajectories with shape M*2 in continous frame, indicator for whether velocity included)
        # only for person in CURRENT track's online_id -> trajectories up to current time step 
        self.interpolate_online_trajs = None

        # (T, 2) array for robot history locations
        self.robot_trajs = None


    def detect(self, frame: np.ndarray):

        results = self.detection_model.predict(source=frame, conf=self.detect_conf_thresh, imgsz=self.imgsz, verbose=False)
        detect_outputs = results[0].boxes.data.cpu().numpy()   # shape: N*6, x1,y1,x2,y2,conf,cls

        bboxes_arr, scores_arr = detect_outputs[:, :4], detect_outputs[:, 4]
        person_idx = np.where(detect_outputs[:, -1] == 0)

        if person_idx[0].size == 0:
            return None
        
        bboxes_arr_tlbr_person = bboxes_arr[person_idx][:, [1, 0, 3, 2]]
        scores_arr_person = scores_arr[person_idx]

        detections_person = np.concatenate((bboxes_arr_tlbr_person, scores_arr_person.reshape(-1, 1)), axis=1)

        return detections_person
    

    def track(self, dets: np.ndarray, frame_shape, frame_id, min_box_area=10):
        """        
        Returns:
            track_tuple: A tuple contains frame ID, online targets bboxes, ids and scores.
        """

        h, w = frame_shape[0], frame_shape[1]
        online_targets = self.tracker.update(dets, img_info=(h, w), img_size=(h, w))

        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh     # in byte_tracker, tlbr->tlwh ret[2:] -= ret[:2], w/h is actually reversed
            tid = t.track_id
            horizontal = tlwh[3] / tlwh[2] > 1.2   # tlwh[2] is b-t, tlwh[3] is r-l

            if (tlwh[2] * tlwh[3]) > min_box_area: # and not horizontal:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
        
        track_tuple = (frame_id, online_tlwhs, online_ids, online_scores)
        return track_tuple
    

    def pixel_to_camera(self, pixel_coord: np.ndarray) -> np.ndarray:
        """
        Args:
            pixel_coord: (N, 3) pixel coordinates with depth, in px, py (column, row), depth order.
        
        Returns:
            A (M, 3) np.array for [x, y, z] in camera frame
        """
        xy_coord = np.empty_like(pixel_coord)

        uv_distorted = pixel_coord[:, :2].reshape(-1, 1, 2).astype(np.float32)
        # undistorted = cv2.undistortPoints(uv_distorted, self.camera_intrinsics, self.camera_distorsion)
        # NOTE: change to the commented line if not using fisheye
        undistorted = cv2.fisheye.undistortPoints(uv_distorted, self.camera_intrinsics, self.camera_distorsion[:, :4])

        xy_coord[:, 0] = undistorted.squeeze(1)[:, 0] * pixel_coord[:, 2]
        xy_coord[:, 1] = undistorted.squeeze(1)[:, 1] * pixel_coord[:, 2]
        xy_coord[:, 2] = pixel_coord[:, 2]

        return xy_coord
    

    def camera_to_pixel(self, points_inCam: np.ndarray, image_size) -> np.ndarray:
        """
        Args:
            points_inCam: (N, 3) array in camera coordinates
        Returns:
            A (N, 3) np.array in pixel coordinates, with order (px, py, depth)
        """ 
        
        points_3d = points_inCam.reshape(-1, 1, 3).astype(np.float32)
        # NOTE: change to pin-hole model if not using fisheye
        points_2d, _ = cv2.fisheye.projectPoints(points_3d, np.zeros(3), np.zeros(3), self.camera_intrinsics, self.camera_distorsion[:, :4])
        points_2d = points_2d.reshape(-1, 2)

        valid_mask = (0 <= points_2d[:, 0]) & (points_2d[:, 0] < image_size[0]) & \
                     (0 <= points_2d[:, 1]) & (points_2d[:, 1] < image_size[1])

        points_2d = points_2d[valid_mask, :]

        points_inImg = np.hstack((points_2d, points_inCam[:, 2][valid_mask].reshape(-1, 1)))
        # shape: N * 3, (width, height, depth)

        return points_inImg
    
    

    def get_depth_from_lidar(self, raw_lidar_data: np.ndarray, axis_specs: dict, 
                             filter_range: dict, image_size=(1920, 960)) -> np.ndarray:
        """
        Return:
            depth_data: image shape-like 2D array, at each pixel coordinate is depth
        """
        
        filtered_lidar_data = filter_lidar_data(raw_lidar_data, axis_specs, 
                                                fwd_range=filter_range['fwd'], side_range=filter_range['side'], height_range=filter_range['height'])
        
        points_inCam = points_lidar_to_camera(filtered_lidar_data, self.lidar_extrinsics)
        # contrain negative depth values
        in_front_mask = points_inCam[:, 2] >= filter_range['fwd'][0]
        points_inCam = points_inCam[in_front_mask, :]

        lidar_data_inImg = self.camera_to_pixel(points_inCam, image_size)

        depth_data = np.full((image_size[1], image_size[0]), 0).astype(np.float32)
        depth_data[lidar_data_inImg[:, 1].astype(int), lidar_data_inImg[:, 0].astype(int)] = lidar_data_inImg[:, 2]

        return depth_data, lidar_data_inImg
        

    def get_person_coords_inCam_from_depth(self, depth_data, person_tlwhs, person_ids):
        '''
        In byte_tracker, tlbr->tlwh ret[2:] -= ret[:2], w/h is actually reversed
        tlwh[2] is b-t, tlwh[3] is r-l
        
        Args:
            dapth_data: image-like array, same size with image, but at each coordinate is depth value
            person_tlwhs: list of (4,) arrays for bbox
            person_ids: list of track ids

        Return: A list of MEAN position of all points inside each bounding box (3D),
                A flag to indicate whether the bounding box is valid (enough depth info inside)
        '''
        mask_img = np.zeros_like(depth_data, dtype=np.int32)

        for idx, tlwh in enumerate(person_tlwhs):
            top, left, height, width = tlwh
            bottom, right = top + height, left + width
            mask_img[int(top):int(bottom), int(left):int(right)] = person_ids[idx] + 1   # +1, in case of person id is 0

        person_depth_img = copy.deepcopy(depth_data)
        person_depth_img[mask_img == 0] = 0  # not person region, outside every person bboxes

        person_lidar_depth, person_lidar_points_inImg, valid_lidar_flag = self.get_person_bbox_with_depth(mask_img, person_depth_img, person_ids)

        # convert image coordinates into physical coordinates
        person_inCam_xyz_coords = []
        person_inCam_xyz_points_list = []
        for person_pixel_coord in person_lidar_points_inImg:
            person_camera_coord = self.pixel_to_camera(person_pixel_coord)   # shape: N * 3

            person_inCam_xyz_coords.append(np.mean(person_camera_coord, axis=0))
            person_inCam_xyz_points_list.append(person_camera_coord)

        return person_inCam_xyz_coords, valid_lidar_flag
    

    def get_person_bbox_with_depth(self, person_mask, person_depth_img, person_ids):
        '''
        Check whether points are in polygons defined by person bounding boxes
        Split into 3 clusters: foreground/person/background, and choose the largest cluster
        '''
        lidar_inImg_inbbox = []
        lidar_inCam_depth = []
        valid_lidar_flag = []

        for pid in person_ids:
            person_mask_bbox_id = pid + 1
            
            valid_mask = (person_mask == person_mask_bbox_id) & (person_depth_img != 0)
            bbox_indices = np.argwhere(valid_mask)
            depths = person_depth_img[valid_mask]
            # depth map into coordinates [img_x, img_y, depth]
            in_bbox_points = np.hstack((bbox_indices[:, [1, 0]], depths[:, None]))

            if len(depths) < 3:
                valid_lidar_flag.append(False)
                continue

            kmeans = KMeans(n_clusters=3, max_iter=50)
            kmeans.fit(in_bbox_points[:, 2].reshape(-1, 1))

            labels = kmeans.labels_
            largest_cluster_label = np.bincount(labels).argmax()
            
            points_to_compute = in_bbox_points[labels == largest_cluster_label]

            lidar_inImg_inbbox.append(points_to_compute)
            lidar_inCam_depth.append(np.mean(points_to_compute, axis=0))
            valid_lidar_flag.append(True)
        
        return lidar_inCam_depth, lidar_inImg_inbbox, valid_lidar_flag
    

    def process_frame(self, frame_id, frame: np.ndarray, depth_data: np.ndarray, 
                      ref_homo_trans: np.ndarray, ego_homo_trans: np.ndarray, min_box_area=10):
        """
        Args:
            frame_id: synchronized timestamp of the subscribed image/pcd info
            frame: image for person detection
            depth_data: H*W*1, image-like array, extracted from Lidar data
            min_box_area: minimum bbox area to assign a valid tracklet

        Returns:
            self variables of three lists: 
                recent (up to max_frames) 2D bounding box tracks of persons
                recent (up to max_frames) 3D tracks of persons in reference frame
                recent robot locations
        """
        # 1. detect person for bboxes and scores
        person_dets = self.detect(frame)

        if person_dets is not None:
            # 2. update track results with detections
            # tracked tuple: (frame_id, online_tlwhs, online_ids, online_scores)
            person_tracks = self.track(person_dets, frame.shape, frame_id, min_box_area=min_box_area)
            self.recent_tracks.append(person_tracks)

            # 3. find the depth of each detected person, if has < 3 lidar points inside -> no valid
            person_inCam_coords, valid_flag = self.get_person_coords_inCam_from_depth(depth_data, person_tlwhs=person_tracks[1], person_ids=person_tracks[2])
            person_inCam_coords = np.array(person_inCam_coords)

            # when depth-assigned & detected person number is non-zero
            if person_inCam_coords.shape[0] > 0:
                # 4. transform into reference frame
                person_inLidar_coords = points_camera_to_lidar(person_inCam_coords, self.lidar_extrinsics)

                person_inRef_coords = points2ref_frame(ref_homo_trans, ego_homo_trans, person_inLidar_coords)  # shape (ped_num, 3)

                online_ids = person_tracks[2]
                self.recent_3d_coord_tracks.append((frame_id, person_inRef_coords, np.array(online_ids)[valid_flag]))

                # pop out history frame if total number is larger than buffer
                if len(self.recent_tracks) > self.max_frames:
                    self.recent_tracks.pop(0)
                
                if len(self.recent_3d_coord_tracks) > self.max_frames:  # depth frequency might not be same with image, set max_frames for simplicity
                    self.recent_3d_coord_tracks.pop(0)
        
        # 5. record robot location as well
        ego_inRef_coords = points2ref_frame(ref_homo_trans, ego_homo_trans, np.array([0., 0., 0.]).reshape(1, -1))  # shape (1, 3)
        self.recent_robot_locs.append((frame_id, ego_inRef_coords))

        if len(self.recent_robot_locs) > self.max_frames:
            self.recent_robot_locs.pop(0)
        

    def convert_tracks_to_human_states(self, speed_max=10., window_size=3, std_thresh=4.0, interp_interval=0.1):
        """
        Returns:
            human_state_buffer: A list of (4,) array for current human state (x, y, vx, vy)
            static_buffer: A list of (2,) array for position (x, y), since no velocity obtained. 
                           For person first detected, so there's no history locations
        """
        ## 1. Robot
        ego_inRef_coords = self.recent_robot_locs[-1][1]  # latest frame

        # process to obtain robot trajectory
        fids, ego_locs = zip(*self.recent_robot_locs)
        fids = np.array(fids).astype(float)
        ego_locs = np.stack(ego_locs)  # shape: T, 1, 3 since each ego_pos is a (1, 3) array for x-y-z pose

        if len(ego_locs) < 2:
            self.robot_trajs = ego_locs.squeeze(1)[:, :2]
        else:
            self.robot_trajs = self.interpolate_trajectory_across_frame(ego_locs.squeeze(1)[:, :2], fids, interp_interval)


        ### 2. Human
        # post-process to generate trajectory for each person (identified by track ID)
        traj_dict = self.preprocess_track_buffer(self.recent_3d_coord_tracks)

        filtered_velocities = {}
        filtered_person_trajs = {}

        # empty the online trajectory dict
        self.interpolate_online_trajs = {}

        human_state_buffer = []
        static_buffer = []

        for person_id, frame_locs in traj_dict.items():
            frame_ids, coordinates = zip(*frame_locs)
            frame_ids = np.array(frame_ids).astype(float)
            coordinates = np.array(coordinates)

            # in case have identical frame_id between frames
            _, unique_idx = np.unique(frame_ids, return_index=True)
            unique_idx = np.sort(unique_idx)
            frame_ids = frame_ids[unique_idx]
            coordinates = coordinates[unique_idx]

            time_diff = np.diff(frame_ids).reshape(-1, 1)   # unit is frame
            coord_diff = np.diff(coordinates, axis=0)

            velocity = coord_diff / time_diff   # shape: M*3

            # shape: M*2, (M-1)*2
            filtered_traj, filtered_velocity = self.trajectory_velocity_filtering(coordinates, ego_inRef_coords, velocity[:, :2], 
                                                                                  time_diff, speed_max, window_size, std_thresh)
            
            filtered_velocities[person_id] = filtered_velocity
            filtered_person_trajs[person_id] = np.concatenate((frame_ids.reshape(-1, 1), filtered_traj), axis=1)  # (M, 3) -> fid, x, y
            
            # online id from the latest frame
            online_ids = self.recent_tracks[-1][2]

            if person_id in online_ids:
                if len(filtered_velocity) >= 1:
                    interp_trajs = self.interpolate_trajectory_across_frame(filtered_traj, frame_ids, interp_interval)
                    self.interpolate_online_trajs[person_id] = (interp_trajs, True)

                    human_state_buffer.append(np.append(interp_trajs[-1], np.mean(filtered_velocity[-3:, :], axis=0)))

                else:
                    # only one frame, no velocity -> consider as static obstacles
                    self.interpolate_online_trajs[person_id] = (filtered_traj, False)

                    static_buffer.append(filtered_traj[-1])  # filtered_traj shape: 1*2
        
        return human_state_buffer, static_buffer
    
    
    def preprocess_track_buffer(self, track_buffer):
        trajectories = defaultdict(list)

        for frame_id, coords, track_ids in track_buffer:
            for i, track_id in enumerate(track_ids):
                trajectories[track_id].append((frame_id, coords[i]))

        return trajectories
    

    def trajectory_velocity_filtering(self, coordinates, ego_pos_inWorld, velocity, time_diff, 
                                      speed_max=10., window_size=3, std_thresh=4.0):
        """
        Filtered trajectory and velocity arrays, time difference between adjacent frames are referred to time_diff

        Args:
            coordinates: (M, 3) array of tracked 3d positions for each person
            velocity: (M-1, 2) array of velocities in x-y plane
            ego_pos_inWorld: (1, 3) array of robot ego position in reference frame
            speed_max: max velocity per FRAME, if frame id is time stamp => velocity w.r.t seconds

        Returns: 
            filtered_traj: (M, 2) array of filtered trajectory
            filtered_velocity: (M-1, 2) of filtered velocity, calculated by filtered_traj and time_diff
        """
        agent2ego_inWorld = coordinates - ego_pos_inWorld
        filtered_velocity = copy.deepcopy(velocity)
        filtered_traj = copy.deepcopy(coordinates[:, :2])

        magnitudes = np.linalg.norm(filtered_velocity, axis=1)
        agent2ego_dist = np.linalg.norm(agent2ego_inWorld[:, :2], axis=1)

        change_idx = []
        for i in range(len(velocity)):
            # not correct points jump from far away to close (could be right)
            # correct only if jumping from close to far: distance suddenly increase
            if agent2ego_dist[i+1] > agent2ego_dist[i]:

                # max velocity per *frame*
                if magnitudes[i] > speed_max:
                    scale_median = np.median(magnitudes) / magnitudes[i]
                    filtered_velocity[i] *= scale_median
                    
                    change_idx.append(i)
                    if (i - 1) not in change_idx:
                        filtered_traj[i + 1] = filtered_velocity[i] * time_diff[i] + coordinates[i, :2]
                    else:
                        filtered_traj[i + 1] = filtered_velocity[i] * time_diff[i] + filtered_traj[i]
                
                # smoothen with local window
                if i >= window_size:
                    local_magnitudes = magnitudes[i-window_size:i]
                    local_mean = np.mean(local_magnitudes)
                    local_std = np.std(local_magnitudes)

                    scale_mean = local_mean / magnitudes[i] if magnitudes[i] != 0 else 1
                    # filter if current depth suddenly becomes large 
                    if abs(magnitudes[i] - local_mean) > std_thresh * local_std:
                        if np.linalg.norm(velocity[i] * scale_mean) < np.linalg.norm(filtered_velocity[i]):
                            filtered_velocity[i] = velocity[i] * scale_mean

                            change_idx.append(i)
                            if (i - 1) not in change_idx:
                                filtered_traj[i + 1] = filtered_velocity[i] * time_diff[i] + coordinates[i, :2]
                            else:
                                filtered_traj[i + 1] = filtered_velocity[i] * time_diff[i] + filtered_traj[i]
        
        filtered_velocity = np.diff(filtered_traj, axis=0) / time_diff
        
        return filtered_traj, filtered_velocity


    def interpolate_trajectory_across_frame(self, traj_arr, frame_ids, interp_interval):
        '''
        interpolate the trajectory for positions across continous frames

        Args:
            frame_ids: (M, ) array for all tracked frame ids in the buffer, they might not be continous
        '''
        continuous_frames = np.arange(frame_ids[-1], frame_ids[0], -interp_interval)[::-1]

        # if difference between all frame_ids is smaller than interval, manually keep 2 frames
        if len(continuous_frames) < 2:
            continuous_frames = np.array([frame_ids[-1]-interp_interval, frame_ids[-1]])

        interp_func = interp1d(frame_ids, traj_arr, axis=0, kind='linear', fill_value="extrapolate")
        interpolated_positions = interp_func(continuous_frames)
        # results = np.column_stack((continuous_frames, interpolated_positions))

        return interpolated_positions
    

    def annotate_with_detections(self, image):
        annotated_image = copy.deepcopy(image)
        online_tlwhs = self.recent_tracks[-1][1]

        for idx in range(len(online_tlwhs)):
            top, left, height, width = online_tlwhs[idx].astype(int)
            cv2.rectangle(annotated_image, (left, top), (left+width, top+height), (0, 255, 0), 2)  ## (start_x, start_y), (end_x, end_y)
        
        return annotated_image
    

    def draw_lidar_points_depth(self, image, lidar_data_inImg):
        '''
        lidar_data_inImg: shape N * 3, in the order of (width, height, depth)
        '''
        h, w = image.shape[0], image.shape[1]
        depths = lidar_data_inImg[:, 2]
        depth_ranges = [(-2, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 15)]

        colormap = cv2.COLORMAP_RAINBOW
        colors = []
        for depth in depths:
            if abs(depth) < 2:
                normalized_depth = 0
            elif 2 <= abs(depth) <= 4:
                normalized_depth = 40
            elif 4 <= abs(depth) <= 6:
                normalized_depth = 80
            elif 6 < abs(depth) <= 8:
                normalized_depth = 120
            elif 8 < abs(depth) <= 10:
                normalized_depth = 150
            else:
                normalized_depth = 180
            color_bgr = cv2.applyColorMap(np.uint8([[normalized_depth]]), colormap)[0, 0]
            colors.append(color_bgr)

        hs = lidar_data_inImg[:, 1].astype(int)
        ws = lidar_data_inImg[:, 0].astype(int)
        image[hs, ws] = colors
        image[np.clip(hs-1, 0, h-1), ws] = colors
        image[np.clip(hs+1, 0, h-1), ws] = colors
        image[hs, np.clip(ws-1, 0, w-1)] = colors
        image[hs, np.clip(ws+1, 0, w-1)] = colors


