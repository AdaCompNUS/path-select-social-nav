import os
import numpy as np
import copy
from .utils import filter_lidar_data, points2ref_frame

from scipy.ndimage import binary_dilation, gaussian_filter  #grey_dilation


class CostMapping:
    def __init__(self, filter_range: dict, axis_specs: dict, resolution=0.15, spread_radius=1):
        """
        Args:
            filter_range: dict of range tuple, i.e. fwd: (-2, 15)
            axis_specs: dict for lidar-direction correspondence tuple, i.e. {'fwd': (0, 1)}, means fwd align with x-axis <index 0> positive direction <+1>
            resolution: meters correspond to one pixel
            spread_radius: radius for gaussian filter / structure size for dilation to process cost map
        """
        self.fwd_range = filter_range['fwd']
        self.side_range = filter_range['side']
        self.height_range = filter_range['height']

        self.axis_specs = axis_specs

        self.resolution = resolution
        self.spread_radius = spread_radius

        # instant map from direct lidar data projection & cost map with prediction
        self.instant_occ_map = None
        self.pred_human_map = None
        self.cost_map = None
    
    
    def process_possible_walls(self, lidar_data, lidar_pos):
        wall_mask = np.abs(lidar_data[:, 2]) > 1.2
        wall_pts = lidar_data[wall_mask, :2]  # (N, 2)

        if wall_pts.shape[0] == 0:
            return None
        
        vectors = wall_pts - lidar_pos  # lidar_pos should be (1, 2)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-6
        directions = vectors / norms

        steps = np.linspace(0.8, 8.0, num=18).reshape(-1, 1, 1)  #np.arange(3, 12).reshape(-1, 1, 1)  # shape (K,1,1)
        ext_pts = wall_pts[None, :, :] + directions[None, :, :] * steps  # shape (K, N, 2)
        ext_pts = ext_pts.reshape(-1, 2)  # (K*N, 2)

        return ext_pts


    def generate_costmap(self, raw_lidar_data: np.ndarray, pred_locs: np.ndarray, 
                         ref_homo_trans: np.ndarray, ego_homo_trans: np.ndarray):
        """
        Args:
            raw_lidar_data: raw LiDAR pointclouds array from sensor, need to apply filtering and transform.
            pred_locs: (peds, N=20, T, 2) numpy array for human predicted trajectories, already in reference frame
        """
        ## create map for currently detected obstable
        filtered_lidar_data = filter_lidar_data(raw_lidar_data, self.axis_specs,
                                                fwd_range=self.fwd_range, side_range=self.side_range, height_range=self.height_range)
        
        lidar_data_inRef = points2ref_frame(ref_homo_trans, ego_homo_trans, filtered_lidar_data)

        ego_pos_inRef = points2ref_frame(ref_homo_trans, ego_homo_trans, np.array([0., 0., 0.]).reshape(1, -1))  # shape (1, 3)
        wall_data_inRef = self.process_possible_walls(lidar_data_inRef, lidar_pos=ego_pos_inRef[:, :2])

        lidar_occgrid = self.lidarpcl_to_occgrid(lidar_data_inRef)     # binary map
        lidar_occgrid = binary_dilation(lidar_occgrid, structure=np.ones((5, 5))).astype(lidar_occgrid.dtype)

        if wall_data_inRef is not None:
            ext_wall_occgrid = self.lidarpcl_to_occgrid(wall_data_inRef)
            ext_wall_occgrid = binary_dilation(ext_wall_occgrid, structure=np.ones((3, 3))).astype(ext_wall_occgrid.dtype)
            self.instant_occ_map = lidar_occgrid * 2 + ext_wall_occgrid  # 1: unknow area, 2: exact obstacle by lidar, 3: overlap due to dilation
        else:
            self.instant_occ_map = lidar_occgrid


        if pred_locs is None:
            self.cost_map = self.instant_occ_map
            
        else:
            ## assign weights to predicted human
            time_weights = np.linspace(0.8, 0.1, pred_locs.shape[2]).reshape(1, 1, pred_locs.shape[2])  # shape: 1,1,T
            w = np.tile(1. * time_weights, (pred_locs.shape[0], pred_locs.shape[1], 1)).flatten()   # shape before flatten: peds, N, T

            flat_pred_meter = np.concatenate((pred_locs[..., 0].flatten().reshape(-1, 1), pred_locs[..., 1].flatten().reshape(-1, 1)), axis=1)
            flat_pred_map = self.transform_meter2map(flat_pred_meter)
            map_row, map_col = flat_pred_map[0], flat_pred_map[1]

            ## filter for in-range predicted locations
            valid_mask = (0 <= map_row) & (map_row < self.instant_occ_map.shape[0]) & (0 <= map_col) & (map_col < self.instant_occ_map.shape[1])
            map_row, map_col, w = map_row[valid_mask], map_col[valid_mask], w[valid_mask]

            ## create a cost map only contains prediction information
            grid_index = map_row * self.instant_occ_map.shape[1] + map_col    # flattened 2D grid index

            cost_values = np.bincount(grid_index, weights=w, minlength=self.instant_occ_map.shape[0] * self.instant_occ_map.shape[1])
            visits = np.bincount(grid_index, weights=np.ones_like(w), minlength=self.instant_occ_map.shape[0] * self.instant_occ_map.shape[1])

            cost_matrix = cost_values.reshape(self.instant_occ_map.shape[0], self.instant_occ_map.shape[1])
            visit_matrix = visits.reshape(self.instant_occ_map.shape[0], self.instant_occ_map.shape[1])
            
            self.pred_human_map = np.divide(cost_matrix, visit_matrix, 
                                            where=visit_matrix > 0, out=np.zeros_like(self.instant_occ_map).astype(float))

            ## smoothen to obtain map for probability
            self.pred_human_map = gaussian_filter(self.pred_human_map, sigma=self.spread_radius, mode='constant')

            ## gaussian filter will smooth the peak, normalize to original peak value
            max_value = np.max(self.pred_human_map)
            if max_value > 0:
                self.pred_human_map *= (0.8 / max_value)
            
            ## add instant obstacle map and prediction map
            self.cost_map = self.instant_occ_map + self.pred_human_map
        
        return self.cost_map


    def lidarpcl_to_occgrid(self, filtered_lidar_data):
        '''
        axis_specs: A dict for lidar coordinate correspondence, i.e. {'fwd': (0, 1)}, means fwd align with x-axis <index 0> positive direction <+1>
        '''
        map_height = int(np.ceil((self.fwd_range[1] - self.fwd_range[0]) / self.resolution))
        map_width = int(np.ceil((self.side_range[1] - self.side_range[0]) / self.resolution))
        
        occupancy_grid = np.zeros((map_height, map_width), dtype=np.int8)
        
        # Usually for map image: height -> forward direction, width -> side direction
        h_points = filtered_lidar_data[:, self.axis_specs['fwd'][0]] * self.axis_specs['fwd'][1]
        w_points = filtered_lidar_data[:, self.axis_specs['side'][0]] * self.axis_specs['side'][1]

        h_indices = ((h_points - self.fwd_range[0]) / self.resolution).astype(int)
        w_indices = ((w_points - self.side_range[0]) / self.resolution).astype(int)
        
        h_indices = np.clip(h_indices, 0, map_height - 1).astype(int)
        w_indices = np.clip(w_indices, 0, map_width - 1).astype(int)

        occupancy_grid[h_indices, w_indices] = 1   # (row, column)

        return occupancy_grid
    

    def transform_meter2map(self, meter_loc):
        """
        Args:
            meter_loc: 2D array of postions in meter unit
        Returns:
            map_pos: (row_idx, col-idx), tuple of int array
        """
        # align with 'lidarpcl_to_occgrid' function
        # map_pos: (h-pos, w-pos) = (row, column)
        map_pos = (((self.axis_specs['fwd'][1] * meter_loc[:, self.axis_specs['fwd'][0]] - self.fwd_range[0]) / self.resolution).astype(int),
                   ((self.axis_specs['side'][1] * meter_loc[:, self.axis_specs['side'][0]] - self.side_range[0]) / self.resolution).astype(int))
        return map_pos
    

    def transform_map2meter(self, map_loc, pseudo_height=-0.5):
        """
        Transform map indices (2D) into lidar <reference frame> coordinates (3D)
        """
        meter_h = map_loc[:, 0] * self.resolution + self.fwd_range[0]
        meter_w = map_loc[:, 1] * self.resolution + self.side_range[0]

        point_3d = np.zeros((meter_h.shape[0], 3))

        point_3d[:, self.axis_specs['fwd'][0]] = meter_h * self.axis_specs['fwd'][1]
        point_3d[:, self.axis_specs['side'][0]] = meter_w * self.axis_specs['side'][1]
        point_3d[:, self.axis_specs['height'][0]] = pseudo_height

        return point_3d # shape: num * 3


