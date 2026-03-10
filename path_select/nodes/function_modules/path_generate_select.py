import os
import numpy as np
import heapq
from scipy.ndimage import binary_dilation
from scipy.spatial.distance import directed_hausdorff

from PIL import Image
import requests
import base64
from io import BytesIO

import json
import re
from openai import OpenAI

class PathPlanner:
    def __init__(self):
        self.selected_path = None
    
    # Euclidean distance
    def heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    
    def path_length(self, path):
        return np.sum(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1))
    

    def generate_anchors(self, map_grid, start_point, goal_point, anchor_cfgs):
        """
        Args:
            map_grid: instantly directed obstacle map by lidar
            start_point: position expressed in tuple, in map coordinates
            goal_point: goal position in map coordinates (row, column)

            anchor_cfgs: dict of configs to generate anchors, including min_dist, range, number, etc.
                'min_dist': minimum distance to separate grids for anchors, in map pixel unit
                'range_width': horizontal range (width) for anchor generating, in map pixel unit
                'forward_scale': scale to multiply for start-goal distance
                'dilate_radius': size of structure np.ones((r, r)) to dilate the map for obstacle filtering 
        """
        min_anchor_dist = anchor_cfgs['min_dist']
        anchor_range_width = anchor_cfgs['range_width']
        anchor_forward_scale = anchor_cfgs['forward_scale']
        num_anchors = anchor_cfgs['num_anchors']
        dilate_radius = anchor_cfgs['dilate_radius']
        
        # convert map row-column index into x-y Cartesian coordinates
        start = np.array([start_point[1], start_point[0]])
        goal = np.array([goal_point[1], goal_point[0]])
        direction = goal - start

        length = np.linalg.norm(direction)
        
        unit_dir = direction / length
        perp_direction = np.array([-unit_dir[1], unit_dir[0]]) * (anchor_range_width / 2)

        # create a regular grid in a normalized coordinate space (0 to 1)
        num_along_path = int(anchor_forward_scale * length / min_anchor_dist) + 1
        num_perp_path = int(anchor_range_width / min_anchor_dist) + 1

        # rectangular region
        t_grid, u_grid = np.meshgrid(np.linspace(0.15, 1, num_along_path), np.linspace(-0.5, 0.5, num_perp_path))
        grid_points = np.vstack([t_grid.ravel(), u_grid.ravel()]).T
        
        # transform points from normalized space to the parallelogram
        transform_matrix = np.array([unit_dir * (anchor_forward_scale * length), perp_direction * 2]).T
        sampled_points = start + grid_points @ transform_matrix.T
        
        # convert points to integer indices for obstacle checking
        sampled_points = sampled_points.astype(int)
        xs, ys = sampled_points[:, 0], sampled_points[:, 1]
        
        # filter out points that are outside obstacle-free cells
        in_bounds = (xs >= 0) & (ys >= 0) & (xs < map_grid.shape[1]) & (ys < map_grid.shape[0])
        sampled_points = sampled_points[in_bounds]
        xs = xs[in_bounds]
        ys = ys[in_bounds]
        
        dilated_grid = binary_dilation(map_grid, structure=np.ones((dilate_radius, dilate_radius))).astype(map_grid.dtype)
        mask = dilated_grid[ys, xs] == 0
        traversable_points = sampled_points[mask]

        # select the desired number of points
        np.random.seed(0)
        if len(traversable_points) >= num_anchors:
            selected_indices = np.random.choice(len(traversable_points), num_anchors, replace=False)
            valid_points = traversable_points[selected_indices]
        else:
            valid_points = traversable_points
        
        distances = np.linalg.norm(valid_points - start, axis=1)  # in pixel unit

        sorted_indices = np.argsort(distances)
        sorted_points = valid_points[sorted_indices]

        # convert back into grid map indexing - (fwd, side)/(row, column)
        anchors = [tuple(pt[::-1]) for pt in sorted_points]

        return anchors


    def astar_with_predicted_map(self, cost_map_with_preds, start, goal):
        rows, cols = cost_map_with_preds.shape
        
        # obstacles / unknowns (>=1) have infinite cost, others keep the same
        # 1: unknow area, 2: exact obstacle by lidar, 3: overlap due to dilation
        cost_map = np.where(cost_map_with_preds >= 1, np.inf, cost_map_with_preds)

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4 directions: right, up, left, down

        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start))  # (f_cost, g_cost, node)
        # f_cost: total cost
        # g_cost: actual cost, from start to current node

        g_cost = {start: 0}
        came_from = {}

        iter_count = 0
        while open_set and iter_count < 2500:
            _, current_g, current = heapq.heappop(open_set)
            iter_count += 1

            if current == goal:
                # reconstruct path from goal to start
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                    tentative_g = current_g + cost_map[neighbor] * 5 + (direction[0] < 0) * 20

                    if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                        g_cost[neighbor] = tentative_g
                        f_cost = tentative_g + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_cost, tentative_g, neighbor))
                        came_from[neighbor] = current

        print(f"Discard path with end point {goal}")
        return None # if no path found
    

    def find_paths_through_anchors(self, grid, start, goal, anchors):
        """Generate multiple paths through each anchor.
        """
        all_paths = []
        smoother = LineOfSightSmoothing()

        for anchor in anchors:
            # path from start to anchor
            path_to_anchor = self.astar_with_predicted_map(grid, start, anchor)
            
            if path_to_anchor:
                # path from anchor to goal
                path_to_goal = self.astar_with_predicted_map(grid, anchor, goal)
                
                if path_to_goal:
                    # combine the two parts of the path
                    full_path = smoother.smooth_path(grid, path_to_anchor)[:-1] + smoother.smooth_path(grid, path_to_goal)
                    all_paths.append(full_path)

        return all_paths
    

    def find_closest_path_to_reference(self, new_paths, ref_path, resample_num_pts=20, trunc_dist_thresh=3./0.15):
        """
        Args:
            new_paths: list of tuple for new paths, without clustering/resampling/truncating
            ref_path: tuple, the previously chosen path, from start to goal without resampling/truncating
        """
        hausdorff_dists = []
        for k in range(len(new_paths)):
            resamp_trunc_p = self.resample_and_truncate_path(new_paths[k], resample_num_pts, pixel_threshold=trunc_dist_thresh)
            ref_ = self.resample_and_truncate_path(ref_path, resample_num_pts, pixel_threshold=trunc_dist_thresh)

            d1 = directed_hausdorff(resamp_trunc_p, ref_)[0]
            d2 = directed_hausdorff(ref_, resamp_trunc_p)[0]
            hausdorff_dists.append(max(d1, d2))
        
        idx = np.argmin(hausdorff_dists)
        closest_path = new_paths[idx]

        return closest_path


    def multiple_paths_clustering(self, paths, cluster_threshold=5, resample_num_pts=20, trunc_dist_thresh=3./0.15):
        """
        Args:
            paths: list of tuple for waypoints
            cluster_threshold: threshold to set two paths into one cluster, in pixel unit
            trunc_dist_thresh: threshold to truncate a path, in pixel unit, compute by: distance_in_meter / resolution
        Returns:
            clustered_paths: list of tuple, only one path for a cluster is kept
            clustered_resample_paths: list of arrays with shape (Num_waypoints, 2)
        """
        num_paths = len(paths)
        visited = set()
        groups = []

        # resample to get more waypoints for hausdorff distance
        resampled_paths = []
        for k in range(num_paths):
            resampled_paths.append(self.resample_path(np.array(paths[k]), num_points=resample_num_pts))

        for i in range(num_paths):
            if i in visited:
                continue
            group = [i]
            visited.add(i)

            trunc_resample_path_i = self.truncate_path(resampled_paths[i], pixel_threshold=trunc_dist_thresh)
            for j in range(i + 1, num_paths):
                if j not in visited:
                    trunc_resample_path_j = self.truncate_path(resampled_paths[j], pixel_threshold=trunc_dist_thresh)

                    d1 = directed_hausdorff(trunc_resample_path_i, trunc_resample_path_j)[0]
                    d2 = directed_hausdorff(trunc_resample_path_j, trunc_resample_path_i)[0]
                    hausdorff_distance = max(d1, d2)
                    if hausdorff_distance <= cluster_threshold:
                        group.append(j)
                        visited.add(j)
            
            groups.append(group)
        
        clustered_resample_paths = []
        clustered_paths= []

        for group in groups:
            lengths = [self.path_length(paths[i]) for i in group]   # choose according to the full length of path
            shortest_idx = np.argmin(lengths)
            clustered_paths.append(paths[group[shortest_idx]])
            clustered_resample_paths.append(resampled_paths[group[shortest_idx]])

        return clustered_paths, clustered_resample_paths
    


    def resample_path(self, path, num_points=20):
        """Upsample for dense points along the path."""
        cumulative_distances = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        cumulative_distances = np.insert(cumulative_distances, 0, 0)  # start at distance 0

        # generate new distances uniformly spaced between start and end
        new_distances = np.linspace(0, cumulative_distances[-1], num=num_points)

        # interpolate to get the new points
        resampled_path = np.empty((num_points, path.shape[1]))  # num_points rows, 2 columns for (x, y)
        for dim in range(path.shape[1]):  # interpolate each dimension separately (x and y)
            resampled_path[:, dim] = np.interp(new_distances, cumulative_distances, path[:, dim])
        
        return resampled_path


    def truncate_path(self, path, pixel_threshold):
        """
        Args:
            pixel_threshold: threshold to truncate path, in pixel unit. pixel_threshold = dist_threshold / map_resolution
        """
        distances = np.linalg.norm(path - path[0], axis=1)
        within_idx = np.where(distances <= pixel_threshold)[0]

        truncated_path = path[:within_idx[-1] + 1]
        return truncated_path


    def resample_and_truncate_path(self, path, resample_num_pts=20, pixel_threshold=3./0.15):
        """
        Args:
            path: list of waypoints in tuple format
        """
        resampled = self.resample_path(np.array(path), resample_num_pts)
        resampled_trunc_path = self.truncate_path(resampled, pixel_threshold)
        return resampled_trunc_path
    

    
class LineOfSightSmoothing:
    def __init__(self, max_jump=6):
        self.line_of_sight_cache = {}
        self.max_jump = max_jump  # limit the jump to reduce risk of large wrong shortcuts

    def line_of_sight(self, grid, start, end):
        y0, x0 = start
        y1, x1 = end

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while (y0, x0) != (y1, x1):
            if grid[y0][x0] == 1:  # obstacle
                return False
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return grid[y1][x1] == 0  # also check end cell

    def line_of_sight_cached(self, grid, start, end):
        key = (start, end)
        if key not in self.line_of_sight_cache:
            self.line_of_sight_cache[key] = self.line_of_sight(grid, start, end)
        return self.line_of_sight_cache[key]

    def smooth_path(self, grid, path):
        if not path:
            return []

        smoothed_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            # look ahead within a small window (max_jump)
            max_j = min(i + self.max_jump, len(path) - 1)
            found = False
            for j in range(max_j, i, -1):
                if self.line_of_sight_cached(grid, path[i], path[j]):
                    smoothed_path.append(path[j])
                    i = j
                    found = True
                    break
            if not found:
                # fallback: can't jump, just move to next point
                i += 1
                smoothed_path.append(path[i])
        return smoothed_path



### ------------------------------ VLMs ---------------------------------


class GPT_Query():
    def __init__(self, api_key="your-openai-api-key"):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }
        
    def encode_image(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def create_payload_object(self, instruction, base64_image):
        payload = {
            "model": "gpt-4o-2024-05-13",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": f"{instruction}"
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
                }
            ],
            "max_tokens": 800,
            "temperature": 0.0001,
            # "top_p": 1
            }
        return payload
    
    def chat(self, prompt, image):
        base64_img = self.encode_image(image)
        payload = self.create_payload_object(prompt, base64_img)
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)

        return response
    
    def str2path_num(self, vlm_answer):
        match = re.search(r"'Chosen Path'\s*:\s*['\"**]*\s*([\w\s]+)\s*['\"**]*", vlm_answer)
        if not match:
            match2 = re.search(r"\"Chosen Path\"\s*:\s*['\"**]*\s*([\w\s]+)\s*['\"**]*", vlm_answer)
            path_idx = match2.group(1).strip()
            if len(path_idx) != 1:
                path_idx = path_idx[-1]
        else:
            path_idx = match.group(1).strip()
            if len(path_idx) != 1:
                path_idx = path_idx[-1]
        
        return int(path_idx)


class Qwen_Query():
    def __init__(self, server_url="", temperature=0.0001):
        self.server_url = server_url
        self.temperature = temperature

        self.client = OpenAI(api_key="your-openai-api-key",
                             base_url=self.server_url)

    def chat(self, prompt, image):
        base64_image = self.convert_pil_image_to_base64(image)
        response = self.client.chat.completions.create(
            model="your-Qwen2.5-VL-model-path",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": f"{prompt}"
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
                }
            ],
            temperature=self.temperature,
        )

        return response
    
    def convert_pil_image_to_base64(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    

    def str2path_num(self, vlm_answer):
        match = re.search(r"'Chosen Path'\s*:\s*['\"**]*\s*([\w\s]+)\s*['\"**]*", vlm_answer)
        if not match:
            match2 = re.search(r"\"Chosen Path\"\s*:\s*['\"**]*\s*([\w\s]+)\s*['\"**]*", vlm_answer)
            path_idx = match2.group(1).strip()
            if len(path_idx) != 1:
                path_idx = path_idx[-1]
        else:
            path_idx = match.group(1).strip()
            if len(path_idx) != 1:
                path_idx = path_idx[-1]
        
        return int(path_idx)


