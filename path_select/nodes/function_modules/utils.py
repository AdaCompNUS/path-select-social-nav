import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from collections import deque

def convert_to_homo_matrix(translation: np.ndarray, rot_vec: np.ndarray):
    rot_matrix = Rotation.from_quat(rot_vec).as_matrix()
    homo_matrix = np.vstack((np.hstack((rot_matrix, translation.reshape(-1, 1))), 
                             np.array([0., 0., 0., 1.])))
    return homo_matrix

def points2ref_frame(ref_homo_transform, toprocess_homo_transform, points_in_toprocess):
    """
    Transform from current frame to a reference frame, mainly for transformation from a local (ego) coordinates
    into the global coordinates
    """
    n = points_in_toprocess.shape[0]
    pts_3d_hom = np.hstack((points_in_toprocess, np.ones((n, 1))))

    P_pro2ref = np.dot(np.linalg.inv(ref_homo_transform), toprocess_homo_transform)

    points_in_ref_frame = np.transpose(np.dot(P_pro2ref, np.transpose(pts_3d_hom)))

    return points_in_ref_frame[:, :3]


def points_camera_to_lidar(points_inCam: np.ndarray, lidar_extrinsics: np.ndarray) -> np.ndarray:
    """
    Transform points in camera coordinates to LiDAR coordinates using extrinsics.
    
    Args:
        points_inCam: A (N, 3) np.array in camera coordinates.
        lidar_extrinsics (np.ndarray): 4x4 matrix transform lidar points into camera frame: P_cam=Ext*P_lidar.
    
    Returns:
        A (N, 3) np.array in LiDAR coordinates.
    """
    # convert 3d xyz coordinates from camera to lidar frame
    pts_hom = np.hstack((points_inCam, np.ones((points_inCam.shape[0], 1))))
    points_inLidar = np.transpose(np.dot(np.linalg.inv(lidar_extrinsics), np.transpose(pts_hom)))
    points_inLidar = points_inLidar[:, :3]

    return points_inLidar


def points_lidar_to_camera(points_inLidar: np.ndarray, lidar_extrinsics: np.ndarray) -> np.ndarray:
    """
    Transform points in lidar coordinates to Camera coordinates using extrinsics.

    Args:
        points_inLidar: A (N, 3) np.array in lidar coordinates.
        lidar_extrinsics (np.ndarray): 4x4 matrix transform lidar points into camera frame: P_cam=Ext*P_lidar.

    Return: 
        A (N, 3) np.array in Camera coordinates
    """
    n = points_inLidar.shape[0]
    pts_3d_hom = np.hstack((points_inLidar, np.ones((n, 1))))
    points_inCam = np.transpose(np.dot(lidar_extrinsics, np.transpose(pts_3d_hom)))
    points_inCam = points_inCam[:, :3]

    return points_inCam


def filter_lidar_data(lidar_data: np.ndarray, axis_specs: dict,
                      fwd_range=(-2, 15), side_range=(-10, 10), height_range=(-0.2, 2)) -> np.ndarray:
    """
    Args:
        lidar_data: A (N, 3) np.array in original lidar coordinates.
        axis_specs: A dict for lidar<-->direction correspondence, i.e. {'fwd': (0, 1)}, means fwd align with x-axis <index 0> positive direction <+1>

        fwd_range: back-most to forward-most
        side_range: left-most to right-most
        height_range: bottom-most to upper-most
        
    """
    f_points = lidar_data[:, axis_specs['fwd'][0]] * axis_specs['fwd'][1]
    s_points = lidar_data[:, axis_specs['side'][0]] * axis_specs['side'][1]
    h_points = lidar_data[:, axis_specs['height'][0]] * axis_specs['height'][1]

    f_filt = np.logical_and((f_points > fwd_range[0]), (f_points < fwd_range[1]))
    s_filt = np.logical_and((s_points > side_range[0]), (s_points < side_range[1]))
    h_filt = np.logical_and((h_points > height_range[0]), (h_points < height_range[1]))

    filter = np.logical_and(np.logical_and(f_filt, s_filt), h_filt)
    indices = np.argwhere(filter).flatten()

    filtered_lidar_data = lidar_data[indices]
    return filtered_lidar_data


def find_nearest_free_point(grid, start_x, start_y, max_radius=10):
    """
    Performs a BFS search to find the nearest free cell
    Args:
        start_x, start_y: row and col index for point in map
    """
    rows, cols = grid.shape
    visited = set()
    visited.add((start_x, start_y))
    queue = deque([(start_x, start_y)])

    # 8-connected neighbour
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    while queue:
        cx, cy = queue.popleft()
        if grid[cx, cy] < 1:
            return cx, cy
        
        if abs(cx - start_x) > max_radius or abs(cy - start_y) > max_radius:
            continue

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) not in visited and 0 <= nx < rows and 0 <= ny < cols:
                visited.add((nx, ny))
                queue.append((nx, ny))

    return None  # no free point found


def draw_paths_and_labels(valid_points_2d, image):
    """
    Draw planned paths on the image.
    
    Args:
        valid_points_2d: A list of 2D points in pixel coordinates, projected from path waypoints.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 3
    text_color = (0, 0, 0)
    circle_radius = 20

    # cmap = plt.get_cmap("Dark2")
    # colors = [tuple((np.array(cmap(i)[:3]) * 255).astype(int)) for i in range(8)]  # "Dark2" has 8 distinct colors

    for v_id in range(len(valid_points_2d)):
        points2d = valid_points_2d[v_id]

        if points2d.shape[0] < 1:
            continue
        
        # color_bgr = tuple(map(int, colors[v_id]))[::-1]
        
        # cv2.polylines(image, [points2d.reshape(-1, 1, 2).astype(np.int32)], isClosed=False, color=color_bgr, thickness=8)
        cv2.polylines(image, [points2d.reshape(-1, 1, 2).astype(np.int32)], isClosed=False, color=(0, 0, 255), thickness=8)  # red

        for r in range(1, points2d.shape[0]):
            center = points2d[r]

            cv2.circle(image, (int(center[0]), int(center[1])), circle_radius, (255, 255, 255), thickness=-1) # white for the patch
            # cv2.circle(image, (int(center[0]), int(center[1])), circle_radius, color_bgr, thickness=3)   # for border
            cv2.circle(image, (int(center[0]), int(center[1])), circle_radius, (0, 0, 255), thickness=5)

            ## put number in the center of the circle
            number = str(v_id + 1)
            text_size, _ = cv2.getTextSize(number, font, font_scale, font_thickness)
            text_x = int(center[0] - text_size[0] // 2)
            text_y = int(center[1] + text_size[1] // 2)
            cv2.putText(image, number, (text_x, text_y), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)


def show_paths_inMap(costmap, paths):
    cost_map_8bit = (costmap * 255).astype(np.uint8)
    cost_map_image = 255 - cv2.cvtColor(cost_map_8bit, cv2.COLOR_GRAY2BGR)
    cost_map_image = cv2.flip(cost_map_image, 0)

    height = cost_map_image.shape[0]
    
    for k in range(len(paths)):
        path_id = k + 1
        # path_arr = np.array(paths[k][1], dtype=int)
        path_arr = np.array(paths[k], dtype=int)

        for i in range(len(path_arr) - 1):
            pt1 = (path_arr[i][1], height - 1 - path_arr[i][0])
            pt2 = (path_arr[i + 1][1], height - 1 - path_arr[i + 1][0])
            cv2.line(cost_map_image, pt1, pt2, color=(0, 0, 255), thickness=1)
            cv2.circle(cost_map_image, pt1, 2, color=(0, 0, 255), thickness=-1)
        
        end_pt = (path_arr[-2][1], height - 1 - path_arr[-2][0])
        cv2.putText(cost_map_image, str(path_id), (end_pt[0] - 2 , end_pt[1] + 1),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 0, 0), thickness=2)
    
    return cost_map_image

