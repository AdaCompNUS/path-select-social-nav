import os
import warnings
import torch
import numpy as np
import yaml

from scipy.interpolate import interp1d
from .nmrf_predict.model_mrf_stride_sample import MRF_CVAE

class HumanPredictor:
    def __init__(self, pred_model_path: str, hyper_config_path: str, obs_step=4, fut_step=6, N=20):
        """
        Args:
            N: number of generate samples, by default is 20; 
               if changes, only gaussian sampling can be used -> noise=False
        """
        self.obs_step = obs_step
        self.fut_step = fut_step
        self.N = N

        self.traj_mean = torch.FloatTensor([0, 0]).cuda().unsqueeze(0).unsqueeze(0)
        self.traj_scale = 1.

        with open(hyper_config_path, 'r') as file:
            self.hyper_config = yaml.safe_load(file)

        self.mrf_predictor = MRF_CVAE(fut_step=self.fut_step, z_dim=self.hyper_config['z_dim'], f2_dim=self.hyper_config['f2_dim'],
                                      sigma1=1., sigma2=1., stride=self.hyper_config['stride'], N=self.N).cuda()
        
        model_dict = torch.load(pred_model_path, map_location=torch.device('cpu'))
        self.mrf_predictor.load_state_dict(model_dict)
        self.mrf_predictor.eval()
    

    def preprocess_trajectory(self, traj_dict: dict, robot_traj: np.ndarray, downsample_scale=4):
        """
        Sample to fit the prediction network input's frequency

        Args:
            traj_dict: dict of tuple for human trajectories, (x-y coordinates array, True/False for velocity)
            robot_traj: (T, 2) array for robot history locations
            downsample_scale: prediction input is 2.5Hz, calculate with: <info freq. / 2.5>

        Returns:
            pre_motion: (1+M, obs_step, 2) tensor for robot AND human past trajectories, first entry is robot
        """
        
        traj_list = []

        # continue for pedestrian trajectory only if robot ego motion has more than 1 frame (can compute velocity)
        if robot_traj.shape[0] > 1:
            traj_list.append(self.resample_and_fit_observation(robot_traj, downsample_scale))
        else:
            return None

        for person_id, frame_locs in traj_dict.items():
            coordinates, indicator = frame_locs[0], frame_locs[1]

            if indicator is True:  # at least 2 frames in obs_step
                person_traj = self.resample_and_fit_observation(coordinates, downsample_scale)
                traj_list.append(person_traj)
        
        if len(traj_list) == 1:
            return None  # except for robot, all human agents only have a single frame
        
        pre_motion = torch.tensor(np.stack(traj_list)).type(torch.float) 
        return pre_motion
    

    def resample_and_fit_observation(self, coordinates, downsample_scale):
        """ 
        Downsample the trajectory with a scale, truncate or pad to the defined obs_step
        """
        if coordinates.shape[0] > downsample_scale:
            downsampled = coordinates[::-1][::downsample_scale][::-1]  # keep the latest frame by inverse / inverse back

        elif coordinates.shape[0] <= downsample_scale:
            x = np.arange(coordinates.shape[0], 0, -1)
            y = coordinates[::-1]

            extra_x = np.arange(coordinates.shape[0], coordinates.shape[0] - (downsample_scale+1), -1)

            interp_hist = interp1d(x, y, axis=0, kind='linear', fill_value="extrapolate")
            extra_y = interp_hist(extra_x)
            downsampled = extra_y[::downsample_scale][::-1]
        
        # truncating or padding to fit obs_step
        if downsampled.shape[0] >= self.obs_step:
            return downsampled[:self.obs_step]
        else:
            pad_size = self.obs_step - downsampled.shape[0]
            padding = np.tile(downsampled[0], (pad_size, 1))  # repeat the first row (oldest history)
            return np.vstack((padding, downsampled))
    

    def data_preprocess(self, pre_motion: torch.FloatTensor):
        """
        data pre-processing for prediction network: normalization, augmentation

        Args:
            pre_motion: robot + human history trajectory tensor, first row is for robot
            robot_pre_motion: robot history numpy array, to include robot's influence to human
        """

        agents_num = pre_motion.shape[0]   # number is: 1 robot + N peds

        batch_past_traj = pre_motion.cuda()
        traj_mask = torch.zeros(agents_num, agents_num).cuda()

        # current position (last frame of history)
        initial_pos = batch_past_traj[:, -1:]

        scene_coords = initial_pos.squeeze(1)  # agents, 1, 2 -> agents, 2
        dist_to_nbor = torch.cdist(scene_coords, scene_coords)
        pair_indices = (dist_to_nbor <= 3.0).nonzero()

        traj_mask[pair_indices[:, 0], pair_indices[:, 1]] = 1.

        # augment input: absolute position, relative position, velocity
        batch_past_traj_abs = (batch_past_traj - self.traj_mean) / self.traj_scale
        batch_past_traj_rel = (batch_past_traj - initial_pos) / self.traj_scale
        batch_past_traj_vel = torch.cat((
            batch_past_traj_rel[:, 1:] - batch_past_traj_rel[:, :-1], torch.zeros_like(batch_past_traj_rel[:, -1:])), dim=1)
        
        aug_past_traj = torch.cat((batch_past_traj_abs, batch_past_traj_rel, batch_past_traj_vel), dim=-1)

        data_dict = {
            'peds_stats': torch.Tensor([agents_num]).reshape(-1).long(),   # actually is robot + human
            'traj_mask': traj_mask,
            'past_traj': aug_past_traj,
        }
        return data_dict
    


    def predict_future_trajectory(self, traj_dict: dict, robot_traj: np.ndarray, downsample_scale=4):
        """
        generate predictions

        Args:
            traj_dict: online trajectory dict from detection-tracker
            robot_ego_list: list of tuple (frame_id, ego_inRef_coords) 

        Returns:
            pred_trajs: (peds, self.N, fut_step, 2) numpy array for predicted human trajectory
        """

        pre_motion = self.preprocess_trajectory(traj_dict, robot_traj, downsample_scale)

        if pre_motion is not None:
            proc_data_dict = self.data_preprocess(pre_motion)

            with torch.no_grad():
                pred_mrf_sample, _ = self.mrf_predictor(proc_data_dict, self.N, self.fut_step, noise=True)
                # N, (robot + peds), fut_step, 2
            
            peds_sample = pred_mrf_sample.detach().cpu()[:, 1:]  # only extract samples for human
            peds_sample = peds_sample * self.traj_scale + pre_motion[1:, -1].contiguous().view(-1, 1, 2)[None, :]
            
            pred_trajs = peds_sample.permute(1, 0, 2, 3).numpy()
            return pred_trajs
        
        else:
            warnings.warn("Not enough history frames for trajectory prediction.", category=UserWarning)
            return None