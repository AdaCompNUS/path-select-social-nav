# From Obstacles to Etiquette: Robot Social Navigation with VLM-Informed Path Selection

Implementation for RA-L'26 paper [From Obstacles to Etiquette: Robot Social Navigation with VLM-Informed Path Selection](https://path-etiquette.github.io/).

- [x] Codes released.
- [x] Update instructions for running the ROS codes.
- [ ] Update instructions for the VLM part.
- [ ] We will also upload the fine-tuned VLM to huggingface.


## 1. Overview
We deploy the system on an Nvidia Jetson Orin (JetPack 5.1) mounted on a Boston Dynamics Spot robot, with:

- OS: Ubuntu 20.04
- ROS version: Noetic
- Cuda version:  CUDA 11.8
- Python version: 3.8, with PyTorch 2.1/2.0

**NOTE:** We will test the system on upgraded software versions very soon and update the repository accordingly.

We uploaded the pedestrian trajectory prediction model (`ckp_prediction.p`) and one example rosbag to test our system in the [resource](https://drive.google.com/drive/folders/1hhV9VPZipW1IFXRS4YUYzT6QQrs-S2Oq?usp=sharing) folder. Feel free to download and test them. The human detection model (YOLOv10) can be downloaded from the [Ultralytics website](https://docs.ultralytics.com/models/yolov10/). For human tracking, we have already modified and integrated [ByteTrack](https://github.com/FoundationVision/ByteTrack) into our code. Thanks to the authors for their great work.


## 2. Setup
In `path_select` folder, we include the code running on ROS Noetic for robot navigation. Since we also use a conda environment with ROS, there are some additional configurations required to run the system. Please follow the instructions below.

In `vlm_train_inference` folder, we provide ... (TBD)


### 2.1 Robot
1. Create a workspace and copy our `path_select` folder into the `src` folder of this workspace. For example:
```bash
mkdir -p ~/ros_ws/src
cd ~/ros_ws/src
cp -r /your_download_path/path-select-social-nav/path_select .
```

2. Install the ROS packages required for your sensors.
   
3. Install a suitable PyTorch version according to your CUDA version, and then install the required Python packages for our system. If you are using a conda environment, we recommend installing packages with `pip` (*instead of* `conda`) after setting up torch and torchvision.
```bash
# For human motion extraction modules
pip install ultralytics 
pip install cython_bbox lap

python -m pip install scipy
pip3 install -U scikit-learn

# For the path selection module
pip install openai 
pip install requests

# For the local controller
pip install Cython

# Other packages
pip install opencv-python pyyaml
``` 
The local controller in our system is modified from the [implementation](https://github.com/sybrenstuvel/Python-RVO2). We use an in-place build that places the compiled library directly in the `adapt_rvo` folder. If you prefer building and installing it normally, please refer to the original implementation.
```bash
cd path_select/nodes/function_modules/adapt_rvo
python setup.py build_ext --inplace
```

4. Download the human detection model `yolov10b.pt` and place it in the folder `path_select/nodes/function_modules/yolo_model`. Download the trajectory prediction model `ckp_prediction.p` and place it in `path_select/nodes/function_modules/nmrf_predict`. You may also use YOLOv10 models with different sizes. In that case, modify the specified model path in `path_select/nodes/human_env_info_node.py` Line 78.
   
5. Modify the configuration in `path_select/config/params.yaml`, such as the calibrated camera intrinsics and extrinsics, the camera image size. Set the goal position with the key `final_goal` if the goal is defined relative to the starting point.
   
6. **IMPORTANT**: In our launch files `path_select/launch/path_select_navigate.launch` and `path_select/launch/human_perception.launch`, the node type is a shell script (e.g., `conda_run_human_env.sh`) where we run `exec python` inside the script. This is because the system must run inside a conda environment (named `social` here) with a preloaded library path to avoid dependency errors. If you are not using this setup, you can modify the node type to directly run the Python file (e.g., `human_env_info_node.py`).
   
7. Compile and run (ROS1 Noetic):
```bash
cd ~/ros_ws/src
catkin_make
source ~/ros_ws/devel/setup.bash
roslaunch path_select human_perception.launch
roslaunch path_select path_select_navigate.launch
```

### 2.2 VLM
TBD


## Citation

If you find this repo useful, please consider citing our paper as:
```bibtex
@article{fang2026socialnav,
  title={From Obstacles to Etiquette: Robot Social Navigation With VLM-Informed Path Selection},
  author={Fang, Zilin and Xiao, Anxing and Hsu, David and Lee, Gim Hee},
  journal={IEEE Robotics and Automation Letters},
  year={2026},
  volume={11},
  number={4},
  pages={3947-3954},
  doi={10.1109/LRA.2026.3662586}
}
```