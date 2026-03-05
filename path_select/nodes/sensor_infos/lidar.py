import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2

from .ros_numpy import point_cloud2 as pcl2

class LidarListener:
    def __init__(self, lidar_topic="/lidar_points"):
        self.lidar_sub = rospy.Subscriber(lidar_topic, PointCloud2, self._callback_lidar)

        self.pcls = None
        self.timestamp = None

        msg = rospy.wait_for_message(lidar_topic, PointCloud2, timeout=30)
        self._callback_lidar(msg)
        rospy.loginfo("First LiDAR point clouds received.")
    
    def _callback_lidar(self, msg):
        self.timestamp = msg.header.stamp.to_sec()
        self.pcls = pcl2.pointcloud2_to_xyz_array(msg)
    
    @property
    def pcl_array(self) -> np.ndarray:
        return np.array(self.pcls)
    
    @property
    def pcl_time(self) -> float:
        return float(self.timestamp)



if __name__ == "__main__":
    
    rospy.init_node("lidar_listener", anonymous=True)
    lidar_listener = LidarListener()
    
    rate = rospy.Rate(10)
    rospy.spin()
