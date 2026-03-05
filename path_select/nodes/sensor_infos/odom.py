import rospy
import numpy as np
from nav_msgs.msg import Odometry

import tf2_ros
from geometry_msgs.msg import TransformStamped

class OdomListener:
    def __init__(self, odom_topic="/spot/odometry"):
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self._callback_odom)
        
        self.timestamp = None

        self._position = None
        self._orientation = None
        self._linear_vel = None
        self._angular_vel = None

        self.initial_pose = None   ## tuple of four arrays
        self.initial_received = False

        msg = rospy.wait_for_message(odom_topic, Odometry, timeout=30)
        self._callback_odom(msg)
        rospy.loginfo("First odometry message received.")
    
    def _callback_odom(self, msg):
        self.timestamp = msg.header.stamp.to_sec()

        self._position = msg.pose.pose.position
        self._orientation = msg.pose.pose.orientation
        self._linear_vel = msg.twist.twist.linear
        self._angular_vel = msg.twist.twist.angular

        if not self.initial_received:
            self.initial_pose = (
                np.array([self._position.x, self._position.y, self._position.z]), 
                np.array([self._orientation.x, self._orientation.y, self._orientation.z, self._orientation.w]),
                np.array([self._linear_vel.x, self._linear_vel.y, self._linear_vel.z]),
                np.array([self._angular_vel.x, self._angular_vel.y, self._angular_vel.z])
            )
            self.initial_received = True
            rospy.loginfo("Initial robot pose set as reference.")
        
        if np.sum((self.position - self.initial_pose[0])**2, axis=-1) > 400:
            self.initial_received = False
            rospy.loginfo("Wait to reset initial robot pose.")
    
    @property
    def position(self) -> np.ndarray:
        return np.array([self._position.x, self._position.y, self._position.z])

    @property
    def orientation(self) -> np.ndarray:
        return np.array([self._orientation.x, self._orientation.y, self._orientation.z, self._orientation.w])
    
    @property
    def linear_velocity(self) -> np.ndarray:
        return np.array([self._linear_vel.x, self._linear_vel.y, self._linear_vel.z])
    
    @property
    def angular_velocity(self) -> np.ndarray:
        return np.array([self._angular_vel.x, self._angular_vel.y, self._angular_vel.z])



class OdomListener_TF:
    def __init__(self):

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self._position = None
        self._orientation = None

        self.initial_pose = None  
        self.initial_received = False

    def get_transform(self, parent_frame="map", child_frame="body"):
        try:
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                parent_frame, child_frame, rospy.Time(0)  # get latest available transform
            )

            self._position = trans.transform.translation
            self._orientation = trans.transform.rotation

            # rospy.loginfo(f"Robot Position: x={_position.x}, y={_position.y}, z={_position.z}")
            # rospy.loginfo(f"Orientation: x={_orientation.x}, y={_orientation.y}, z={_orientation.z}, w={_orientation.w}")

            if not self.initial_received:
                self.initial_pose = (
                    np.array([self._position.x, self._position.y, self._position.z]),
                    np.array([self._orientation.x, self._orientation.y, self._orientation.z, self._orientation.w])
                )
                self.initial_received = True
                rospy.loginfo("Initial robot pose from map set as reference.")


        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Could not retrieve transform: {e}")
    
    @property
    def position(self) -> np.ndarray:
        return np.array([self._position.x, self._position.y, self._position.z])

    @property
    def orientation(self) -> np.ndarray:
        return np.array([self._orientation.x, self._orientation.y, self._orientation.z, self._orientation.w])



# if __name__ == "__main__":
#     rospy.sleep(2)
#     rospy.init_node("tf_listener", anonymous=True)
#     odom_listener = OdomListener()
    
#     rate = rospy.Rate(10)
#     while not rospy.is_shutdown():
#         odom_listener.get_transform()
#         rate.sleep()



if __name__ == "__main__":
    rospy.init_node("odom_listener", anonymous=True)
    odom_listener = OdomListener()
    
    rate = rospy.Rate(30)
    rospy.spin()