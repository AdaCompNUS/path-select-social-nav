import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageListener:
    def __init__(self, rgb_image_topic="/usb_cam1/image_raw"):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(rgb_image_topic, Image, self._callback_image)

        self.image = None
        self.timestamp = None

        msg = rospy.wait_for_message(rgb_image_topic, Image, timeout=30)
        self._callback_image(msg)
        rospy.loginfo("First RGB image received.")
    
    def _callback_image(self, msg):
        self.timestamp = msg.header.stamp.to_sec()
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    
    @property
    def rgb_image(self) -> np.ndarray:
        return np.array(self.image)
    
    @property
    def rgb_time(self) -> float:
        return float(self.timestamp)
    


# if __name__ == "__main__":
    
#     rospy.init_node("image_listener", anonymous=True)
#     image_listener = ImageListener()
    
#     rate = rospy.Rate(30)
#     rospy.spin()