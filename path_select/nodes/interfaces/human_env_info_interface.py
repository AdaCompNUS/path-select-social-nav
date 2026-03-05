import rospy
import numpy as np
from types import SimpleNamespace

from cv_bridge import CvBridge
from path_select.msg import CostMap
from path_select.msg import AgentStates


class HumanEnvInterface:
    def __init__(self, namespace="human_env_info",
                 agents_topic="/pedestrian_robot_states", costmap_topic="/costmap"):
        
        ns = rospy.resolve_name(namespace)
        self.agents_sub = rospy.Subscriber(ns + agents_topic, AgentStates, self._callback_agents)
        self.costmap_sub = rospy.Subscriber(ns + costmap_topic, CostMap, self._callback_costmap)

        self.bridge = CvBridge()

        self.agent_msg = None
        self.map_msg = None

        # in AgentStates message
        self.dynamic_human_states = None
        self.pseudo_static_humans = None
        self.robot_state = None
        self.robot_init_state = None
        self.image = None

        # in CostMap message
        self.costmap_with_preds = None
        self.instant_occ_map = None

        self.ego_arr_inMap = None
        self.goal_arr_inMap = None

        rospy.wait_for_message(ns + agents_topic, AgentStates, timeout=30)
        rospy.wait_for_message(ns + costmap_topic, CostMap, timeout=30)
    
    def _callback_agents(self, msg):
        self.agent_msg = msg

        self.dynamic_human_states = np.array(msg.dynamic_human, dtype=np.float32).reshape(-1, 4)
        self.pseudo_static_humans = np.array(msg.static_human, dtype=np.float32).reshape(-1, 2)
        # stacked vector for robot state, (4, 4) matrix, row-order is position/orient/linear-vel/angular-vel
        self.robot_state = np.array(msg.robot, dtype=np.float32).reshape(4, 4)
        self.robot_init_state = np.array(msg.robot_ref, dtype=np.float32).reshape(4, 4)
        self.image = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding='bgr8')

    def _callback_costmap(self, msg):
        self.map_msg = msg

        row = msg.row
        col = msg.col
        self.costmap_with_preds = np.array(msg.map_preds, dtype=np.float32).reshape(row, col)
        self.instant_occ_map = np.array(msg.map_instant, dtype=np.int8).reshape(row, col)
        # robot current position & goal
        self.ego_arr_inMap = np.array(msg.ego_inMap, dtype=np.int8)
        self.goal_arr_inMap = np.array(msg.goal_inMap, dtype=np.int8)
    
    
    def get_latest_agent_states(self):
        return self.agent_msg
    
    def get_latest_costmap(self):
        return self.map_msg
    

    @property
    def human_states(self) -> list:
        num = self.dynamic_human_states.shape[0]
        if num > 0:
            return [self.dynamic_human_states[i] for i in range(num)]
        else:
            return self.dynamic_human_states.tolist()
    
    @property
    def static_humans(self) -> list:
        num = self.pseudo_static_humans.shape[0]
        if num > 0:
            return [self.pseudo_static_humans[i] for i in range(num)]
        else:
            return self.pseudo_static_humans.tolist()
    
    @property
    def robot(self):
        state_dict = {
            "position": self.robot_state[0, :3],
            "orientation": self.robot_state[1, :],
            "linear_velocity": self.robot_state[2, :3],
            "angular_velocity": self.robot_state[3, :3],
        }
        return SimpleNamespace(**state_dict)
    
    @property
    def robot_init(self):
        state_dict = {
            "position": self.robot_init_state[0, :3],
            "orientation": self.robot_init_state[1, :],
            "linear_velocity": self.robot_init_state[2, :3],
            "angular_velocity": self.robot_init_state[3, :3],
        }
        return SimpleNamespace(**state_dict)
    
    @property
    def rgb_image(self) -> np.ndarray:
        return np.array(self.image)
    
    @property
    def costmap(self) -> np.ndarray:
        return self.costmap_with_preds
    
    @property
    def instant_map(self) -> np.ndarray:
        return self.instant_occ_map
    
    @property
    def ego(self) -> tuple:
        return tuple(self.ego_arr_inMap)
    
    @ property
    def goal(self) -> tuple:
        return tuple(self.goal_arr_inMap)
