import rospy
import numpy as np
from types import SimpleNamespace

from path_select.msg import AgentStates
from path_select.srv import ControlService, ControlServiceRequest

class ControllerInterface:
    """
    In fact it is "sequencer interface", it receives published message from sequencer and acts as service client.
    """

    def __init__(self):

        rospy.wait_for_service("/controller_path", timeout=60)

        self.path_service = rospy.ServiceProxy("/controller_path", ControlService)

        self.sub_agents = rospy.Subscriber("/controller_all_agents", AgentStates, self._callback_agents)

        # in AgentStates message
        self.dynamic_human_states = None
        self.pseudo_static_humans = None
        self.robot_state = None
        self.robot_init_state = None

        msg = rospy.wait_for_message("/controller_all_agents", AgentStates, timeout=30)
        self._callback_agents(msg)
        rospy.loginfo("First all-agents states for controller received.")

    def request_path(self):
        sequencer_response = self.path_service()
        if sequencer_response.success:
            path_to_follow = np.array(sequencer_response.path, dtype=np.float32).reshape(-1, 3)   # (x,y,z) in meter unit        
            return path_to_follow
        else:
            return None

    
    def _callback_agents(self, msg):
        self.dynamic_human_states = np.array(msg.dynamic_human, dtype=np.float32).reshape(-1, 4)
        self.pseudo_static_humans = np.array(msg.static_human, dtype=np.float32).reshape(-1, 2)
        # stacked vector for robot state, (4, 4) matrix, row-order is position/orient/linear-vel/angular-vel
        self.robot_state = np.array(msg.robot, dtype=np.float32).reshape(4, 4)
        self.robot_init_state = np.array(msg.robot_ref, dtype=np.float32).reshape(4, 4)
    

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
    

