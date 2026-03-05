import rospy
from std_msgs.msg import String
from path_select.srv import PlanningService, PlanningServiceRequest

import numpy as np

class PathPlanInterface:
    def __init__(self, namespace="path_planning", state_topic="/planning_node_state"):
        
        ns = rospy.resolve_name(namespace)
        self.state_sub = rospy.Subscriber(ns + state_topic, String, self._callback_state)

        rospy.wait_for_service("/execute_planning", timeout=30)
        rospy.wait_for_service("/retrieve_path", timeout=30)

        self.plan_service = rospy.ServiceProxy("/execute_planning", PlanningService)
        self.retrieve_service = rospy.ServiceProxy("/retrieve_path", PlanningService)

        self.planning_node_state = None

        rospy.wait_for_message(ns + state_topic, String, timeout=30)
    
    def _callback_state(self, msg):
        self.planning_node_state = msg.data

    @property
    def status(self) -> str:
        return str(self.planning_node_state)
    

    def request_plan(self, agent_msg, map_msg):
        request = PlanningServiceRequest()
        request.agent_states = agent_msg
        request.cost_map = map_msg

        response = self.plan_service(request)
        rospy.loginfo(f"Plan Response: success={response.success}, message={response.message}")
    

    def request_retrieve(self, map_msg):
        request = PlanningServiceRequest()
        # request.agent_states = agent_msg
        request.cost_map = map_msg

        response = self.retrieve_service(request)
        rospy.loginfo(f"Retrieve Response: success={response.success}, message={response.message}")
        return response
    