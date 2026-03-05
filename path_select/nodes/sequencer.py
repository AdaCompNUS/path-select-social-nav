#!/usr/bin/env python
import rospy
import numpy as np

from interfaces.human_env_info_interface import HumanEnvInterface
from interfaces.path_planning_interface import PathPlanInterface

from path_select.msg import AgentStates
from path_select.srv import ControlService, ControlServiceResponse


class PathSendService:
    def __init__(self):
        self.service = rospy.Service("/controller_path", ControlService, self.path_handler)
        self.latest_path = None
        self.is_updated = False

    def update_path(self, new_path):
        self.latest_path = new_path
        self.is_updated = True

    def path_handler(self, req):
        if self.is_updated:
            self.is_updated = False
            return ControlServiceResponse(path=self.latest_path, success=True, message="Send new path to controller successfully.")
        else:
            return ControlServiceResponse(path=[], success=False, message="No new path, wait.")


if __name__ == "__main__":
    rospy.init_node("sequencer")

    human_env_interface = HumanEnvInterface(
        namespace="human_env_info",
        agents_topic="/pedestrian_robot_states", 
        costmap_topic="/costmap",
    )
    path_plan_interface = PathPlanInterface(
        namespace="path_planning", 
        state_topic="/planning_node_state",
    )

    pub_states_for_control = rospy.Publisher("/controller_all_agents", AgentStates, queue_size=1)
    service_path = PathSendService()


    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        agent_msg = human_env_interface.get_latest_agent_states()
        map_msg = human_env_interface.get_latest_costmap()

        if path_plan_interface.status == 'idle':
            path_plan_interface.request_plan(agent_msg, map_msg)

        elif path_plan_interface.status == 'finish':
            planner_response = path_plan_interface.request_retrieve(map_msg)
            # set the latest planned path for controller
            if planner_response.success:
                service_path.update_path(planner_response.path)
    
        # publish latest pedestrian states to controller
        pub_states_for_control.publish(agent_msg)

        rate.sleep()