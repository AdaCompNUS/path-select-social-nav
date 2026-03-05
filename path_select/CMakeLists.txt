cmake_minimum_required(VERSION 3.0.2)
project(path_select)

find_package(catkin REQUIRED COMPONENTS 
  rospy
  message_generation
  std_msgs 
  sensor_msgs
  nav_msgs
  geometry_msgs
  cv_bridge
  roslaunch
)

roslaunch_add_file_check(launch)

catkin_python_setup()

add_message_files(
  FILES
  AgentStates.msg 
  CostMap.msg
)

add_service_files(
  FILES
  PlanningService.srv
  ControlService.srv
)

generate_messages(
  DEPENDENCIES 
  std_msgs 
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS 
  rospy 
  message_runtime 
  std_msgs 
  sensor_msgs 
  cv_bridge
)

# remember to change files types to make them executable
catkin_install_python(PROGRAMS
  nodes/human_env_info_node.py
  nodes/path_planning_node.py
  nodes/controller_node.py
  nodes/sequencer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  OPTIONAL
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)