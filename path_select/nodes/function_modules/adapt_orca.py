import numpy as np
import math
import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'adapt_rvo'))
sys.path.append(module_path)
import adapt_rvo2

from types import SimpleNamespace

class AdaptORCA:
    def __init__(self, config):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step.

        """
        self.config = SimpleNamespace(**config)
        
        self.robot_goal = None
        self.max_neighbors = None
        self.sim = None
        # self.safety_space = self.config.robot_orca.safety_space
    
    def set_robot_goal(self, robot_goal: np.ndarray):
        """
        robot_goal: (2,) array, [x(m), y(m)] coordinate for goal position
        """
        self.robot_goal = robot_goal


    def predict(self, robot_state, human_states, pseudo_static_humans):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        Args:
            robot_state: A (5,) array for robot current position and velocity, with the order [x(m), y(m), theta(rad), v(m/s), omega(rad/s)]
            human_states: A list of pedestrians' current position and velocity: [x(m), y(m), vx(m/s), vy(m/s)]
            (All in global frame)
            pseudo_static_humans: A list for humans that tracking is not completed, no velocity.
        Returns:
            action: A tuple for (vx, vy)
        """
        self.max_neighbors = len(human_states)  # max number of humans = current number of humans
        
        robot_px, robot_py = robot_state[0], robot_state[1]
        robot_vx = robot_state[3] * math.cos(robot_state[2])   # remember to check x, y, theta direction definition
        robot_vy = robot_state[3] * math.sin(robot_state[2])

        params = self.config.sensor_range, self.max_neighbors, self.config.time_horizon, self.config.time_horizon_obst

        if self.sim is not None and self.sim.getNumAgents() != len(human_states) + 1:
            del self.sim
            self.sim = None
        
        if self.sim is None:
            self.sim = adapt_rvo2.PyRVOSimulator(self.config.time_step, *params, self.config.robot_radius, self.config.max_speed)
            self.sim.addAgent((robot_px, robot_py), *params, self.config.robot_radius + 0.01 + self.config.safety_space,
                              self.config.robot_v_pref, (robot_vx, robot_vy))
            if len(human_states) > 0:
                for human_state in human_states:
                    if np.linalg.norm(human_state[2:]) < 0.1:
                        pseudo_static_humans.append(human_state[:2])
                        continue
                    human_px, human_py = human_state[0], human_state[1]
                    human_vx, human_vy = human_state[2], human_state[3]
                    self.sim.addAgent((human_px, human_py), *params, self.config.human_radius + 0.01 + self.config.safety_space,
                                      self.config.max_speed, (human_vx, human_vy))
            
            if len(pseudo_static_humans) > 0:
                for human_static in pseudo_static_humans:
                    self.sim.addAgent((human_static[0], human_static[1]), *params, self.config.human_radius + 0.01 + self.config.safety_space,
                                      self.config.max_speed, (0, 0))  # human vx/vy=0

        else:
            self.sim.setAgentPosition(0, (robot_px, robot_py))
            self.sim.setAgentVelocity(0, (robot_vx, robot_vy))
            if len(human_states) > 0:
                for i, human_state in enumerate(human_states):
                    human_px, human_py = human_state[0], human_state[1]
                    human_vx, human_vy = human_state[2], human_state[3]
                    self.sim.setAgentPosition(i + 1, (human_px, human_py))
                    self.sim.setAgentVelocity(i + 1, (human_vx, human_vy))
        

        if self.robot_goal is None:
            raise ValueError("Goal position is not specified.")
        
        # set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        velocity = np.array((self.robot_goal[0] - robot_px, self.robot_goal[1] - robot_py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity * (self.config.robot_v_pref / speed) if speed > self.config.robot_v_pref else velocity


        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, human_state in enumerate(human_states):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        action = self.sim.getAgentVelocity(0)  # a tuple for (vx, vy)

        return action
    

    def velocity_xy_to_unicycle(self, vx, vy, current_heading, K=1.0):
        """
        Convert planar velocity (vx, vy) into linear velocity and angular velocity
        for a non-holonomic (unicycle) robot, given the robot's current heading.
        
        Args:
            current_heading: theta in robot state, it is the rotation angle along z-axis when expressing current frame in global frame
            K: Proportional gain for heading correction.
        Returns:
            (v, w): velocity and angular velocity
        """
        v = np.sqrt(vx**2 + vy**2)

        # desired T(world-> directed head), which means finally the robot's frontal should point to the disired direction
        desired_heading = math.atan2(vy, vx)

        heading_error = desired_heading - current_heading
        # wrap to [-pi, pi] for stability
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        
        # a simple proportional controller for angular velocity
        w = K * heading_error
        
        return v, w
    