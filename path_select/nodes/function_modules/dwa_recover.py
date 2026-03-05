import numpy as np
import math

class DeadlockDetector:
    def __init__(self, window_size=15, min_progress=0.2, velocity_threshold=0.05):
        self.window_size = window_size
        self.positions = []        # stores (position, subgoal)
        self.velocity_threshold = velocity_threshold
        self.min_progress = min_progress
        self.deadlock = False

    def update(self, position: np.ndarray, linear_velocity: float, subgoal: np.ndarray):
        self.positions.append((position, subgoal))

        if len(self.positions) > self.window_size:
            self.positions.pop(0)
        
        if len(self.positions) < self.window_size:
            return False
        
        
        start_pos, start_goal = self.positions[0]
        current_pos, _ = self.positions[-1]
        
        # check if robot is moving
        motion = np.linalg.norm(current_pos - start_pos)
        if motion > self.min_progress:
            self.deadlock = False
            return False

        # check if robot is not approaching the goal
        start_dist = np.linalg.norm(start_goal - start_pos)
        current_dist = np.linalg.norm(start_goal - current_pos)
        progress = start_dist - current_dist

        if progress < self.min_progress:
            self.deadlock = True
            return True
        else:
            self.deadlock = False
            return False


class DWA_control(object):
    def __init__(self):

        # robot parameter
        self.max_speed = 0.5
        self.min_speed = 0.1
        self.max_yaw_rate = 0.698132         # 40.0 * math.pi / 180.0 [rad/s]
        self.max_accel = 1.0                 # [m/ss]
        self.max_delta_yaw_rate = 0.698132   # 40.0 * math.pi / 180.0 [rad/ss]

        self.v_resolution = 0.05             # 0.01 [m/s]
        self.yaw_rate_resolution = 0.034907  # 0.1 * math.pi / 180.0 [rad/s]

        self.dt = 0.1                  # time for motion prediction
        self.predict_time = 3.0

        self.to_goal_cost_gain = 5.0
        self.obstacle_cost_gain = 1.0

        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked

        # rectangle robot
        self.robot_width = 0.5           # [m] for collision check
        self.robot_length = 1.2          # [m] for collision check
        
    
    
    def calc_dynamic_window(self, self_state):
        """
        calculation dynamic window based on current state
        state contains: [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        """

        # Dynamic window from robot specification
        Vs = [self.min_speed, self.max_speed,
              -self.max_yaw_rate, self.max_yaw_rate]

        # Dynamic window from motion model
        Vd = [self_state[3] - self.max_accel * self.dt,
              self_state[3] + self.max_accel * self.dt,
              self_state[4] - self.max_delta_yaw_rate * self.dt,
              self_state[4] + self.max_delta_yaw_rate * self.dt]

        v_min = max(Vs[0], Vd[0])
        v_max = min(Vs[1], Vd[1])
        yaw_min = max(Vs[2], Vd[2])
        yaw_max = min(Vs[3], Vd[3])

        dw = [v_min, v_max, yaw_min, yaw_max]

        return dw
    

    def motion(self_state, u, dt):
        """
        u: input [forward speed, yaw_rate]
        """
        self_state[2] += u[1] * dt
        self_state[0] += u[0] * math.cos(self_state[2]) * dt
        self_state[1] += u[0] * math.sin(self_state[2]) * dt
        # velocity and angular velocity
        self_state[3] = u[0]
        self_state[4] = u[1]

        return self_state
    

    def predict_trajectory(self, init_state, v, angular_v):
        '''
        Input: vectorized grids for faster computation
        init state entries: [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        Return: array of trajectory, with shape: N * time_step * state_entry 
        '''
        n_trajectories = v.size
        time_steps = int(self.predict_time / self.dt) + 1
        trajectories = np.zeros((n_trajectories, time_steps, 5))

        trajectories[:, 0, :] = init_state
        for t in range(1, time_steps):
            # compute heading angle first
            trajectories[:, t, 2] = trajectories[:, t-1, 2] + angular_v * self.dt
            # (x, y) position with updated heading
            trajectories[:, t, 0] = trajectories[:, t-1, 0] + v * np.cos(trajectories[:, t, 2]) * self.dt
            trajectories[:, t, 1] = trajectories[:, t-1, 1] + v * np.sin(trajectories[:, t, 2]) * self.dt

            trajectories[:, t, 3] = v
            trajectories[:, t, 4] = angular_v

        return trajectories


    def predict(self, self_state, human_states, goal):
        """
        Produce action for agent with dynamic window approach.
        self_state: 5 entries with [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        goal: [x(m), y(m)] coordinate for goal position
        """
        dw = self.calc_dynamic_window(self_state)

        # Evaluate all trajectory with sampled input in dynamic window
        v_range = np.arange(dw[0], dw[1], self.v_resolution)
        angular_v_range = np.arange(dw[2], dw[3], self.yaw_rate_resolution)

        # Create a meshgrid for velocity and angular velocity (omega)
        v_grid, w_grid = np.meshgrid(v_range, angular_v_range)
        v_grid = v_grid.flatten()
        w_grid = w_grid.flatten()

        trajectories = self.predict_trajectory(init_state=self_state, v=v_grid, angular_v=w_grid)
        n_trajectories = trajectories.shape[0]

        # Calculate costs
        to_goal_costs = self.to_goal_cost_gain * np.array([self.calc_to_goal_cost(trajectories[i], goal) for i in range(n_trajectories)])

        ob_costs = 0.
        if human_states is not None:
            human_states = human_states[:, :2]
            ob_costs = self.obstacle_cost_gain * np.array([self.calc_obstacle_cost(trajectories[i], human_states) for i in range(n_trajectories)])
        
        # (speed and direction inferred with gpt are included in self attributes)
        final_costs = to_goal_costs + ob_costs

        # Find the trajectory with the minimum cost
        min_cost_idx = np.argmin(final_costs)
        min_cost = final_costs[min_cost_idx]

        best_u = [v_grid[min_cost_idx], w_grid[min_cost_idx]]
        best_trajectory = trajectories[min_cost_idx]

        # to ensure the robot do not get stuck in best v=0 m/s (in front of an obstacle) and
        # best omega=0 rad/s (heading to the goal with angle difference of 0)
        if abs(best_u[0]) < self.robot_stuck_flag_cons and abs(self_state[3]) < self.robot_stuck_flag_cons:
            best_u[1] = - self.max_delta_yaw_rate

        return best_u[0], best_u[1]  # (v,w) // input u: velocity and angular velocity (yaw rate)


    def calc_obstacle_cost(self, trajectory, human_states):
        """
        calc obstacle cost, inf: collision
        human_states: (person_num * 2) array for human [x, y] coordinates
        """
        ox, oy = [], []
        for other_human_state in human_states:
            ox.append(other_human_state[0])
            oy.append(other_human_state[1])

        ox = np.array(ox)
        oy = np.array(oy)
        dx = trajectory[:, 0] - ox[:, None]
        dy = trajectory[:, 1] - oy[:, None]
        r = np.hypot(dx, dy)   # equivalent to sqrt(x1**2 + x2**2), element-wise

        # assume human radius is 0.5m
        if np.array(r <= max(self.robot_length, self.robot_width) + 0.5).any():
            return float("Inf")

        min_r = np.min(r)
        return 1.0 / min_r  # larger distance to human means lower cost
    

    def calc_to_goal_cost(self, trajectory, goal):
        """
        goal cost with angle difference
        """

        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

        return cost
    
