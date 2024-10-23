import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BatteryEnv(gym.Env):
    def __init__(self):
        super(BatteryEnv, self).__init__()

        # Define the number of battery cells
        self.num_cells = 5

        # Time step (Δt) for discrete updates
        self.time_step = 1

        # Total battery capacity (in milliamp-hours, mAh)
        self.total_capacity = 3400  # Example: 3400 mAh

        # Coulombic efficiency (η), assumed to be 1 for discharge and <= 1 for charge
        self.coulomb_efficiency = 1.0

        # Initialize internal resistance (R0) and RC branch parameters (Ri, Ci)
        self.R0 = np.array([0.1] * self.num_cells)  # Internal resistance for each cell
        self.Ri = np.array([0.03] * self.num_cells)  # Resistances in the RC branches
        self.Ci = np.array([750] * self.num_cells)  # Capacitors in the RC branches

        # Define action and observation spaces
        # Action space: Each cell has a binary action (0 = switch off, 1 = switch on)
        self.action_space = spaces.MultiBinary(self.num_cells)

        # Observation space: SoC (State of Charge) and switch state for each cell
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_cells, 2), dtype=np.float32)

        # Initialize the state of charge (SoC) for each cell and switch states
        self.soc = np.random.uniform(0.4, 0.6, self.num_cells)  # Random SoC between 40% and 60%
        self.switch_state = np.zeros(self.num_cells)  # All switches initially off (0)

        # Maximum steps per episode
        self.max_steps = 100  # Set a maximum number of steps per episode
        self.current_step = 0  # Track the number of steps in the current episode

    def reset(self, seed=None, options=None):
        # Reset the environment to its initial state at the start of each episode
        super().reset(seed=seed)
        self.soc = np.random.uniform(0.4, 0.6, self.num_cells)  # Reset SoC for each cell
        self.switch_state = np.zeros(self.num_cells)  # Reset all switches to off
        self.current_step = 0  # Reset time step counter
        return self._get_obs(), {}

    def step(self, action):
        # Apply the action from the agent and update the environment state
        self.switch_state = action

        # Update the state of charge (SoC) for each cell based on whether its switch is on or off
        for i in range(self.num_cells):
            if action[i] == 1:  # If the switch for the cell is on
                current = self._calculate_current(i)  # Calculate the current using the RC model
                # Update SoC using the formula: ΔSoC = -(η * Δt / Q) * i, where i is the current
                self.soc[i] -= (self.coulomb_efficiency * self.time_step / self.total_capacity) * current

        # Variance of SoC (penalty for unbalanced SoC)
        soc_variance = np.var(self.soc)

        # Reward function based on the paper's recommendation:
        reward = (-np.log(soc_variance + 1e-6) - 4) / 2  # Penalty for SoC variance, logarithmic for stability
        reward -= 4 * np.sum(action)  # Penalty for switching too many cells

        # Positive reward for keeping the SoC in the desired range
        if np.all(self.soc > 0.4) and np.all(self.soc < 0.6):
            reward += 0.5  # Positive reward for keeping the SoC in the desired range

        # Check for terminal conditions (SoC out of bounds)
        done = np.any(self.soc <= 0) or np.any(self.soc >= 1)

        # Increment the step counter and check if we have reached the maximum step limit
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, done, truncated, {}

    def _calculate_current(self, i):
        # Calculate the current for a given cell using the RC branch model
        exp_term = np.exp(-1 / (self.Ri[i] * self.Ci[i]))  # Exponential decay term
        ir_next = (1 - exp_term) * (self.soc[i] - exp_term * self.Ri[i])  # Next current in the RC branch
        return ir_next  # Return the calculated current

    def _get_obs(self):
        # Return the current observation, including SoC and switch states
        return np.column_stack((self.soc, self.switch_state))  # Combine SoC and switch state into a single observation

    def render(self, mode='human'):
        # Render the current state of the environment (for debugging or monitoring)
        print(f'SoC: {self.soc}, Switches: {self.switch_state}')  # Display the SoC and switch states