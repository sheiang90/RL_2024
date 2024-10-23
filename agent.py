import numpy as np
import gymnasium as gym
from battery_env import BatteryEnv  # Assuming the environment is in a file named 'battery_env.py'


class RandomAgent:
    def __init__(self, env: gym.Env):
        """
        Initialize the RandomAgent with the environment.

        :param env: The Gym environment the agent will interact with.
        """
        self.env = env
        self.action_space = env.action_space

    def get_action(self):
        """
        Get a random action from the environment's action space.

        :return: A random action.
        """
        return self.action_space.sample()

    def run_episode(self):
        """
        Run a single episode in the environment, taking random actions.
        """
        # Reset the environment at the start of the episode
        observation, info = self.env.reset()

        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Get a random action
            action = self.get_action()

            # Take the action in the environment
            observation, reward, done, truncated, info = self.env.step(action)

            # Accumulate the total reward and count the steps
            total_reward += reward
            steps += 1

            # Optionally render the environment (useful for debugging)
            self.env.render()

        print(f"Episode finished after {steps} steps with total reward: {total_reward}")


if __name__ == "__main__":
    # Create the environment (assuming BatteryEnv is defined in 'battery_env.py')
    env = BatteryEnv()

    # Create the random agent
    agent = RandomAgent(env)

    # Run one episode
    agent.run_episode()
