import numpy as np
import gymnasium as gym
from battery_env import BatteryEnv  
from torch.utils.tensorboard import SummaryWriter  
import time


class RandomAgent:
    def __init__(self, env: gym.Env, writer: SummaryWriter):
        """
        Initialize the RandomAgent with the environment and a TensorBoard writer.

        :param env: The Gym environment the agent will interact with.
        :param writer: TensorBoard writer to log data.
        """
        self.env = env
        self.action_space = env.action_space
        self.writer = writer
        self.episode = 0

    def get_action(self):
        """
        Get a random action from the environment's action space.

        :return: A random action.
        """
        return self.action_space.sample()

    def run_episode(self):
        """
        Run a single episode in the environment, taking random actions.
        Log rewards and steps to TensorBoard.
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

        # Log the total reward and steps to TensorBoard
        self.writer.add_scalar('Total Reward', total_reward, self.episode)
        self.writer.add_scalar('Steps', steps, self.episode)
        print(f"Episode {self.episode} finished after {steps} steps with total reward: {total_reward}")

        # Increment episode count
        self.episode += 1

    def train(self, num_episodes):
        """
        Train the agent by running the specified number of episodes.
        """
        for _ in range(num_episodes):
            self.run_episode()


if __name__ == "__main__":
    # Create the environment 
    env = BatteryEnv()

    # Create the TensorBoard writer to log data
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f'runs/battery_env_{timestamp}'
    writer = SummaryWriter(log_dir)

    # Create the random agent with TensorBoard logging
    agent = RandomAgent(env, writer)

    # Train the agent for 100 episodes and log the results
    agent.train(num_episodes=100)

    # Close the TensorBoard writer
    writer.close()