# ping_pong_env.py

import gym
from gym import spaces
import numpy as np
import random

class PingPongEnv(gym.Env):
    """
    Enhanced Ping Pong Environment with Goalkeeper Mechanics
    """
    def __init__(self):
        super(PingPongEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0: Stay, 1: Move Up, 2: Move Down
        # Define observation space with agent and ball positions
        self.observation_space = spaces.Dict({
            'agent_position': spaces.Discrete(5),  # Vertical positions: 0 to 4
            'ball_position': spaces.Discrete(10),  # Horizontal positions: 0 to 9
            'ball_shape': spaces.Discrete(2)       # 0: Circle, 1: Square
        })
        self.reset()

    def reset(self):
        self.agent_position = 2  # Start at the middle vertical position
        self.ball_position = random.randint(7, 9)  # Ball starts near the right side
        self.ball_shape = random.randint(0,1)      # Random shape
        self.ball_y_position = random.randint(0,4) # Ball's vertical position
        self.episode_history = []  # To store positions for visualization
        self._record_history()
        return self._get_obs()

    def step(self, action):
        done = False
        reward = 0

        # Update agent position based on action
        if action == 1 and self.agent_position > 0:
            self.agent_position -= 1  # Move Up
        elif action == 2 and self.agent_position < 4:
            self.agent_position += 1  # Move Down
        # action == 0: Stay

        # Move the ball towards the agent
        self.ball_position -= 1

        # Check if ball reaches the agent's column (left side, column 0)
        if self.ball_position == 0:
            # Check if agent is aligned vertically with the ball
            # Considering the agent spans two vertical positions for width
            agent_span = [self.agent_position, self.agent_position + 1] if self.agent_position < 4 else [self.agent_position -1, self.agent_position]
            if self.ball_y_position in agent_span:
                # Agent successfully stops the ball
                reward = 2
            else:
                # Ball enters the goal
                reward = -2
            done = True

        # Record current positions
        self._record_history()

        # Prepare next ball if episode is done
        if done:
            self.agent_position = 2  # Reset agent to starting vertical position
            self.ball_position = random.randint(7,9)  # Reset ball position near the right
            self.ball_shape = random.randint(0,1)      # Reset ball shape
            self.ball_y_position = random.randint(0,4) # Reset ball's vertical position
            self._record_history()

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        grid_rows = 5
        grid_cols = 10
        grid = [['.' for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        # Place the agent (Goalkeeper) on the left side (column 0)
        # Agent spans two vertical positions for width
        if self.agent_position < 4:
            grid[self.agent_position][0] = 'G'  # Top part of Goalkeeper
            grid[self.agent_position + 1][0] = 'G'  # Bottom part of Goalkeeper
        else:
            grid[self.agent_position -1][0] = 'G'  # Top part
            grid[self.agent_position][0] = 'G'    # Bottom part

        # Place the ball on its current position
        if 0 <= self.ball_position < grid_cols:
            shape = 'O' if self.ball_shape == 0 else 'S'  # O: Circle, S: Square
            grid[self.ball_y_position][self.ball_position] = shape

        # Display the grid
        print("Grid:")
        for row in grid:
            print(' '.join(row))
        print()

    def _record_history(self):
        self.episode_history.append({
            'agent_position': self.agent_position,
            'ball_position': self.ball_position,
            'ball_y_position': self.ball_y_position,
            'ball_shape': self.ball_shape
        })

    def get_episode_history(self):
        return self.episode_history

    def _get_obs(self):
        return {
            'agent_position': self.agent_position,
            'ball_position': self.ball_position,
            'ball_shape': self.ball_shape
        }
