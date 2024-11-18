# src/ping_pong_env.py

import gym
from gym import spaces
import numpy as np
import logging

class PingPongEnv(gym.Env):
    """
    Custom Environment for a simplified Ping Pong game.

    The environment consists of a grid where an agent (goalkeeper) tries to intercept a moving ball.
    The agent can move up, down, or stay in place to intercept the ball before it enters the goal.
    
    Observation Space:
        - agent_position: Discrete space indicating the vertical position of the agent.
        - ball_position: Discrete space indicating the horizontal position of the ball.
        - ball_shape: Discrete space indicating the shape of the ball (e.g., Circle or Square).
    
    Action Space:
        - 0: Move Up
        - 1: Move Down
        - 2: Stay
    
    Reward Structure:
        - +1: Successful interception of the ball.
        - -1: Failed interception; the ball enters the goal.
        - 0: Neutral outcome.
    """

    def __init__(self, grid_size=10, use_ontology=False):
        """
        Initialize the Ping Pong environment.

        Parameters:
        - grid_size (int): The size of the grid (number of columns). Also used to set grid_rows.
        - use_ontology (bool): Flag to determine if ontology is used (for agent enhancements).
        """
        super(PingPongEnv, self).__init__()
        self.grid_size = grid_size
        self.grid_cols = grid_size
        self.grid_rows = grid_size  # Dynamically set grid_rows based on grid_size

        # Define action space: 0 = Move Up, 1 = Move Down, 2 = Stay
        self.action_space = spaces.Discrete(3)

        # Define observation space
        # - agent_position: Discrete from 0 to grid_rows - 1
        # - ball_position: Discrete from 0 to grid_cols - 1
        # - ball_shape: Discrete (e.g., 0 = Circle, 1 = Square)
        self.observation_space = spaces.Dict({
            'agent_position': spaces.Discrete(self.grid_rows),
            'ball_position': spaces.Discrete(self.grid_cols),
            'ball_shape': spaces.Discrete(2)
        })

        # Initialize state variables
        self.agent_position = self.grid_rows // 2  # Start agent in the middle vertically
        self.ball_position = np.random.randint(0, self.grid_cols)  # Random horizontal position
        self.ball_shape = np.random.randint(0, 2)  # Random shape: 0 or 1
        self.ball_y_position = np.random.randint(0, self.grid_rows)  # Random vertical position

        self.use_ontology = use_ontology  # Flag for ontology usage

        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def reset(self):
        """
        Reset the environment to an initial state and return the initial observation.

        Returns:
        - observation (dict): The initial observation of the environment.
        """
        self.agent_position = self.grid_rows // 2
        self.ball_position = np.random.randint(0, self.grid_cols)
        self.ball_shape = np.random.randint(0, 2)
        self.ball_y_position = np.random.randint(0, self.grid_rows)
        return {
            'agent_position': self.agent_position,
            'ball_position': self.ball_position,
            'ball_shape': self.ball_shape
        }

    def step(self, action):
        """
        Execute one time step within the environment.

        Parameters:
        - action (int): The action taken by the agent.

        Returns:
        - observation (dict): The next observation of the environment.
        - reward (float): The reward obtained from taking the action.
        - done (bool): Whether the episode has ended.
        - info (dict): Additional information.
        """
        done = False
        reward = 0

        # Update agent position based on action
        if action == 0:  # Move Up
            if self.agent_position > 0:
                self.agent_position -= 1
                logging.debug(f"Agent moves up to position {self.agent_position}.")
            else:
                logging.debug("Agent is already at the top boundary.")
        elif action == 1:  # Move Down
            if self.agent_position < self.grid_rows - 1:
                self.agent_position += 1
                logging.debug(f"Agent moves down to position {self.agent_position}.")
            else:
                logging.debug("Agent is already at the bottom boundary.")
        elif action == 2:  # Stay
            logging.debug("Agent stays in the current position.")
        else:
            logging.warning(f"Received invalid action: {action}. Action ignored.")

        # Move the ball horizontally towards the agent
        self.ball_position -= 1  # Ball moves left towards the agent
        logging.debug(f"Ball moves to position {self.ball_position}.")

        # Randomly update ball_y_position (vertical movement)
        # Ball can move up, down, or stay in the same vertical position
        y_move = np.random.choice([-1, 0, 1])
        self.ball_y_position += y_move
        # Ensure ball_y_position stays within grid boundaries
        self.ball_y_position = max(0, min(self.ball_y_position, self.grid_rows - 1))
        logging.debug(f"Ball vertical position updated to {self.ball_y_position}.")

        # Check if the ball has reached the agent's side
        if self.ball_position == 0:
            # Check for interception: agent can intercept if ball_y_position is at agent_position or agent_position +1
            if self.agent_position == self.ball_y_position or self.agent_position + 1 == self.ball_y_position:
                reward += 1  # Successful interception
                logging.info("Agent intercepted the ball successfully.")
            else:
                reward -= 1  # Failed interception
                logging.info("Ball entered the goal. Agent failed to intercept.")
            done = True  # Episode ends after the ball reaches the agent's side
        elif self.ball_position < 0:
            # Ball has gone past the agent without interception
            reward -= 1  # Negative reward for missing the ball
            logging.info("Ball missed the agent and went out of bounds.")
            done = True
        else:
            # Ball is still in play
            logging.debug("Ball is still in play.")
            done = False

        # Update ball_shape randomly for the next step if the episode continues
        if not done:
            self.ball_shape = np.random.randint(0, 2)

        # Prepare the next observation
        observation = {
            'agent_position': self.agent_position,
            'ball_position': self.ball_position,
            'ball_shape': self.ball_shape
        }

        return observation, reward, done, {}

    def render(self, mode='human'):
        """
        Render the current state of the environment.

        Parameters:
        - mode (str): The mode to render with. Currently supports only 'human'.
        """
        if mode != 'human':
            raise NotImplementedError("Currently, only 'human' rendering mode is supported.")

        # Create an empty grid
        grid = [[' ' for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]

        # Place the agent on the grid
        grid[self.agent_position][0] = 'A'  # Agent represented by 'A'

        # For visualization purposes, the agent could occupy multiple cells if desired
        # Uncomment below lines if the agent should occupy two vertical positions
        # if self.agent_position + 1 < self.grid_rows:
        #     grid[self.agent_position + 1][0] = 'A'

        # Place the ball on the grid
        if 0 <= self.ball_position < self.grid_cols and 0 <= self.ball_y_position < self.grid_rows:
            shape = 'O' if self.ball_shape == 0 else 'S'  # 'O' for Circle, 'S' for Square
            grid[self.ball_y_position][self.ball_position] = shape

        # Display the grid
        horizontal_border = '-' * (self.grid_cols * 2 + 1)
        print(horizontal_border)
        for row in grid:
            print('|' + ' '.join(row) + '|')
        print(horizontal_border)
