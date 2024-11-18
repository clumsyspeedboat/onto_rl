# src/q_learning_agent.py

import numpy as np
import random
import logging


class QLearningAgent:
    """
    Standard Q-Learning Agent for the Ping Pong environment.
    """

    def __init__(self, action_size, state_size, learning_rate=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initialize the Q-Learning agent.

        Parameters:
        - action_size (int): Number of possible actions.
        - state_size (tuple): Dimensions of the state space.
        - learning_rate (float): Learning rate (alpha).
        - gamma (float): Discount factor.
        - epsilon (float): Initial exploration rate.
        - epsilon_min (float): Minimum exploration rate.
        - epsilon_decay (float): Decay rate for exploration probability.
        """
        self.action_size = action_size
        self.state_size = state_size  # Tuple: (agent_position, ball_position, ball_shape)
        self.q_table = np.zeros(state_size + (action_size,))
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        logging.info(f"Initialized Standard Q-Learning Agent with state size {self.state_size} and action size {self.action_size}")

    def choose_action(self, state):
        """
        Choose an action based on the current state using an epsilon-greedy policy.

        Parameters:
        - state (tuple): Current state.

        Returns:
        - action (int): Chosen action.
        """
        if random.uniform(0, 1) < self.epsilon:
            action = random.randrange(self.action_size)  # Explore: random action
            logging.debug(f"Agent explores and chooses action {action}")
            return action
        else:
            action = np.argmax(self.q_table[state])       # Exploit: best action
            logging.debug(f"Agent exploits and chooses action {action}")
            return action

    def learn(self, state, action, reward, next_state, done):
        """
        Update the Q-table based on the agent's experience.

        Parameters:
        - state (tuple): Previous state.
        - action (int): Action taken.
        - reward (float): Reward received.
        - next_state (tuple): Current state after action.
        - done (bool): Whether the episode has ended.
        """
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state])

        # Q-learning update rule
        old_value = self.q_table[state + (action,)]
        self.q_table[state + (action,)] += self.learning_rate * (target - self.q_table[state + (action,)])
        logging.debug(f"Updated Q-table at state {state} and action {action}: {old_value} -> {self.q_table[state + (action,)]}")

        # Decay epsilon
        if done:
            if self.epsilon > self.epsilon_min:
                old_epsilon = self.epsilon
                self.epsilon *= self.epsilon_decay
                logging.debug(f"Decayed epsilon from {old_epsilon} to {self.epsilon}")