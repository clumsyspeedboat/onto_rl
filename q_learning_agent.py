# q_learning_agent.py

import numpy as np
import random

class QLearningAgent:
    def __init__(self, action_size, state_size, learning_rate=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.action_size = action_size
        self.state_size = state_size  # Tuple: (agent_position, ball_position, ball_shape)
        self.q_table = np.zeros(state_size + (action_size,))
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount rate
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):
        if random.uniform(0,1) < self.epsilon:
            return random.randrange(self.action_size)  # Explore
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state + (action,)] += self.learning_rate * (target - self.q_table[state + (action,)])

        if done:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
