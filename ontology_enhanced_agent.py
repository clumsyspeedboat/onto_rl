# ontology_enhanced_agent.py

import numpy as np
import random
from owlready2 import *

class OntologyEnhancedAgent:
    def __init__(self, action_size, state_size, ontology, learning_rate=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.action_size = action_size
        self.state_size = state_size  # Tuple: (agent_position, ball_position, ball_shape, ball_y_position)
        self.ontology = ontology
        self.q_table = np.zeros(state_size + (action_size,))
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount rate
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_ball_shape_properties(self, shape):
        # Retrieve properties from ontology
        if shape == 0:
            # Access the 'circle_obj' instance of the Circle class
            circle_instances = list(self.ontology.Circle.instances())
            if not circle_instances:
                raise ValueError("No instances of Circle found in the ontology.")
            ball = circle_instances[0]
        else:
            # Access the 'square_obj' instance of the Square class
            square_instances = list(self.ontology.Square.instances())
            if not square_instances:
                raise ValueError("No instances of Square found in the ontology.")
            ball = square_instances[0]
        
        # Access properties directly without indexing
        shape_str = ball.has_shape
        speed = ball.has_speed
        
        return shape_str, speed

    def choose_action(self, state):
        # Incorporate ontological knowledge into state
        agent_pos, ball_pos, ball_shape, ball_y_pos = state
        if random.uniform(0,1) < self.epsilon:
            action = random.randrange(self.action_size)  # Explore
            # Optionally log
            # print(f"Exploration: Action {action}")
        else:
            action = np.argmax(self.q_table[state])
            # Optionally log
            # print(f"Exploitation: Action {action}")
        return action

    def learn(self, state, action, reward, next_state, done):
        # Compute target
        target = reward
        if not done:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        # Update Q-table
        self.q_table[state + (action,)] += self.learning_rate * (target - self.q_table[state + (action,)])

        if done:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
