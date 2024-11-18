# src/visualize_agents.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from ping_pong_env import PingPongEnv
from q_learning_agent import QLearningAgent
from ontology_enhanced_agent import OntologyEnhancedAgent
from owlready2 import get_ontology
import pickle
import logging
import yaml


def load_config(config_path):
    """
    Load configuration from a YAML file.

    Parameters:
    - config_path (str): Path to the configuration file.

    Returns:
    - config (dict): Configuration parameters.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def load_agents(models_dir, ontology_path):
    """
    Load pre-trained agents from saved Q-tables.

    Parameters:
    - models_dir (str): Directory where Q-tables are saved.
    - ontology_path (str): Path to the ontology file.

    Returns:
    - standard_agent (QLearningAgent): Trained Standard Q-Learning Agent.
    - ontology_agent (OntologyEnhancedAgent): Trained Ontology-Enhanced Q-Learning Agent.
    - env (PingPongEnv): The Ping Pong environment.
    - env_with_ontology (PingPongEnv): The Ping Pong environment with ontology.
    """
    env = PingPongEnv(grid_size=10, use_ontology=False)
    env_with_ontology = PingPongEnv(grid_size=10, use_ontology=True)

    # Define state sizes
    state_size_standard = (
        env.observation_space['agent_position'].n,  # 10 vertical positions
        env.observation_space['ball_position'].n,   # 10 horizontal positions
        env.observation_space['ball_shape'].n       # 2 ball shapes
    )

    state_size_ontology = (
        env_with_ontology.observation_space['agent_position'].n,  # 10 vertical positions
        env_with_ontology.observation_space['ball_position'].n,   # 10 horizontal positions
        env_with_ontology.observation_space['ball_shape'].n,      # 2 ball shapes
        env_with_ontology.grid_rows                                # ball_y_position based on grid_rows
    )

    # Initialize agents
    try:
        ontology_uri = 'file:///' + os.path.abspath(ontology_path).replace('\\', '/')
        onto = get_ontology(ontology_uri).load()
        logging.info("Ontology loaded successfully from local file.")
    except Exception as e:
        logging.error(f"Failed to load ontology: {e}")
        sys.exit(1)

    standard_agent = QLearningAgent(action_size=env.action_space.n, state_size=state_size_standard)
    ontology_agent = OntologyEnhancedAgent(
        action_size=env_with_ontology.action_space.n,
        state_size=state_size_ontology,
        ontology=onto,
        learning_rate=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )

    # Define Q-table paths
    standard_qtable_path = os.path.join(models_dir, 'standard_agent_qtable.pkl')
    ontology_qtable_path = os.path.join(models_dir, 'ontology_agent_qtable.pkl')

    # Load Q-tables
    try:
        with open(standard_qtable_path, 'rb') as f:
            standard_agent.q_table = pickle.load(f)
        logging.info(f"Standard Agent Q-table loaded from {standard_qtable_path}")
    except FileNotFoundError:
        logging.error(f"Standard Agent Q-table not found at {standard_qtable_path}. Please train the agent first.")
        sys.exit(1)

    try:
        with open(ontology_qtable_path, 'rb') as f:
            ontology_agent.q_table = pickle.load(f)
        logging.info(f"Ontology-Enhanced Agent Q-table loaded from {ontology_qtable_path}")
    except FileNotFoundError:
        logging.error(f"Ontology-Enhanced Agent Q-table not found at {ontology_qtable_path}. Please train the agent first.")
        sys.exit(1)

    return standard_agent, ontology_agent, env, env_with_ontology


def run_test_episode(agent, env, use_ontology=False):
    """
    Run a single test episode and record the history.

    Parameters:
    - agent (QLearningAgent or OntologyEnhancedAgent): The agent to test.
    - env (PingPongEnv): The environment.
    - use_ontology (bool): Whether the agent uses ontology.

    Returns:
    - history (list): Recorded history of states during the episode.
    """
    state = env.reset()
    done = False
    history = []

    while not done:
        # Record state with 'ball_y_position'
        state_record = state.copy()
        state_record['ball_y_position'] = env.ball_y_position
        history.append(state_record)

        if use_ontology:
            agent_pos = state['agent_position']
            ball_pos = state['ball_position']
            ball_shape = state['ball_shape']
            ball_y_pos = env.ball_y_position
            enriched_state = (agent_pos, ball_pos, ball_shape, ball_y_pos)
            action = agent.choose_action(enriched_state)
        else:
            agent_pos = state['agent_position']
            ball_pos = state['ball_position']
            ball_shape = state['ball_shape']
            enriched_state = (agent_pos, ball_pos, ball_shape)
            action = agent.choose_action(enriched_state)

        next_state, reward, done, _ = env.step(action)
        state = next_state
    # Append final state
    state_record = state.copy()
    state_record['ball_y_position'] = env.ball_y_position
    history.append(state_record)

    return history


def create_animation(agent_history, agent_type='Standard Agent'):
    """
    Create an animation from the episode history.

    Parameters:
    - agent_history (list): History of states during the episode.
    - agent_type (str): Type of the agent for labeling.

    Returns:
    - anim (FuncAnimation): Matplotlib animation object.
    """
    agent_positions = [step['agent_position'] for step in agent_history]
    ball_positions = [step['ball_position'] for step in agent_history]
    ball_shapes = [step['ball_shape'] for step in agent_history]
    ball_y_positions = [step['ball_y_position'] for step in agent_history]

    # Determine grid size based on maximum positions
    grid_rows = max(max(agent_positions), max(ball_y_positions)) + 1
    grid_cols = max(ball_positions) + 1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-1, grid_cols)
    ax.set_ylim(-1, grid_rows)
    ax.set_xticks(range(0, grid_cols + 1))
    ax.set_yticks(range(0, grid_rows + 1))
    ax.set_xlabel('Horizontal Position')
    ax.set_ylabel('Vertical Position')
    ax.set_title(f'Ping Pong Agent Performance: {agent_type}')

    # Plot agent and ball
    agent_dot_top, = ax.plot([], [], 'bs', markersize=20, label='Goalkeeper Top')      # Blue square
    agent_dot_bottom, = ax.plot([], [], 'bs', markersize=20, label='Goalkeeper Bottom')# Blue square
    ball_dot, = ax.plot([], [], 'ro', markersize=12, label='Ball')                     # Red circle or square
    ball_shape_text = ax.text(0, 0, '', fontsize=12, ha='center', va='center')

    ax.legend(loc='upper right')

    # Add a goal line on the left
    goal_line = ax.axvline(x=0, color='green', linestyle='--', label='Goal Line')

    def init():
        agent_dot_top.set_data([], [])
        agent_dot_bottom.set_data([], [])
        ball_dot.set_data([], [])
        ball_shape_text.set_text('')
        return agent_dot_top, agent_dot_bottom, ball_dot, ball_shape_text

    def animate(i):
        if i < len(agent_positions):
            agent_pos = agent_positions[i]
            ball_pos = ball_positions[i]
            ball_y_pos = ball_y_positions[i]
            shape = 'O' if ball_shapes[i] == 0 else 'S'

            # Agent spans two vertical positions: agent_pos and agent_pos +1
            agent_top = agent_pos
            agent_bottom = agent_pos + 1

            # Update agent positions
            agent_dot_top.set_data([0], [agent_top])
            agent_dot_bottom.set_data([0], [agent_bottom])

            # Ball moves horizontally with vertical position
            ball_dot.set_data([ball_pos], [ball_y_pos])

            # Display ball shape
            ball_shape_text.set_text(shape)
            ball_shape_text.set_position((ball_pos, ball_y_pos))
        return agent_dot_top, agent_dot_bottom, ball_dot, ball_shape_text

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(agent_positions), interval=500, blit=True, repeat=False)
    return anim
