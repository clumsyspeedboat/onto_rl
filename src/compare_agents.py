# src/compare_agents.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from ping_pong_env import PingPongEnv
from q_learning_agent import QLearningAgent
from ontology_enhanced_agent import OntologyEnhancedAgent
from ontology_definition import create_ontology
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

def train_agent(agent, env, episodes, use_ontology=False):
    """
    Train a Q-Learning agent.

    Parameters:
    - agent (QLearningAgent or OntologyEnhancedAgent): The agent to train.
    - env (PingPongEnv): The environment.
    - episodes (int): Number of training episodes.
    - use_ontology (bool): Whether the agent uses ontology.

    Returns:
    - rewards (list): Total rewards per episode.
    - steps (list): Total steps per episode.
    """
    rewards = []
    steps = []
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            if use_ontology:
                agent_pos = state['agent_position']
                ball_pos = state['ball_position']
                ball_shape = state['ball_shape']
                ball_y_pos = env.ball_y_position
                enriched_state = (agent_pos, ball_pos, ball_shape, ball_y_pos)
                action = agent.choose_action(enriched_state)
                next_state, reward, done, _ = env.step(action)
                enriched_next_state = (
                    next_state['agent_position'],
                    next_state['ball_position'],
                    next_state['ball_shape'],
                    env.ball_y_position
                )
                agent.learn(enriched_state, action, reward, enriched_next_state, done)
            else:
                agent_pos = state['agent_position']
                ball_pos = state['ball_position']
                ball_shape = state['ball_shape']
                enriched_state = (agent_pos, ball_pos, ball_shape)
                action = agent.choose_action(enriched_state)
                next_state, reward, done, _ = env.step(action)
                enriched_next_state = (
                    next_state['agent_position'],
                    next_state['ball_position'],
                    next_state['ball_shape']
                )
                agent.learn(enriched_state, action, reward, enriched_next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1
        rewards.append(total_reward)
        steps.append(step_count)
        if episode % 100 == 0:
            logging.info(f"Episode {episode}: Total Reward: {total_reward}, Steps: {step_count}, Epsilon: {agent.epsilon:.4f}")
    return rewards, steps

def main():
    """
    Main function to train and compare agents.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("onto_rl.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'config.yaml')
    config = load_config(config_path)

    grid_size = config['grid_size']
    episodes = config['episodes']
    test_episodes = config['test_episodes']
    window_size = config['moving_average_window']

    # Ensure ontology exists
    ontology_path = os.path.join(script_dir, '..', 'ontology', 'pingpong.owl')
    if not os.path.exists(ontology_path):
        logging.info("Ontology not found. Creating ontology...")
        create_ontology()
    else:
        logging.info("Ontology already exists. Skipping creation.")

    # Load the ontology
    from owlready2 import get_ontology

    try:
        onto = get_ontology("http://example.org/pingpong.owl").load()
        logging.info("Ontology loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load ontology: {e}")
        sys.exit(1)

    # Initialize environments
    env_with_ontology = PingPongEnv(grid_size=grid_size, use_ontology=True)
    env_without_ontology = PingPongEnv(grid_size=grid_size, use_ontology=False)

    # Define state sizes
    state_size_standard = (
        env_without_ontology.observation_space['agent_position'].n,  # 5 vertical positions
        env_without_ontology.observation_space['ball_position'].n,   # 10 horizontal positions
        env_without_ontology.observation_space['ball_shape'].n       # 2 ball shapes
    )

    state_size_ontology = (
        env_with_ontology.observation_space['agent_position'].n,    # 5 vertical positions
        env_with_ontology.observation_space['ball_position'].n,     # 10 horizontal positions
        env_with_ontology.observation_space['ball_shape'].n,        # 2 ball shapes
        5                                                        # ball_y_position: 0 to 4
    )

    # Initialize agents
    agent_with_ontology = OntologyEnhancedAgent(
        action_size=env_with_ontology.action_space.n,
        state_size=state_size_ontology,
        ontology=onto,
        learning_rate=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )

    agent_without_ontology = QLearningAgent(
        action_size=env_without_ontology.action_space.n,
        state_size=state_size_standard,
        learning_rate=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )

    # Define paths for saving Q-tables
    models_dir = os.path.join(config['paths']['models'])
    os.makedirs(models_dir, exist_ok=True)
    standard_qtable_path = os.path.join(models_dir, 'standard_agent_qtable.pkl')
    ontology_qtable_path = os.path.join(models_dir, 'ontology_agent_qtable.pkl')

    # Train Standard Agent
    logging.info("\nTraining the Standard Agent (Without Ontology)...")
    rewards_without, steps_without = train_agent(agent_without_ontology, env_without_ontology, episodes, use_ontology=False)

    # Train Ontology-Enhanced Agent
    logging.info("\nTraining the Ontology-Enhanced Agent (With Ontology)...")
    rewards_with, steps_with = train_agent(agent_with_ontology, env_with_ontology, episodes, use_ontology=True)

    # Plotting the results
    plt.figure(figsize=(16, 12))

    # Total Reward Plot
    plt.subplot(2, 1, 1)
    plt.plot(rewards_with, label='With Ontology', alpha=0.7)
    plt.plot(rewards_without, label='Without Ontology', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.grid(True)
    # Plot moving average
    ma_with = moving_average(rewards_with, window_size=window_size)
    ma_without = moving_average(rewards_without, window_size=window_size)
    plt.plot(range(window_size - 1, episodes), ma_with, label='With Ontology (MA)', color='blue')
    plt.plot(range(window_size - 1, episodes), ma_without, label='Without Ontology (MA)', color='orange')
    plt.legend()

    # Steps Plot
    plt.subplot(2, 1, 2)
    plt.plot(steps_with, label='With Ontology', alpha=0.7)
    plt.plot(steps_without, label='Without Ontology', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')
    plt.legend()
    plt.grid(True)
    # Plot moving average
    ma_steps_with = moving_average(steps_with, window_size=window_size)
    ma_steps_without = moving_average(steps_without, window_size=window_size)
    plt.plot(range(window_size - 1, episodes), ma_steps_with, label='With Ontology (MA)', color='blue')
    plt.plot(range(window_size - 1, episodes), ma_steps_without, label='Without Ontology (MA)', color='orange')
    plt.legend()

    plt.tight_layout()

    # Save plot to assets
    assets_dir = os.path.join(config['paths']['assets'])
    os.makedirs(assets_dir, exist_ok=True)
    plot_path = os.path.join(assets_dir, 'agent_performance_comparison.png')
    plt.savefig(plot_path)
    logging.info(f"Performance comparison plot saved to {plot_path}")
    plt.show()

    # Save Q-tables
    with open(standard_qtable_path, 'wb') as f:
        pickle.dump(agent_without_ontology.q_table, f)
    logging.info(f"Standard Agent Q-table saved to {standard_qtable_path}")

    with open(ontology_qtable_path, 'wb') as f:
        pickle.dump(agent_with_ontology.q_table, f)
    logging.info(f"Ontology-Enhanced Agent Q-table saved to {ontology_qtable_path}")

    # Testing both agents
    logging.info("\nTesting the Standard Agent (Without Ontology)...")
    for ep in range(1, test_episodes + 1):
        state = env_without_ontology.reset()
        done = False
        total_reward = 0
        logging.info(f"Test Episode {ep}:")
        while not done:
            agent_pos = state['agent_position']
            ball_pos = state['ball_position']
            ball_shape = state['ball_shape']
            enriched_state = (agent_pos, ball_pos, ball_shape)
            action = agent_without_ontology.choose_action(enriched_state)
            next_state, reward, done, _ = env_without_ontology.step(action)
            state = next_state
            total_reward += reward
            env_without_ontology.render()
        logging.info(f"Total Reward: {total_reward}\n")

    logging.info("Testing the Ontology-Enhanced Agent (With Ontology)...")
    for ep in range(1, test_episodes + 1):
        state = env_with_ontology.reset()
        done = False
        total_reward = 0
        logging.info(f"Test Episode {ep}:")
        while not done:
            agent_pos = state['agent_position']
            ball_pos = state['ball_position']
            ball_shape = state['ball_shape']
            ball_y_pos = env_with_ontology.ball_y_position
            enriched_state = (agent_pos, ball_pos, ball_shape, ball_y_pos)
            action = agent_with_ontology.choose_action(enriched_state)
            next_state, reward, done, _ = env_with_ontology.step(action)
            state = next_state
            total_reward += reward
            env_with_ontology.render()
        logging.info(f"Total Reward: {total_reward}\n")

    logging.info("Training and testing completed.")

if __name__ == "__main__":
    main()
