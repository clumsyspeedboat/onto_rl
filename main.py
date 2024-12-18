# main.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import logging
import yaml
from pathlib import Path
from owlready2 import get_ontology
from matplotlib import animation

# Add src directory to sys.path for module imports
script_dir = Path(__file__).parent.resolve()
src_dir = script_dir / 'src'
sys.path.append(str(src_dir))

from ping_pong_env import PingPongEnv
from q_learning_agent import QLearningAgent
from ontology_enhanced_agent import OntologyEnhancedAgent
from ontology_definition import create_ontology
from visualize_agents import create_animation, run_test_episode, load_agents, load_config


def moving_average(data, window_size=100):
    """
    Compute moving average to smooth the plot.

    Parameters:
    - data (list): List of numerical values.
    - window_size (int): The size of the moving window.

    Returns:
    - numpy.ndarray: Smoothed data.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


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
    Main function to orchestrate training and testing of agents.
    """
    # Setup logging
    log_file_path = script_dir / 'onto_rl.log'
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more detailed logs
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Load configuration
    config_path = script_dir / 'config.yaml'
    config = load_config(str(config_path))

    grid_size = config['grid_size']
    episodes = config['episodes']
    test_episodes = config['test_episodes']
    window_size = config['moving_average_window']

    # Define the path to the ontology file from config and ensure the directory exists
    ontology_path = Path(config['paths']['ontology']).resolve()
    ontology_dir = ontology_path.parent
    ontology_dir.mkdir(parents=True, exist_ok=True)

    # Ensure ontology exists
    if not ontology_path.exists():
        logging.info("Ontology not found. Creating ontology...")
        create_ontology()
    else:
        logging.info("Ontology already exists. Skipping creation.")

    # Load the ontology from the local file system by passing the absolute path as a string
    try:
        logging.info(f"Loading ontology from path: {ontology_path}")
        onto = get_ontology(str(ontology_path)).load()
        logging.info("Ontology loaded successfully from local file.")
    except Exception as e:
        logging.error(f"Failed to load ontology: {e}")
        sys.exit(1)

    # Initialize environments
    env_with_ontology = PingPongEnv(grid_size=grid_size, use_ontology=True)
    env_without_ontology = PingPongEnv(grid_size=grid_size, use_ontology=False)

    # Retrieve grid_rows dynamically based on grid_size
    grid_rows = env_with_ontology.grid_rows  # Ensure this attribute exists in PingPongEnv

    # Define state sizes
    state_size_standard = (
        env_without_ontology.observation_space['agent_position'].n,  # e.g., 5 vertical positions
        env_without_ontology.observation_space['ball_position'].n,   # e.g., 10 horizontal positions
        env_without_ontology.observation_space['ball_shape'].n       # e.g., 2 ball shapes
    )

    state_size_ontology = (
        env_with_ontology.observation_space['agent_position'].n,    # e.g., 5 vertical positions
        env_with_ontology.observation_space['ball_position'].n,     # e.g., 10 horizontal positions
        env_with_ontology.observation_space['ball_shape'].n,        # e.g., 2 ball shapes
        grid_rows  # Use the correct grid_rows instead of hardcoding 5
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

    # Define paths for saving Q-tables and plots
    models_dir = Path(config['paths']['models']).resolve()
    assets_dir = Path(config['paths']['assets']).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    standard_qtable_path = models_dir / 'standard_agent_qtable.pkl'
    ontology_qtable_path = models_dir / 'ontology_agent_qtable.pkl'
    plot_path = assets_dir / 'agent_performance_comparison.png'
    gif_standard_path = assets_dir / 'standard_agent_performance.gif'
    gif_ontology_path = assets_dir / 'ontology_agent_performance.gif'

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
    # Adjust xlim and ylim for better visualization
    plt.xlim(0, episodes)
    min_reward = min(min(rewards_with), min(rewards_without))
    max_reward = max(max(rewards_with), max(rewards_without))
    plt.ylim(min_reward - 1, max_reward + 1)

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
    # Adjust xlim and ylim for better visualization
    plt.xlim(0, episodes)
    min_steps = min(min(steps_with), min(steps_without))
    max_steps = max(max(steps_with), max(steps_without))
    plt.ylim(min_steps - 1, max_steps + 1)

    plt.tight_layout()

    # Save plot to assets
    plt.savefig(plot_path)
    logging.info(f"\nPerformance comparison plot saved to {plot_path}")
    plt.show()

    # Save Q-tables
    with open(standard_qtable_path, 'wb') as f:
        pickle.dump(agent_without_ontology.q_table, f)
    logging.info(f"Standard Agent Q-table saved to {standard_qtable_path}")

    with open(ontology_qtable_path, 'wb') as f:
        pickle.dump(agent_with_ontology.q_table, f)
    logging.info(f"Ontology-Enhanced Agent Q-table saved to {ontology_qtable_path}")

    # Function to run a test episode and record history
    def run_test_episode_record_history(agent, env, use_ontology=False):
        """
        Run a single test episode and record the history of states.

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

    # Testing and creating GIFs
    # Test Standard Agent
    logging.info("\nTesting the Standard Agent (Without Ontology) and creating GIF...")
    test_history_standard = run_test_episode_record_history(agent_without_ontology, env_without_ontology, use_ontology=False)
    anim_standard = create_animation(test_history_standard, agent_type='Standard Agent')
    # Save the animation as a GIF using Pillow
    try:
        anim_standard.save(gif_standard_path, writer='pillow', fps=2)
        logging.info(f"Standard Agent GIF saved to {gif_standard_path}")
    except Exception as e:
        logging.error(f"Failed to save Standard Agent GIF: {e}")

    # Test Ontology-Enhanced Agent
    logging.info("\nTesting the Ontology-Enhanced Agent (With Ontology) and creating GIF...")
    test_history_ontology = run_test_episode_record_history(agent_with_ontology, env_with_ontology, use_ontology=True)
    anim_ontology = create_animation(test_history_ontology, agent_type='Ontology-Enhanced Agent')
    # Save the animation as a GIF using Pillow
    try:
        anim_ontology.save(gif_ontology_path, writer='pillow', fps=2)
        logging.info(f"Ontology-Enhanced Agent GIF saved to {gif_ontology_path}")
    except Exception as e:
        logging.error(f"Failed to save Ontology-Enhanced Agent GIF: {e}")

    logging.info("\nTraining and testing completed.")


if __name__ == "__main__":
    main()
