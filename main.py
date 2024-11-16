# main.py

from gridworld_env import GridWorldEnv
from q_learning_agent import QLearningAgent
import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size=100):
    """Compute moving average to smooth the plot."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def main():
    grid_size = 10
    episodes = 2000
    test_episodes = 5

    # Initialize environments
    env_with_ontology = GridWorldEnv(grid_size=grid_size, use_ontology=True)
    env_without_ontology = GridWorldEnv(grid_size=grid_size, use_ontology=False)

    # Initialize agents
    agent_with_ontology = QLearningAgent(env_with_ontology, use_ontology=True)
    agent_without_ontology = QLearningAgent(env_without_ontology, use_ontology=False)

    # Train both agents
    print("Training the agent WITH ontology...")
    rewards_with, steps_with = agent_with_ontology.train(episodes=episodes)

    print("\nTraining the agent WITHOUT ontology...")
    rewards_without, steps_without = agent_without_ontology.train(episodes=episodes)

    # Plot cumulative rewards
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 1, 1)
    plt.plot(rewards_with, label='With Ontology', alpha=0.7)
    plt.plot(rewards_without, label='Without Ontology', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()

    plt.xlim(-5, 200)

    # Plot moving average for smoother visualization
    ma_with = moving_average(rewards_with, window_size=100)
    ma_without = moving_average(rewards_without, window_size=100)
    plt.plot(ma_with, label='With Ontology (MA)', color='blue')
    plt.plot(ma_without, label='Without Ontology (MA)', color='orange')

    plt.subplot(2, 1, 2)
    plt.plot(steps_with, label='With Ontology', alpha=0.7)
    plt.plot(steps_without, label='Without Ontology', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')
    plt.legend()

    plt.xlim(-5, 200)

    # Plot moving average for steps
    ma_steps_with = moving_average(steps_with, window_size=100)
    ma_steps_without = moving_average(steps_without, window_size=100)
    plt.plot(ma_steps_with, label='With Ontology (MA)', color='blue')
    plt.plot(ma_steps_without, label='Without Ontology (MA)', color='orange')

    plt.tight_layout()
    plt.show()

    # Test both agents
    print("\nTesting the agent WITH ontology...")
    agent_with_ontology.test(episodes=test_episodes)

    print("\nTesting the agent WITHOUT ontology...")
    agent_without_ontology.test(episodes=test_episodes)

if __name__ == "__main__":
    main()
