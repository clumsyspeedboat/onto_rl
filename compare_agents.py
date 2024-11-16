# compare_agents.py

import numpy as np
import matplotlib.pyplot as plt
from ping_pong_env import PingPongEnv
from q_learning_agent import QLearningAgent
from ontology_enhanced_agent import OntologyEnhancedAgent
from ontology_definition import onto
import pickle

def main():
    env = PingPongEnv()
    episodes = 1000
    test_episodes = 5

    # Define state size for standard agent
    state_size_standard = (
        env.observation_space['agent_position'].n,  # 5 vertical positions
        env.observation_space['ball_position'].n,   # 10 horizontal positions
        env.observation_space['ball_shape'].n       # 2 ball shapes
    )

    # Define state size for ontology-enhanced agent (including ball_y_position)
    state_size_ontology = (
        env.observation_space['agent_position'].n,  # 5 vertical positions
        env.observation_space['ball_position'].n,   # 10 horizontal positions
        env.observation_space['ball_shape'].n,      # 2 ball shapes
        5                                           # ball_y_position: 0 to 4 (vertical positions)
    )

    # Initialize agents
    standard_agent = QLearningAgent(action_size=env.action_space.n, state_size=state_size_standard)
    ontology_agent = OntologyEnhancedAgent(action_size=env.action_space.n, state_size=state_size_ontology,
                                           ontology=onto)

    # Training function
    def train_agent(agent, env, episodes, ontology=False):
        rewards = []
        for episode in range(1, episodes+1):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                if ontology:
                    agent_pos = state['agent_position']
                    ball_pos = state['ball_position']
                    ball_shape = state['ball_shape']
                    ball_y_pos = env.ball_y_position  # Retrieve ball's vertical position
                    enriched_state = (agent_pos, ball_pos, ball_shape, ball_y_pos)
                    shape_str, speed = agent.get_ball_shape_properties(ball_shape)
                    action = agent.choose_action(enriched_state)
                    next_state, reward, done, _ = env.step(action)
                    enriched_next_state = (
                        next_state['agent_position'],
                        next_state['ball_position'],
                        next_state['ball_shape'],
                        env.ball_y_position  # Updated ball_y_position after the step
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
            rewards.append(total_reward)
            if episode % 100 == 0:
                print(f"Episode {episode}: Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")
        return rewards

    # Train both agents
    print("Training Standard Agent...")
    rewards_standard = train_agent(standard_agent, env, episodes, ontology=False)

    print("\nTraining Ontology-Enhanced Agent...")
    rewards_ontology = train_agent(ontology_agent, env, episodes, ontology=True)

    # Plotting
    plt.figure(figsize=(12,6))
    plt.plot(rewards_standard, label='Without Ontology', alpha=0.7)
    plt.plot(rewards_ontology, label='With Ontology', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Agent Performance Comparison')
    plt.legend()
    plt.xlim(0,600)  # Focus on the first 600 episodes
    plt.show()

    # Save Q-tables
    with open('standard_agent_qtable.pkl', 'wb') as f:
        pickle.dump(standard_agent.q_table, f)
    with open('ontology_agent_qtable.pkl', 'wb') as f:
        pickle.dump(ontology_agent.q_table, f)

    # Testing agents
    def test_agent(agent, env, episodes, ontology=False):
        for ep in range(1, episodes+1):
            state = env.reset()
            done = False
            total_reward = 0
            print(f"Test Episode {ep}:")
            env.render()
            while not done:
                if ontology:
                    agent_pos = state['agent_position']
                    ball_pos = state['ball_position']
                    ball_shape = state['ball_shape']
                    ball_y_pos = env.ball_y_position  # Retrieve ball's vertical position
                    enriched_state = (agent_pos, ball_pos, ball_shape, ball_y_pos)
                    action = agent.choose_action(enriched_state)
                else:
                    agent_pos = state['agent_position']
                    ball_pos = state['ball_position']
                    ball_shape = state['ball_shape']
                    enriched_state = (agent_pos, ball_pos, ball_shape)
                    action = agent.choose_action(enriched_state)
                next_state, reward, done, _ = env.step(action)
                env.render()
                if ontology:
                    # During testing, no learning is performed
                    pass
                else:
                    agent.learn(enriched_state, action, reward, (
                        next_state['agent_position'],
                        next_state['ball_position'],
                        next_state['ball_shape']
                    ), done)
                state = next_state
                total_reward += reward
            print(f"Total Reward: {total_reward}\n")

    print("\nTesting Standard Agent...")
    test_agent(standard_agent, env, test_episodes, ontology=False)

    print("Testing Ontology-Enhanced Agent...")
    test_agent(ontology_agent, env, test_episodes, ontology=True)

if __name__ == "__main__":
    main()
