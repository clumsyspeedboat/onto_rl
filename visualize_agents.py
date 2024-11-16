# visualize_agents.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from ping_pong_env import PingPongEnv
from q_learning_agent import QLearningAgent
from ontology_enhanced_agent import OntologyEnhancedAgent
from ontology_definition import onto
import pickle

def load_agents():
    """
    Load pre-trained agents from saved Q-tables.
    """
    env = PingPongEnv()
    
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

    # Load trained Q-tables
    try:
        with open('standard_agent_qtable.pkl', 'rb') as f:
            standard_agent.q_table = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Standard agent Q-table not found. Please train the agent first using 'compare_agents.py'.")
    
    try:
        with open('ontology_agent_qtable.pkl', 'rb') as f:
            ontology_agent.q_table = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Ontology-enhanced agent Q-table not found. Please train the agent first using 'compare_agents.py'.")
    
    return standard_agent, ontology_agent, env

def run_test_episode(agent, env, ontology=False):
    """
    Run a single test episode and record the history.
    """
    state = env.reset()
    done = False
    history = []
    
    while not done:
        history.append(state)
        if ontology:
            agent_pos = state['agent_position']
            ball_pos = state['ball_position']
            ball_shape = state['ball_shape']
            ball_y_pos = env.ball_y_position  # Retrieve ball's vertical position
            enriched_state = (agent_pos, ball_pos, ball_shape, ball_y_pos)
            shape_str, speed = agent.get_ball_shape_properties(ball_shape)
            action = agent.choose_action(enriched_state)
        else:
            agent_pos = state['agent_position']
            ball_pos = state['ball_position']
            ball_shape = state['ball_shape']
            enriched_state = (agent_pos, ball_pos, ball_shape)
            action = agent.choose_action(enriched_state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
    history.append(state)
    return env.get_episode_history()

def create_animation(agent_history, agent_type='Standard'):
    """
    Create an animation from the episode history.
    """
    agent_positions = [step['agent_position'] for step in agent_history]
    ball_positions = [step['ball_position'] for step in agent_history]
    ball_shapes = [step['ball_shape'] for step in agent_history]
    ball_y_positions = [step['ball_y_position'] for step in agent_history]

    grid_rows = 5
    grid_cols = 10

    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_xlim(-1, grid_cols)
    ax.set_ylim(-1, grid_rows)
    ax.set_xticks(range(0, grid_cols+1))
    ax.set_yticks(range(0, grid_rows+1))
    ax.set_xlabel('Horizontal Position')
    ax.set_ylabel('Vertical Position')
    ax.set_title(f'Ping Pong Agent Performance: {agent_type}')

    # Plot agent and ball
    agent_dot_top, = ax.plot([], [], 'bs', markersize=20, label='Goalkeeper Top')  # Blue square
    agent_dot_bottom, = ax.plot([], [], 'bs', markersize=20, label='Goalkeeper Bottom')  # Blue square
    ball_dot, = ax.plot([], [], 'ro', markersize=12, label='Ball')        # Red circle or square
    ball_shape_text = ax.text(0,0, '', fontsize=12, ha='center', va='center')

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
            if agent_pos < 4:
                agent_top = agent_pos
                agent_bottom = agent_pos +1
            else:
                agent_top = agent_pos -1
                agent_bottom = agent_pos

            # Update agent positions
            agent_dot_top.set_data([0], [agent_top])
            agent_dot_bottom.set_data([0], [agent_bottom])

            # Ball moves horizontally, vertical position is ball_y_pos
            ball_dot.set_data([ball_pos], [ball_y_pos])

            # Display ball shape
            ball_shape_text.set_text(shape)
            ball_shape_text.set_position((ball_pos, ball_y_pos))
        return agent_dot_top, agent_dot_bottom, ball_dot, ball_shape_text

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(agent_positions), interval=500, blit=True, repeat=False)
    return anim

def main():
    try:
        standard_agent, ontology_agent, env = load_agents()
    except FileNotFoundError as e:
        print(e)
        return
    
    # Run test episodes
    print("Running test episode for Standard Agent...")
    history_standard = run_test_episode(standard_agent, env, ontology=False)
    
    print("Running test episode for Ontology-Enhanced Agent...")
    history_ontology = run_test_episode(ontology_agent, env, ontology=True)
    
    # Create animations
    print("Creating animation for Standard Agent...")
    anim_standard = create_animation(history_standard, agent_type='Standard Agent')
    
    print("Creating animation for Ontology-Enhanced Agent...")
    anim_ontology = create_animation(history_ontology, agent_type='Ontology-Enhanced Agent')
    
    # Display animations
    plt.show()
    
    # Save animations as GIFs using Pillow
    # Ensure Pillow is installed: pip install pillow
    print("Saving Standard Agent animation as 'standard_agent.gif'...")
    anim_standard.save('standard_agent.gif', writer='pillow', fps=2)
    print("Standard Agent GIF saved successfully.")
    
    print("Saving Ontology-Enhanced Agent animation as 'ontology_agent.gif'...")
    anim_ontology.save('ontology_agent.gif', writer='pillow', fps=2)
    print("Ontology-Enhanced Agent GIF saved successfully.")

    # Alternatively, to use ImageMagick, uncomment the following lines and ensure ImageMagick is installed
    # anim_standard.save('standard_agent.gif', writer='imagemagick')
    # anim_ontology.save('ontology_agent.gif', writer='imagemagick')

if __name__ == "__main__":
    main()
