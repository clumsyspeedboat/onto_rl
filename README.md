# Ontology-Enhanced Reinforcement Learning for Goalkeeper Simulation

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Ontology Details](#ontology-details)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Overview

This project demonstrates the integration of **Ontology** with **Reinforcement Learning (RL)** to create an intelligent goalkeeper agent in a simplified Ping Pong game environment. The goalkeeper must prevent the ball from entering the goal on the left side by moving vertically (up and down) to intercept incoming balls thrown from the right.

Two agents are implemented and compared:
1. **Standard Q-Learning Agent**: Utilizes traditional RL without additional knowledge.
2. **Ontology-Enhanced Q-Learning Agent**: Incorporates structured knowledge from an ontology to improve decision-making.

## Features

- **Custom Gym Environment**: Simulates a goalkeeper scenario with vertical agent movement and horizontally moving balls.
- **Ontology Integration**: Uses OWL ontologies to provide agents with structured knowledge about the game entities.
- **Q-Learning Agents**: Implements both standard and ontology-enhanced Q-Learning agents.
- **Training and Comparison**: Scripts to train agents and compare their performances.
- **Visualization**: Generates animated GIFs to visualize agent behaviors during test episodes.
- **Extensible Design**: Modular codebase allowing for easy extensions and enhancements.

## Architecture

1. **Environment (`ping_pong_env.py`)**: Defines the game dynamics, agent and ball positions, actions, and rewards.
2. **Ontology (`ontology_definition.py`)**: Specifies the ontology structure using `owlready2`, defining properties like ball shape and speed.
3. **Agents**:
   - **Standard Agent (`q_learning_agent.py`)**: Basic Q-Learning implementation.
   - **Ontology-Enhanced Agent (`ontology_enhanced_agent.py`)**: Q-Learning agent utilizing ontology-derived knowledge.
4. **Training Script (`compare_agents.py`)**: Trains both agents and compares their performances.
5. **Visualization (`visualize_agents.py`)**: Creates animations of agent behaviors and exports them as GIFs.

## Installation

### Prerequisites

- **Python 3.6 or higher**
- **pip** package manager

### Setup Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/ontology-reinforcement-learning-goalkeeper.git
   cd ontology-reinforcement-learning-goalkeeper
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python -m venv env
   ```

3. **Activate the Virtual Environment**

   - **Windows**

     ```bash
     env\Scripts\activate
     ```

   - **macOS/Linux**

     ```bash
     source env/bin/activate
     ```

4. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   **Note:** If `requirements.txt` is not provided, install the necessary packages manually:

   ```bash
   pip install gym numpy matplotlib owlready2 pillow
   ```

## Usage

### 1. Define the Ontology

Before training the agents, ensure that the ontology is correctly defined and saved.

```bash
python ontology_definition.py
```

This script creates and saves the `pingpong.owl` ontology file in the project directory.

### 2. Train the Agents

Run the training script to train both the Standard and Ontology-Enhanced agents.

```bash
python compare_agents.py
```

**Script Details:**

- **Training Process**: Both agents are trained over 1000 episodes. Progress is displayed every 100 episodes, showing total rewards and the exploration rate (`epsilon`).
- **Q-Tables**: After training, the Q-tables are saved as `standard_agent_qtable.pkl` and `ontology_agent_qtable.pkl`.
- **Performance Plot**: A plot comparing the total rewards of both agents over the training episodes is displayed.

### 3. Visualize Agent Performances

Generate and save animations of the agents' behaviors during test episodes.

```bash
python visualize_agents.py
```

**Script Details:**

- **Test Episodes**: Both agents are tested over 5 episodes each.
- **Rendering**: The environment's state is rendered in the console for each step.
- **Animations**: Animated GIFs (`standard_agent.gif` and `ontology_agent.gif`) are created using Pillow and saved in the project directory.

### 4. Review the Results

- **Console Output**: Observe the agents' actions and rewards during training and testing.
- **Animations**: Open the generated GIFs to visualize how each agent interacts with the ball.

## Ontology Details

The ontology provides structured knowledge about the game's entities, enhancing the agent's decision-making process.

### Components:

- **Classes:**
  - `Ball`: Base class for different ball shapes.
  - `Circle`: Subclass representing a circular ball.
  - `Square`: Subclass representing a square-shaped ball.

- **Data Properties:**
  - `has_shape`: Specifies the shape of the ball (`"Circle"` or `"Square"`).
  - `has_speed`: Defines the speed of the ball (float value).

### Ontology Definition Script

```python
# ontology_definition.py

from owlready2 import *

# Create and load ontology
onto = get_ontology("http://example.org/pingpong.owl")

with onto:
    class Ball(Thing):
        pass

    class Circle(Ball):
        pass

    class Square(Ball):
        pass

    class has_shape(DataProperty, FunctionalProperty):
        domain = [Ball]
        range = [str]

    class has_speed(DataProperty, FunctionalProperty):
        domain = [Ball]
        range = [float]

    # Assign properties to Ball instances with distinct names
    circle_obj = Circle("circle_obj")
    circle_obj.has_shape = "Circle"
    circle_obj.has_speed = 1.0

    square_obj = Square("square_obj")
    square_obj.has_shape = "Square"
    square_obj.has_speed = 1.5

# Save the ontology to a file
onto.save(file="pingpong.owl", format="rdfxml")
```

## Project Structure

```
ontology-reinforcement-learning-goalkeeper/
├── compare_agents.py
├── ontology_definition.py
├── ontology_enhanced_agent.py
├── q_learning_agent.py
├── ping_pong_env.py
├── visualize_agents.py
├── pingpong.owl
├── standard_agent_qtable.pkl
├── ontology_agent_qtable.pkl
├── standard_agent.gif
├── ontology_agent.gif
├── requirements.txt
├── README.md
```

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add Your Feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

Please ensure your code adheres to the project's coding standards and includes appropriate documentation and comments.


## Acknowledgements

- **OpenAI Gym**: For providing the RL environment framework.
- **Matplotlib**: For visualization and animation capabilities.
- **OWLReady2**: For ontology management in Python.
- **Pillow**: For image processing and GIF creation.

---
```
