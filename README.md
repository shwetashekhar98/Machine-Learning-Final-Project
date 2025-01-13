#  CS-GY 6923 | Machine Learning Final Project: Reinforcement Learning for Solving 2048

This repository contains the final project for the **Machine Learning course**, where we compare and evaluate three reinforcement learning (RL) algorithms to solve the **2048 Tiles game**. The 2048 game is a combinatorial optimization problem and a benchmark for testing the efficacy of machine learning techniques in high-dimensional, stochastic environments.

## Project Overview

The **2048 Tiles game** involves merging numbered tiles on a 4×4 grid to create a tile with the value 2048. It serves as an excellent challenge for reinforcement learning algorithms due to the game's large state space and stochastic nature.

### Algorithms Explored
1. **Q-learning (Baseline Model)**: A tabular reinforcement learning algorithm that learns an optimal action-value function for each state-action pair.
2. **Deep Q-Network (DQN)**: Combines Q-learning with neural networks to approximate the value function, enabling it to handle high-dimensional inputs effectively.
3. **Monte Carlo Tree Search (MCTS)**: Uses simulation-based planning to evaluate potential moves and determine optimal actions.

### Objectives
- Compare the performance of Q-learning, DQN, and MCTS in the context of the 2048 game.
- Evaluate algorithms based on:
  - **Game score**
  - **Maximum tile reached**
  - **Computational efficiency**

### Key Results
- **DQN** achieved the highest performance, reaching the 2048 tile consistently.
- **MCTS** demonstrated strategic planning capabilities but was computationally intensive.
- **Q-learning**, as the baseline, struggled due to the game's high dimensionality.

## Repository Structure

```
.
├── proposal.pdf                    # Project proposal document
├── project_report.pdf              # Final project report
├── q-learning.py                   # Implementation of Q-learning for 2048
├── dqn.ipynb                       # DQN implementation in a Jupyter notebook
├── mcts.py                         # Monte Carlo Tree Search implementation
├── plots/                          # Generated performance visualizations
├── utils/                          # Helper scripts for state management
└── README.md                       # Repository documentation
```

### Detailed Descriptions
- **proposal.pdf**: Describes the project's objectives, methodology, and expected outcomes.
- **project_report.pdf**: Final report presenting the comparative analysis of the algorithms, methodologies used, and key findings.
- **q-learning.py**: Python script implementing Q-learning for the 2048 game.
- **dqn.ipynb**: Jupyter Notebook showcasing the DQN implementation with experience replay and neural network architecture.
- **mcts.py**: Python script for Monte Carlo Tree Search, including tree expansion and simulation.
- **plots/**: Contains visualizations like `score_vs_episode.png` and `max_tile_vs_episode.png` that summarize the performance of the algorithms.
- **utils/**: Utility scripts for state representation, board management, and data preprocessing.

## How to Use

### Requirements
- Python 3.x
- Libraries: NumPy, TensorFlow/Keras, Matplotlib, Pandas, OpenAI Gym (optional for environment simulation)

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/shwetashekhar98/Machine-Learning-Final-Project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Machine-Learning-Final-Project
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Algorithms
- **Q-learning**:
  ```bash
  python q-learning.py
  ```
- **DQN**:
  Open `dqn.ipynb` in Jupyter Notebook and execute the cells.
- **MCTS**:
  ```bash
  python mcts.py
  ```

### Visualizations
- Results are saved in the `plots/` directory.
- Example: `score_vs_episode.png` shows the scores achieved across episodes.

## Acknowledgments

This project was completed as part of the **Machine Learning course (CS-GY6923)** at NYU. Special thanks to our instructors for their guidance and support.

## Contributors
- Raj Trikha
- Shweta Shekhar
- Rachit Mehul Pathak

---

