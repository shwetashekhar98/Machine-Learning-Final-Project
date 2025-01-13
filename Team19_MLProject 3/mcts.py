import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd  # For storing results in Excel

# Define possible actions
ACTIONS = ["L", "R", "U", "D"]
WINNING_TILE = 2048

def add_new_tile(board):
    """Add a new tile (2 or 4) to a random empty position on the board."""
    empty_positions = [(i, j) for i in range(4) for j in range(4) if board[i][j] == 0]
    if not empty_positions:
        return board
    i, j = random.choice(empty_positions)
    board[i][j] = 2 if random.random() < 0.9 else 4
    return board

def move(board, action):
    """Apply the action to the board and return the new board and score."""
    def slide_and_merge(row):
        """Slide non-zero tiles to the left and merge equal tiles."""
        new_row = [i for i in row if i != 0]
        score = 0
        i = 0
        while i < len(new_row) - 1:
            if new_row[i] == new_row[i + 1]:
                new_row[i] *= 2
                score += new_row[i]
                del new_row[i + 1]
                new_row.append(0)
            i += 1
        new_row = [i for i in new_row if i != 0]
        return new_row + [0] * (4 - len(new_row)), score

    rotated = False
    if action == "U":
        board = np.rot90(board, -1)
        rotated = True
    elif action == "D":
        board = np.rot90(board, 1)
        rotated = True
    elif action == "R":
        board = np.fliplr(board)

    score = 0
    for i in range(4):
        row = board[i]
        new_row, row_score = slide_and_merge(row)
        board[i] = new_row
        score += row_score

    if action == "U":
        board = np.rot90(board, 1)
    elif action == "D":
        board = np.rot90(board, -1)
    elif action == "R":
        board = np.fliplr(board)

    return board, score

def is_game_over(board):
    """Check if there are no valid moves left."""
    for action in ACTIONS:
        new_board, _ = move(copy.deepcopy(board), action)
        if not np.array_equal(board, new_board):
            return False
    return True

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0

    def is_fully_expanded(self):
        return len(self.children) == len(ACTIONS)

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.score / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

def tree_policy(node):
    while not is_game_over(node.state):
        if not node.is_fully_expanded():
            return expand(node)
        else:
            node = node.best_child()
    return node

def expand(node):
    tried_actions = [child.action for child in node.children]
    new_action = random.choice([a for a in ACTIONS if a not in tried_actions])
    new_state, _ = move(copy.deepcopy(node.state), new_action)
    child_node = MCTSNode(new_state, parent=node)
    child_node.action = new_action
    node.children.append(child_node)
    return child_node

def default_policy(state):
    current_state = copy.deepcopy(state)
    while not is_game_over(current_state):
        action = random.choice(ACTIONS)
        current_state, _ = move(current_state, action)
        add_new_tile(current_state)
    return np.max(current_state)

def backup(node, reward):
    while node is not None:
        node.visits += 1
        node.score += reward
        node = node.parent

def mcts(root, n_iter=1000):
    for _ in range(n_iter):
        leaf = tree_policy(root)
        simulation_result = default_policy(leaf.state)
        backup(leaf, simulation_result)
    return root.best_child(c_param=0.)

def play_game(n_iter=1000):
    board = np.zeros((4, 4), dtype=int)
    board = add_new_tile(board)
    board = add_new_tile(board)
    root = MCTSNode(board)

    total_score = 0
    while not is_game_over(board):
        best_child = mcts(root, n_iter=n_iter)
        board, move_score = move(board, best_child.action)
        total_score += move_score
        board = add_new_tile(board)
        root = MCTSNode(board)

    max_tile = np.max(board)
    return total_score, max_tile

def episodic_play(n_episodes=10, n_iter=1000):
    results = []  # To store results for Excel

    for episode in range(n_episodes):
        print(f"Starting Episode {episode + 1}")
        score, max_tile = play_game(n_iter)
        results.append({"Episode": episode + 1, "Score": score, "Max Tile": max_tile})
        print(f"Episode {episode + 1} finished. Score: {score}, Max Tile: {max_tile}")

    # Save results to Excel
    df = pd.DataFrame(results)
    df.to_excel("2048_MCTS_Results.xlsx", index=False)
    print(f"\nResults saved to 2048_MCTS_Results.xlsx.")

if __name__ == "__main__":
    episodic_play(n_episodes=100, n_iter=1000)


excel_file = '2048_MCTS_Results.xlsx' 

# Read the Excel file
try:
    data = pd.read_excel(excel_file)

    
    if all(col in data.columns for col in ['Episode', 'Score', 'Max Tile']):
        
        episodes = data['Episode']
        scores = data['Score']
        max_tiles = data['Max Tile']

        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, scores, marker='o', linestyle='-', label='Score')
        plt.title('Score vs Episode')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.grid(True)
        plt.legend()
        plt.savefig('score_vs_episode.png')  
        plt.show()

        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, max_tiles, marker='o', linestyle='-', label='Max Tile', color='orange')
        plt.title('Max Tile vs Episode')
        plt.xlabel('Episode')
        plt.ylabel('Max Tile')
        plt.grid(True)
        plt.legend()
        plt.savefig('max_tile_vs_episode.png')  # Optional: Save the plot
        plt.show()

except:
    print("failed to read excel")
