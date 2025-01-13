import pandas as pd
import numpy as np
import random
import itertools
import time
import matplotlib.pyplot as plt

# Game Environment for 2048
class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.place_random_tile()
        self.place_random_tile()

    def place_random_tile(self):
        empty_cells = [(r, c) for r, c in itertools.product(range(4), range(4)) if self.board[r][c] == 0]
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.board[r][c] = 2 if random.random() < 0.9 else 4

    def move(self, direction):
        rotated_board = np.rot90(self.board, -direction)
        new_board, score = self._merge(rotated_board)
        self.board = np.rot90(new_board, direction)
        if not np.array_equal(self.board, rotated_board):
            self.score += score
            self.place_random_tile()
            return True, score
        return False, 0

    def _merge(self, board):
        score = 0
        new_board = np.zeros_like(board)
        for r in range(4):
            row = board[r][board[r] != 0]
            merged_row = []
            skip = False
            for i in range(len(row)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(row) and row[i] == row[i + 1]:
                    merged_row.append(2 * row[i])
                    score += 2 * row[i]
                    skip = True
                else:
                    merged_row.append(row[i])
            new_board[r, :len(merged_row)] = merged_row
        return new_board, score

    def get_valid_moves(self):
        valid_moves = []
        for direction in range(4):
            rotated_board = np.rot90(self.board, -direction)
            new_board, _ = self._merge(rotated_board)
            if not np.array_equal(rotated_board, new_board):
                valid_moves.append(direction)
        return valid_moves

    def is_game_over(self):
        return not self.get_valid_moves()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.place_random_tile()
        self.place_random_tile()
        return self.get_state()

    def get_state(self):
        return tuple(self.board.flatten())


# Q-Learning Agent
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_action(self, state, valid_moves):
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        q_values = [self.q_table.get((state, a), 0) for a in valid_moves]
        max_q = max(q_values)
        return random.choice([a for a, q in zip(valid_moves, q_values) if q == max_q])

    def update(self, state, action, reward, next_state, next_valid_moves):
        current_q = self.q_table.get((state, action), 0)
        next_max_q = max([self.q_table.get((next_state, a), 0) for a in next_valid_moves], default=0)
        self.q_table[(state, action)] = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)


# Training Function
def train_2048_agent_with_scores(episodes=2000, target_tile=2048, output_file="2048_scores_results.xlsx"):
    env = Game2048()
    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=1.0)  # High exploration initially

    # Metrics to track
    results = []
    training_time_start = time.time()

    # Epsilon Decay
    epsilon_decay = 0.999
    min_epsilon = 0.01

    for episode in range(episodes):
        state = env.reset()
        total_score = 0
        moves = 0
        won = False

        while not env.is_game_over():
            valid_moves = env.get_valid_moves()
            action = agent.get_action(state, valid_moves)
            moved, score = env.move(action)
            total_score += score
            moves += 1

            # Reward function focuses on the score increment
            reward = score if moved else -10

            if target_tile in env.board:
                won = True

            next_state = env.get_state()
            next_valid_moves = env.get_valid_moves()
            agent.update(state, action, reward, next_state, next_valid_moves)
            state = next_state

        # Log results for the episode
        results.append({
            "Episode": episode + 1,
            "Max Tile": np.max(env.board),
            "Score": total_score,
            "Moves": moves,
            "Won": won
        })

        # Reduce epsilon
        agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)

        # Progress log
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Max Tile: {np.max(env.board)}, Moves: {moves}, Score: {total_score}")

    training_time = time.time() - training_time_start
    print(f"\nTraining Complete in {training_time:.2f} seconds!")

    # Save results to Excel
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

    return agent, df


if __name__ == "__main__":
    _, results_df = train_2048_agent_with_scores(episodes=2000, target_tile=2048, output_file="2048_scores_results.xlsx")

excel_file = '2048_scores_results.xlsx'  

# Read the Excel file
try:
    data = pd.read_excel(excel_file)

    
    if all(col in data.columns for col in ['Episode', 'Score', 'Max Tile']):
        
        episodes = data['Episode']
        scores = data['Score']
        max_tiles = data['Max Tile']

        # Plot Score vs Episode
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, scores, marker='o', linestyle='-', label='Score')
        plt.title('Score vs Episode')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.grid(True)
        plt.legend()
        plt.savefig('score_vs_episode.png')  
        plt.show()

        # Plot Max Tile vs Episode
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, max_tiles, marker='o', linestyle='-', label='Max Tile', color='orange')
        plt.title('Max Tile vs Episode')
        plt.xlabel('Episode')
        plt.ylabel('Max Tile')
        plt.grid(True)
        plt.legend()
        plt.savefig('max_tile_vs_episode.png')  
        plt.show()
except:
    print("failed to read excel")