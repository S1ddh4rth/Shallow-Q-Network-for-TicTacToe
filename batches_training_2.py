import sys
from TicTacToe import *
import numpy as np
import random
from collections import deque

# whyyyy does this not work ????
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense


from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam  # type: ignore


class PlayerSQN:
    def __init__(self):
        """
        Initializes the PlayerSQN class.
        """
        # Define the neural network
        self.state_size = 9  # State is a 9-dimensional vector representing the board
        self.input_dim = 9  # Input is one of 9 options
        self.learning_rate = 0.001

        # Hyperparameters for training
        self.BATCH_SIZE = 32  # Size of mini-batch
        self.GAMMA = 0.99  # Discount factor
        self.EPOCHS = 3  # Number of training epochs each time we sample a batch from the replay buffer

        # Epsilon-greedy exploration parameters
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.exploration_min = 0.01

        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)

        # Sequential model for SQN
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=self.input_dim, activation='relu'))  # First hidden layer
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(9, activation='linear'))  # Output layer
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
    # def move(self, state):
    #     """
    #     Determines Player 2's move based on the current state of the game.

    #     Parameters:
    #     state (list): A list representing the current state of the TicTacToe board.

    #     Returns:
    #     int: The position (0-8) where Player 2 wants to make a move.
    #     """
    #     # In your final submission, PlayerSQN must be controlled by an SQN. Use an epsilon-greedy action selection policy.
    #     # In this implementation, PlayerSQN is controlled by terminal input.

    #     # print(f"Current state: {state}")
    #     # action = int(input("Player 2 (You), enter your move (1-9): ")) - 1
    #     # return action


    #     # Main idea of epsilon greedy:
    #     # We sample a variable uniformly over [0,1]
    #     # if the variable is less than the exploration probability
    #     #     we choose an action randomly
    #     # else
    #     #     we forward the state through the SQN and choose the action
    #     #     with the highest Q-value.

    #     if np.random.uniform(0,1) < self.exploration_rate:
    #         return np.random.choice(range(self.input_dim))
    #     q_values = self.model.predict(np.array([state]))[0]
    #     return np.argmax(q_values)

    
    def move(self, state):
        """
        Determines Player 2's move based on the current state of the game.
        """
        if np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.choice(self.valid_actions(state))

        # Predict Q-values and filter valid actions
        q_values = self.model.predict(np.array([state]))[0]
        valid_moves = self.valid_actions(state)
        valid_q_values = {action: q_values[action] for action in valid_moves}
        return max(valid_q_values, key=valid_q_values.get)

    def valid_actions(self, state):
        """Returns a list of valid actions."""
        return [i for i, x in enumerate(state) if x == 0]

    def decay_exploration_rate(self):
        """Decay the exploration rate."""
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def store_experience(self, current_state, action, reward, next_state, done):
        """Store experience in the replay buffer."""
        self.replay_buffer.append({
            "current_state": current_state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        })

    def train_on_batch(self, batch):
        """Train the model using a single batch."""
        current_states = np.array([exp["current_state"] for exp in batch])
        next_states = np.array([exp["next_state"] for exp in batch])

        current_qs = self.model.predict(current_states)
        next_qs = self.model.predict(next_states)

        x, y = [], []

        for i, experience in enumerate(batch):
            action = experience["action"]
            reward = experience["reward"]
            next_state = experience["next_state"]
            done = experience["done"]

            if done:
                target_q = reward
            else:
                max_next_q = max([next_qs[i][a] for a in self.valid_actions(next_state)])
                target_q = reward + self.GAMMA * max_next_q

            current_qs[i][action] = target_q
            x.append(experience["current_state"])
            y.append(current_qs[i])

        self.model.fit(np.array(x), np.array(y), epochs=1, verbose=0)

def test_model(playerSQN, smartness, num_games=30):
    """
    Test the model's performance by simulating a number of games.
    
    Args:
        playerSQN: The trained PlayerSQN instance.
        smartness: The current smartness level of Player 1.
        num_games: Number of games to simulate for testing.
    
    Returns:
        A list of outcomes for the games: 1 for win, 0 for draw, -1 for loss.
    """
    outcomes = []
    for _ in range(num_games):
        game = TicTacToe(smartness, playerSQN)
        state = [0] * 9
        done = False

        game.player1_move()
        state = game.board.copy()

        while not done:
            action = playerSQN.move(state)
            valid = game.make_move(action, 2)
            if not valid:
                continue

            done = game.current_winner is not None or game.is_full()
            if done:
                break

            game.player1_move()
            state = game.board.copy()
            done = game.current_winner is not None or game.is_full()

        # Record the outcome
        if game.current_winner == 2:
            outcomes.append(1)  # Win
        elif game.current_winner == 1:
            outcomes.append(-1)  # Loss
        else:
            outcomes.append(0)  # Draw

    return outcomes


def main(smartMovePlayer1_test):
    """
    Simulates multiple TicTacToe games, trains the SQN player, and evaluates its performance.
    """
    playerSQN = PlayerSQN()
    episodes_per_level = 500
    max_smartness = 1.0
    smartness_increment = 0.1
    current_smartness = 0.0
    test_file = "tictactoe_test_results1.txt"

    with open(test_file, "w") as f:
        f.write("TicTacToe Test Results\n\n")

    while current_smartness <= max_smartness:
        print(f"\nStarting training with smartness level: {current_smartness:.2f}")

        # Simulate episodes for the current smartness level
        for episode in range(episodes_per_level):
            game = TicTacToe(current_smartness, playerSQN)
            state = [0] * 9
            done = False

            game.player1_move()
            state = game.board.copy()

            while not done:
                action = playerSQN.move(state)
                valid = game.make_move(action, 2)
                if not valid:
                    continue

                reward = game.get_reward()
                next_state = game.board.copy()
                done = game.current_winner is not None or game.is_full()

                playerSQN.store_experience(state, action, reward, next_state, done)

                if done:
                    playerSQN.decay_exploration_rate()
                    break

                game.player1_move()
                state = game.board.copy()
                done = game.current_winner is not None or game.is_full()

        # Train the model on 100 batches
        print("Training the model...")

        for _ in range(100):  # Sample 100 batches and train on each
            if len(playerSQN.replay_buffer) < playerSQN.BATCH_SIZE:
                continue  # Skip if there are not enough experiences to form a batch

            batch = random.sample(playerSQN.replay_buffer, playerSQN.BATCH_SIZE)
            playerSQN.train_on_batch(batch)

        # Test the model's performance after 100 epochs
        outcomes = test_model(playerSQN, current_smartness)
        with open(test_file, "a") as f:
            f.write(f"After training at smartness {current_smartness:.2f}:\n")
            f.write(f"Outcomes: {outcomes}\n")
            f.write(f"Wins: {outcomes.count(1)}, Draws: {outcomes.count(0)}, Losses: {outcomes.count(-1)}\n\n")

        # Clear the replay buffer
        playerSQN.replay_buffer.clear()
        print(f"Replay buffer cleared. Smartness {current_smartness:.2f} completed.\n")

        # Increment the smartness level
        current_smartness = round(min(current_smartness + smartness_increment, max_smartness), 2)
        if current_smartness == 1:
            break

    # Final Testing
    print("Final testing on smartness levels 0 and 0.8...")
    for test_smartness in [0, 0.8]:
        outcomes = test_model(playerSQN, test_smartness)
        with open(test_file, "a") as f:
            f.write(f"Final test at smartness {test_smartness:.2f}:\n")
            f.write(f"Outcomes: {outcomes}\n")
            f.write(f"Wins: {outcomes.count(1)}, Draws: {outcomes.count(0)}, Losses: {outcomes.count(-1)}\n\n")

    print("Training complete.")
    playerSQN.model.save("tictactoe_trained_model_v4.keras")
    print("Model saved as 'tictactoe_trained_model_v4.keras'.")


if __name__ == "__main__":
    main(0)
