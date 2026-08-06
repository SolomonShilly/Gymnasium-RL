import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")
bins = [20, 20, 20, 20]  # Number of bins for each state variable
q_table = np.zeros(bins + [env.action_space.n]) # Creating Q-table with dimensions based on the number of bins and actions

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.95
episodes = 1000

# Run for 200 steps
for episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[tuple[state]])  # Choose the action with the highest Q-value
        next_state, reward, terminated, truncated, info = env.step(action)  # Apply the action to the environment

        # Update Q-table using the Q-learning formula
        q_table[tuple[state]][action] += learning_rate * (reward + discount_factor * np.max(q_table[tuple[next_state]]) - q_table[tuple[state]][action])
        state = next_state

env.close()