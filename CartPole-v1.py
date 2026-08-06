import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")
bins = [20, 20, 20, 20]  # Number of bins for each state variable
q_table = np.zeros(bins + [env.action_space.n])

# CartPole state bounds (velocity/angular velocity are theoretically infinite, so we clip them)
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = (-4.8, 4.8)   # cart velocity
state_bounds[3] = (-6.0, 6.0)   # pole angular velocity

def discretize(state):
    ratios = [(state[i] - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0]) for i in range(len(state))]
    new_state = [int(round((bins[i] - 1) * ratios[i])) for i in range(len(state))]
    new_state = [min(bins[i] - 1, max(0, new_state[i])) for i in range(len(state))]
    return tuple(new_state)

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.95
episodes = 1000

for episode in range(episodes):
    state, info = env.reset()
    state = discretize(state)
    done = False

    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = discretize(next_state)

        q_table[state][action] += learning_rate * (
            reward + discount_factor * np.max(q_table[next_state]) - q_table[state][action]
        )

        state = next_state
        done = terminated or truncated

env.close()