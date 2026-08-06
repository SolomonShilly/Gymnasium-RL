import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")
bins = [20, 20, 20, 20]
q_table = np.zeros(bins + [env.action_space.n])

state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = (-4.8, 4.8)
state_bounds[3] = (-6.0, 6.0)

def discretize(state):
    ratios = [(state[i] - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0]) for i in range(len(state))]
    new_state = [int(round((bins[i] - 1) * ratios[i])) for i in range(len(state))]
    new_state = [min(bins[i] - 1, max(0, new_state[i])) for i in range(len(state))]
    return tuple(new_state)

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.95
episodes = 1000

# Epsilon-greedy parameters
epsilon = 1.0          # start fully random
epsilon_min = 0.01      # never go fully greedy
epsilon_decay = 0.995   # shrink epsilon each episode

for episode in range(episodes):
    state, info = env.reset()
    state = discretize(state)
    done = False
    total_reward = 0

    while not done:
        # Epsilon-greedy: explore vs exploit
        if np.random.random() < epsilon:
            action = env.action_space.sample()      # explore
        else:
            action = np.argmax(q_table[state])       # exploit

        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = discretize(next_state)

        q_table[state][action] += learning_rate * (
            reward + discount_factor * np.max(q_table[next_state]) - q_table[state][action]
        )

        state = next_state
        done = terminated or truncated
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 50 == 0:
        print(f"Episode {episode}, reward: {total_reward}, epsilon: {epsilon:.3f}")

env.close()