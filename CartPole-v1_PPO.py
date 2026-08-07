import gymnasium as gym
from stable_baselines3 import PPO

# Initialize the envrionment
env = gym.make('CartPole-v1')

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train model
model.learn(total_timesteps=10000)

# Test the trained model
state, info = env.reset()

# print(state) Used to debug the env.reset() function for ValueError

for t in range(500):
    action, _ = model.predict(state)
    state, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        print(f"Episode finished after {t+1} timesteps")
        break

env.close()