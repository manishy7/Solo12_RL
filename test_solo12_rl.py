import gym
from stable_baselines3 import PPO
from solo12_gym import Solo12Env

# Create environment
env = Solo12Env()

# Load the trained model
model = PPO.load("solo12_ppo")

# Test the trained model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
