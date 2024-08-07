import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from solo12_gym import Solo12Env

# Create environment with rendering enabled
env = Solo12Env(render=True)

# Check environment
try:
    check_env(env, warn=True)
except Exception as e:
    print(f"Environment check failed: {e}")
    exit(1)

# Train the agent
model = PPO('MlpPolicy', env, verbose=1)
total_timesteps = 100000

# Track time for iterations
start_time = time.time()

try:
    model.learn(total_timesteps=total_timesteps)
except KeyboardInterrupt:
    print("Training interrupted. Saving the model...")

    # Save the model on interrupt
    model.save("solo12_ppo_interrupt")

    end_time = time.time()
    total_time = end_time - start_time
    average_time_per_iteration = total_time / model.num_timesteps

    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Average time per iteration: {average_time_per_iteration:.6f} seconds")
    print("Model saved as 'solo12_ppo_interrupt'. Exiting.")

    # Exit the script
    exit(0)

# If training completes without interruption, save the model normally
model.save("solo12_ppo")

end_time = time.time()
total_time = end_time - start_time
average_time_per_iteration = total_time / total_timesteps

print(f"Total training time: {total_time:.2f} seconds")
print(f"Average time per iteration: {average_time_per_iteration:.6f} seconds")
print("Model saved as 'solo12_ppo'.")

# Test the trained model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()


