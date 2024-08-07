import numpy as np
import torch
from tz_train_env import Solo12Env
from tz_ppo import PPO, Memory, device

def train():
    env = Solo12Env()
    memory = Memory()
    ppo = PPO(env.observation_space, env.action_space)
    num_episodes = 10
    max_timesteps = 1000

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for t in range(max_timesteps):
            action = ppo.select_action(state, memory)
            next_state, reward, done, _ = env.step(action)
            memory.states.append(torch.tensor(state, dtype=torch.float32).to(device))
            memory.actions.append(torch.tensor(action, dtype=torch.float32).to(device))
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            state = next_state
            total_reward += reward

            if done:
                break

        # Update PPO after collecting a complete episode
        ppo.update(memory)
        memory.clear_memory()
        print(f"Episode: {episode+1}, Total Reward: {total_reward}")

    ppo.save("ppo_solo12.pth")
    env.close()

if __name__ == '__main__':
    train()
