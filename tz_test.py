import torch
from tz_train_env import Solo12Env
from tz_ppo import PPO, Memory, device

def test():
    env = Solo12Env()
    memory = Memory()
    ppo = PPO(env.observation_space, env.action_space)
    ppo.load("ppo_solo12.pth")

    num_episodes = 10
    max_timesteps = 200

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for t in range(max_timesteps):
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break

        print(f"Episode: {episode+1}, Total Reward: {total_reward}")

    env.close()

if __name__ == '__main__':
    test()
