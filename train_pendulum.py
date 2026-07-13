import gymnasium as gym
import numpy as np
from sac_agent import SACAgent, ReplayBuffer

env = gym.make("Pendulum-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

print("state_dim:", state_dim)
print("action_dim:", action_dim)

agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
buffer = ReplayBuffer(capacity=100000, state_dim=state_dim, action_dim=action_dim)


#学習ループの実装
NUM_EPISODES = 100
MAX_STEPS = 200
START_STEPS = 1000
BATCH_SIZE = 256

total_steps = 0
episode_rewards = []

for episode in range(NUM_EPISODES):
    state, info = env.reset()
    episode_reward = 0

    for step in range(MAX_STEPS):
        if total_steps < START_STEPS:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.add(state, action, reward, next_state, float(done))

        state = next_state
        episode_reward += reward
        total_steps += 1

        if buffer.size > BATCH_SIZE:
            agent.update(buffer, batch_size=BATCH_SIZE)

        if done:
            break

    episode_rewards.append(episode_reward)
    print(f"エピソード{episode+1}/{NUM_EPISODES}  報酬合計: {episode_reward:.2f}")
 

