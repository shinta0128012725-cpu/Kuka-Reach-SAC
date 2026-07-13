import numpy as np
from  kuka_reach_env import KukaReachEnv
from sac_agent import SACAgent, ReplayBuffer

env = KukaReachEnv()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

print("state_dim:", state_dim)
print("action_dim:", action_dim)

agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
buffer = ReplayBuffer(capacity=100000, state_dim=state_dim, action_dim=action_dim)


NUM_EPISODES = 200
MAX_STEPS = 50
START_STEPS = 1000
BATCH_SIZE = 256

total_steps = 0
episode_rewards = []
critic1_losses = []
critic2_losses = []
actor_losses = []


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
            c1_loss, c2_loss, a_loss = agent.update(buffer, batch_size=BATCH_SIZE)
            critic1_losses.append(c1_loss)
            critic2_losses.append(c2_loss)
            actor_losses.append(a_loss)

        if done:
            break

    episode_rewards.append(episode_reward)
    print(f"エピソード{episode+1}/{NUM_EPISODES}  報酬合計: {episode_reward:.2f}  ステップ数: {step+1}")

env.close()

np.save("kuka_episode_rewards.npy", np.array(episode_rewards))
np.save("kuka_critic1_losses.npy", np.array(critic1_losses))
np.save("kuka_critic2_losses.npy", np.array(critic2_losses))
np.save("kuka_actor_losses.npy", np.array(actor_losses))

print("学習完了。episode_rewardsを保存しました。")