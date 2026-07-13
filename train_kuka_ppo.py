import numpy as np
from kuka_reach_env import KukaReachEnv
from ppo_agent import PPOAgent, RolloutBuffer

env = KukaReachEnv()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

print("state_dim:", state_dim)
print("action_dim", action_dim)

agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
buffer = RolloutBuffer()

NUM_EPISODES = 200
MAX_STEPS = 50
#updateするエピソード数
UPDATE_EVERY = 5


episode_rewards = []
policy_losses = []
value_losses = []

for episode in range(NUM_EPISODES):
    state, info = env.reset()
    episode_reward = 0

    for step in range(MAX_STEPS):
        action, log_prob, value = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.add(state, action, reward, float(done), log_prob, value)

        state = next_state
        episode_reward += reward

        if done:
            break

    episode_rewards.append(episode_reward)
    print(f"エピソード{episode+1}/{NUM_EPISODES}  報酬合計: {episode_reward:.2f}  ステップ数: {step+1}")

    if (episode + 1) % UPDATE_EVERY == 0:
        p_loss, v_loss = agent.update(buffer)
        policy_losses.append(p_loss)
        value_losses.append(v_loss)

env.close()

np.save("ppo_episode_rewards.npy", np.array(episode_rewards))
np.save("ppo_policy_losses.npy", np.array(policy_losses))
np.save("ppo_value_losses.npy", np.array(value_losses))
print("PPO学習完了。結果保存")







