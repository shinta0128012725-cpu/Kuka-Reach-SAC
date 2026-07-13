import numpy as np
import matplotlib.pyplot as plt

sac_rewards = np.load("kuka_episode_rewards.npy")
ppo_rewards = np.load("ppo_episode_rewards.npy")

print("SAC 直近20エピソード平均:", sac_rewards[-20:].mean())
print("PPO 直近20エピソード平均:", ppo_rewards[-20:].mean())

plt.figure(figsize=(10, 5))
plt.plot(sac_rewards, label="SAC", alpha=0.8)
plt.plot(ppo_rewards, label="PPO", alpha=0.8)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("SAC vs PPO on KukaReachEnv")
plt.legend()
plt.savefig("sac_vs_ppo.png", dpi=150)
print("sac_vs_ppo.png として保存しました")
plt.show()