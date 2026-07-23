import numpy as np
import matplotlib.pyplot as plt

data = np.load("collected_data.npz")
rewards = data["rewards"]

# 50ステップごとに区切って、エピソード単位の合計報酬を計算
episode_rewards_random = rewards.reshape(-1, 50).sum(axis=1)

print("ランダム方策、エピソード単位の平均報酬:", episode_rewards_random.mean())

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(episode_rewards_random, bins=10, edgecolor="black")
plt.xlabel("Total Reward (per episode)")
plt.ylabel("Frequency")
plt.title("Reward Distribution (Random Policy)")

plt.subplot(1, 2, 2)
plt.plot(episode_rewards_random, marker="o")
plt.xlabel("Episode")
plt.ylabel("Total Reward (per episode)")
plt.title("Reward per Episode (Random Policy)")

plt.tight_layout()
plt.savefig("reward_visualization.png", dpi=150)
print("reward_visualization.png として保存しました")
plt.show()