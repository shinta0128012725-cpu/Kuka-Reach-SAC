import numpy as np
import matplotlib.pyplot as plt

data = np.load("collected_data.npz")
rewards = data["rewards"]
dones = data["dones"]

#報酬の可視化
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(rewards, bins=30, edgecolor="black")
plt.xlabel("Reward")
plt.ylabel("Frequency")
plt.title("Reward Distribution (Random Policy)")

episode_boundaries = np.where(dones == 1.0)[0]
print("エピソード終了位置(インデックス):", episode_boundaries)

plt.subplot(1, 2, 2)
plt.plot(rewards)
for boundary in episode_boundaries:
    plt.axvline(x=boundary, color="red", linestyle="--", alpha=0.3)
plt.xlabel("Step (across all episodes)")
plt.ylabel("Reward")
plt.title("Reward over Steps (red lines = episode end)")

plt.tight_layout()
plt.savefig("reward_visualization.png", dpi=150)
print("reward_visualization.png として保存しました")
plt.show()