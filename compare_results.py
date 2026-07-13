import numpy as np
import matplotlib.pyplot as plt

random_data = np.load("collected_data.npz")
random_rewards = random_data["rewards"]

sac_rewards = np.load("kuka_episode_rewards.npy")

print("ランダム方策時の平均reward(1ステップあたり):", random_rewards.mean())
print("SAC学習後の平均reward(1エピソードあたり):", sac_rewards[-20:].mean())
print("SAC学習後の平均reward ÷ 50ステップ =", sac_rewards[-20:].mean() / 50)

plt.figure(figsize=(10, 4))
plt.plot(sac_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("SAC Training Progress on KukaReachEnv")
plt.axhline(y=sac_rewards[-20:].mean(), color="red", linestyle="--", label="Average of the last 20 episodes")
plt.legend()
plt.savefig("sac_training_progress.png", dpi=150)
print("sac_training_progress.pngとして保存 ")
plt.show()