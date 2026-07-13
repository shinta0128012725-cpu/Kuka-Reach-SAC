import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

critic1_losses = np.load("kuka_critic1_losses.npy")
critic2_losses = np.load("kuka_critic2_losses.npy")
actor_losses = np.load("kuka_actor_losses.npy")

print("記録されたupdate回数:", len(critic1_losses))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(critic1_losses, label="Critic1 Loss", alpha=0.7)
axes[0].plot(critic2_losses, label="Critic2 Loss", alpha=0.7)
axes[0].set_xlabel("Update Step")
axes[0].set_ylabel("Loss")
axes[0].set_title("Critic Loss")
axes[0].legend()

axes[1].plot(actor_losses, label="Actor Loss", color="green", alpha=0.7)
axes[1].set_xlabel("Update Step")
axes[1].set_ylabel("Loss")
axes[1].set_title("Actor Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("sac_losses.png", dpi=150)
print("sac_losses.png として保存しました")
plt.show()