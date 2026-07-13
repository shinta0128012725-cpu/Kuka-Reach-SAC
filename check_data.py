import numpy as np


#保存データの確認
data = np.load("collected_data.npz")
print("保存されているキー:", data.files)
print("statesの形:", data["states"].shape)
print("actionsの形:", data["actions"].shape)
print("rewardsの形:", data["rewards"].shape)
print("next_statesの形:", data["next_states"].shape)
print("donesの形:", data["dones"].shape)
print("\n最初のデータのreward:", data["rewards"][0])
print("最初のデータのaction:", data["actions"][0])