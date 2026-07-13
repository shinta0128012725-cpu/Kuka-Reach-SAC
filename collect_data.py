import numpy as np
from kuka_reach_env import KukaReachEnv

#エピソード・ステップ数の設定
NUM_EPISODES = 20
MAX_STEPS_PER_EPISODE = 50

def collect_data():
    env = KukaReachEnv()

    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []

    for episode in range(NUM_EPISODES):
        obs, info = env.reset()

        for step_num in range(MAX_STEPS_PER_EPISODE):
            #ランダム行動で取得
            action = env.action_space.sample() 
            next_obs, reward, terminated, truncated, info = env.step(action)

            #データ格納
            all_states.append(obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_states.append(next_obs)
            all_dones.append(terminated or truncated)
            
            obs = next_obs

            if terminated or truncated:
                break
            
        print(f"エピソード{episode+1}/{NUM_EPISODES} 完了(ステップ数: {step_num+1})")

    env.close()
    print(f"\n収集完了。合計データ数: {len(all_states)}")

    # データ保存
    states_array = np.array(all_states, dtype=np.float32)
    actions_array = np.array(all_actions, dtype=np.float32)
    rewards_array = np.array(all_rewards, dtype=np.float32)
    next_states_array = np.array(all_next_states, dtype=np.float32)
    dones_array = np.array(all_dones, dtype=np.float32)

    np.savez(
        "collected_data.npz",
        states=states_array,
        actions=actions_array,
        rewards=rewards_array,
        next_states=next_states_array,
        dones=dones_array,
    )
    print("collected_data.npz の保存完了")

    return all_states, all_actions, all_rewards, all_next_states, all_dones


if __name__ == "__main__":
    states, actions, rewards, next_states, dones = collect_data()
        

