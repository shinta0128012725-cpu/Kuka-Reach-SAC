import numpy as np
import optuna
from kuka_reach_env import KukaReachEnv
from ppo_agent import PPOAgent, RolloutBuffer


def run_training(gamma, gae_lambda, clip_ratio, lr, update_every, num_episodes=200):
    env = KukaReachEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim,
                      gamma=gamma, gae_lambda=gae_lambda, clip_ratio=clip_ratio, lr=lr)
    buffer = RolloutBuffer()

    MAX_STEPS = 50
    episode_rewards = []

    for episode in range(num_episodes):
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

        if (episode + 1) % update_every == 0:
            agent.update(buffer)

    env.close()

    recent = episode_rewards[-20:]
    avg_reward = sum(recent) / len(recent)
    return avg_reward


def objective(trial):
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_ratio = trial.suggest_float("clip_ratio", 0.1, 0.3)
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    update_every = trial.suggest_int("update_every", 2, 10)

    avg_reward = run_training(gamma, gae_lambda, clip_ratio, lr, update_every)
    print(f"gamma={gamma:.4f}, gae_lambda={gae_lambda:.4f}, clip_ratio={clip_ratio:.4f}, "
          f"lr={lr:.6f}, update_every={update_every} -> avg_reward={avg_reward:.2f}")
    return avg_reward


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)

    print("\n最も良かったパラメータ:")
    print(study.best_params)
    print("その時の成功率:", study.best_value)