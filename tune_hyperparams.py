import numpy as np
import optuna
from kuka_reach_env import KukaReachEnv
from sac_agent import SACAgent, ReplayBuffer


def run_training(gamma, tau, alpha, lr, num_episodes=200):
    env = KukaReachEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(state_dim=state_dim, action_dim=action_dim,
                      gamma=gamma, tau=tau, alpha=alpha, lr=lr)
    buffer = ReplayBuffer(capacity=100000, state_dim=state_dim, action_dim=action_dim)

    MAX_STEPS = 50
    START_STEPS = 1000
    BATCH_SIZE = 256
    total_steps = 0
    episode_rewards = []

    for episode in range(num_episodes):
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
                agent.update(buffer, batch_size=BATCH_SIZE)

            if done:
                break

        episode_rewards.append(episode_reward)

    env.close()

    recent = episode_rewards[-20:]
    avg_reward = sum(recent) / len(recent)
    return avg_reward


 


def objective(trial):
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    tau = trial.suggest_float("tau", 0.001, 0.02)
    alpha = trial.suggest_float("alpha", 0.05, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)

    success_rate = run_training(gamma, tau, alpha, lr)
    print(f"gamma={gamma:.4f}, tau={tau:.4f}, alpha={alpha:.4f}, lr={lr:.6f} -> success_rate={success_rate:.2f}")
    return success_rate


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)

    print("\n最善のハイパラ:")
    print(study.best_params)
    print("成功率:", study.best_value)