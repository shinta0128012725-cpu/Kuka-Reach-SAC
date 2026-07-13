import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        self.mu_layer = nn.Linear(256, action_dim)
        # 標準偏差は、マイナスで機能しないためlogを通す
        self.log_std_layer = nn.Linear(256, action_dim)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        #-20 ~ 2は、経験則からのハイパラ
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mu, log_std
    
    # サンプリング
    def sample(self, state):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)

        #正規分布の作成
        normal = torch.distributions.Normal(mu, std)
        # Reparameterization trick
        xi = normal.rsample()

        action = torch.tanh(xi)

        # log πの計算
        log_prob = normal.log_prob(xi)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q_layer = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_layer(x)
        return q_value
    


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        #メモリ効率のため、枠をあらかじめ用意
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)


    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.states[idx]),
            torch.FloatTensor(self.actions[idx]),
            torch.FloatTensor(self.rewards[idx]),
            torch.FloatTensor(self.next_states[idx]),
            torch.FloatTensor(self.dones[idx]),
        )
    

class SACAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        #Actor
        self.actor = Actor(state_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        #Critic(main)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        #Critic(target)
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    #pybullet用にnpへ変換
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _ = self.actor.sample(state)
        return action.detach().numpy()[0]
    

    #Critic更新→Actor更新→Targetnetwork更新
    def update(self, replay_buffer, batch_size=256):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        #Critic更新
        critic1_loss, critic2_loss = self._update_critic(states, actions, rewards, next_states, dones)
        #Actor更新
        actor_loss = self._update_actor(states)
        #Target network更新
        self._update_target_networks()

        return critic1_loss, critic2_loss, actor_loss

    def _update_critic(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target = self.critic1_target(next_states, next_actions)
            q2_target = self.critic2_target(next_states, next_actions)
            q_target_min = torch.min(q1_target, q2_target)
            target_q = rewards + self.gamma * (1 - dones) * (q_target_min - self.alpha * next_log_probs)

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        return critic1_loss.item(), critic2_loss.item()

    def _update_actor(self, states):
        actions, log_probs = self.actor.sample(states)

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q_min = torch.min(q1, q2)

        actor_loss = (self.alpha * log_probs - q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def _update_target_networks(self):
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



#テスト
if __name__ == "__main__":
    state_dim = 34
    action_dim = 7

    actor = Actor(state_dim, action_dim)
    critic1 = Critic(state_dim, action_dim)
    critic2 = Critic(state_dim, action_dim)

    dummy_state = torch.randn(1, state_dim)

    action, log_prob = actor.sample(dummy_state)
    print("action shape:", action.shape)
    print("log_prob shape:", log_prob.shape)

    q1_value = critic1(dummy_state, action)
    q2_value = critic2(dummy_state, action)
    print("Q1 value:", q1_value)
    print("Q2 value:", q2_value)

    buffer = ReplayBuffer(capacity=1000, state_dim=state_dim, action_dim=action_dim)

    for _ in range(50):
        dummy_s = np.random.randn(state_dim).astype(np.float32)
        dummy_a = np.random.uniform(-1, 1, action_dim).astype(np.float32)
        dummy_r = np.random.randn()
        dummy_s2 = np.random.randn(state_dim).astype(np.float32)
        dummy_done = 0.0
        buffer.add(dummy_s, dummy_a, dummy_r, dummy_s2, dummy_done)

    print("\nバッファに貯まっているデータ数:", buffer.size)

    states, actions, rewards, next_states, dones = buffer.sample(batch_size=16)
    print("サンプリングしたstatesの形:", states.shape)
    print("サンプリングしたactionsの形:", actions.shape)
    print("サンプリングしたrewardsの形:", rewards.shape)

    # ここから今回追加した部分(同じインデントの深さで続ける)
    print("\n--- SACAgent全体の動作確認 ---")
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim)

    dummy_state_np = np.random.randn(state_dim).astype(np.float32)
    selected_action = agent.select_action(dummy_state_np)
    print("選択された行動:", selected_action)
    print("行動の形:", selected_action.shape)

    for _ in range(300):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.uniform(-1, 1, action_dim).astype(np.float32)
        r = np.random.randn()
        s2 = np.random.randn(state_dim).astype(np.float32)
        d = 0.0
        buffer.add(s, a, r, s2, d)

    print("\nupdate実行前のQ1ネットワークの一部の重み:")
    print(list(agent.critic1.parameters())[0][0][:5])

    agent.update(buffer, batch_size=256)

    print("\nupdate実行後のQ1ネットワークの一部の重み:")
    print(list(agent.critic1.parameters())[0][0][:5])

    print("\nエラーなくupdateが完了しました")