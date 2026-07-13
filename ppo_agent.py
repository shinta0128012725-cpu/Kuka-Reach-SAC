import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        self.mu_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

        self.value_layer = nn.Linear(256, 1)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)

        value = self.value_layer(x)

        return mu, log_std, value
    

    def get_action(self, state):
        mu, log_std, value = self.forward(state)
        std = torch.exp(log_std)

        normal = torch.distributions.Normal(mu, std)
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)

        return action, log_prob, value
    
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []


class PPOAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, lr=3e-4):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio

        self.ac = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr)


    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = self.ac.get_action(state)

        action_np = action.detach().numpy()[0]
        action_np = np.clip(action_np, -1.0, 1.0)

        return action_np, log_prob.item(), value.item()
    

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return advantages
    

    def update(self, buffer, epochs=10, batch_size=64):
        states = torch.FloatTensor(np.array(buffer.states))
        actions = torch.FloatTensor(np.array(buffer.actions))
        old_log_probs = torch.FloatTensor(buffer.log_probs).unsqueeze(-1)
        rewards = buffer.rewards
        dones = buffer.dones
        values = buffer.values

        with torch.no_grad():
            _, _, next_value = self.ac.get_action(states[-1].unsqueeze(0))
            next_value = next_value.item()

            advantages = self.compute_gae(rewards, values, dones, next_value)
            advantages = torch.FloatTensor(advantages).unsqueeze(-1)
            returns = advantages + torch.FloatTensor(values).unsqueeze(-1)

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            mu, log_std, value = self.ac.forward(states)
            std = torch.exp(log_std)
            normal = torch.distributions.Normal(mu, std)
            new_log_probs = normal.log_prob(actions).sum(dim=-1, keepdim=True)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(value, returns)

            loss = policy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        buffer.clear()
        return policy_loss.item(), value_loss.item()









if __name__ == "__main__":
    state_dim = 34
    action_dim = 7

    ac = ActorCritic(state_dim, action_dim)
    dummy_state = torch.randn(1, state_dim)

    action, log_prob, value = ac.get_action(dummy_state)
    print("action:", action)
    print("action shape:", action.shape)
    print("log_prob shape:", log_prob.shape)
    print("value:", value)
    print("value shape:", value.shape)

