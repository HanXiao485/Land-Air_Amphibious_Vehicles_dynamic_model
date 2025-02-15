from torch import nn
import torch
import numpy as np
from torch.distributions import Normal
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"computing device:{device}")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.tanh(self.fc_mean(x)) * 2
        std = self.softplus(self.fc_std(x)) + 1e-3
        
        return mean, std
    
    def select_action(self, s):
        with torch.no_grad():
            mu, sigma = self.forward(s)
            normal_dist = Normal(mu, sigma)
            action = normal_dist.sample()
            action = action.clamp(-2, 2)
        
        return action
        
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)

        return value
        
class ReplayMemory:
    def __init__(self, batch_size):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []
        self.BATCH_SIZE = batch_size

    def add_memory(self, state, action, reward, value, done):
        self.state_cap.append(state)
        self.action_cap.append(action)
        self.reward_cap.append(reward)
        self.value_cap.append(value)
        self.done_cap.append(done)

    def sample(self):
        num_state = len(self.state_cap)
        batch_start_points = np.arange(0, num_state, self.BATCH_SIZE)
        memory_indices = np.arange(num_state, dtype=np.int32)
        np.random.shuffle(memory_indices)
        batches = [memory_indices[i:i+self.BATCH_SIZE] for i in batch_start_points]

        return self.state_cap, self.action_cap, self.reward_cap, self.value_cap, self.done_cap, batches
    
    def clear_memory(self):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []
        
class PPOAgent:
    def __init__(self, state_dim, action_dim, batch_size):
        # self.LR_ACTOR = 3e-4
        # self.LR_CRITIC = 1e-3
        self.LR_ACTOR = 3e-3
        self.LR_CRITIC = 1e-2
        self.GAMMA = 0.99
        self.LAMBDA = 0.95 
        self.EPOCH = 10
        self.EPSILON_CLIP = 0.2

        self.actor = Actor(state_dim, action_dim).to(device)
        self.old_actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)
        self.replay_buffer = ReplayMemory(batch_size)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor.select_action(state) 
        value = self.critic.forward(state)

        return action.detach().cpu().numpy()[0] , value.detach().cpu().numpy()[0]
    
    def update(self):
        # Update actor
        for epoch_i in range(self.EPOCH):
            memo_states, memo_actions, memo_rewards, memo_value, memo_dones, batches = self.replay_buffer.sample()
            T = len(memo_rewards)
            # Calculate advantages
            memo_advantages = np.zeros(T, dtype=np.float32)
            
            for t in range(T):
                discount = 1
                a_t = 0
                for k in range(t, T-1):
                    a_t += memo_rewards[k] *self.GAMMA * memo_value[k+1] * (1-int(memo_dones[k])) - memo_value[k]
                    discount *= self.GAMMA * self.LAMBDA
                memo_advantages[t] = a_t
            
            with torch.no_grad():
                memo_advantages_tensor = torch.tensor(memo_advantages).unsqueeze(1).to(device)
                memo_values_tensor = torch.tensor(memo_value).to(device)
                
            memo_states_tensor = torch.FloatTensor(memo_states).to(device)
            memo_actions_tensor = torch.FloatTensor(memo_actions).to(device)

            for batch in batches:
                with torch.no_grad():
                    old_mu, old_sigma = self.old_actor(memo_states_tensor[batch])
                    old_pi = Normal(loc=old_mu, scale=old_sigma)
                batch_old_probs_tensor = old_pi.log_prob(memo_actions_tensor[batch])

                mu, sigma = self.actor(memo_states_tensor[batch])
                pi = Normal(loc=mu, scale=sigma)
                batch_probs_tensor = pi.log_prob(memo_actions_tensor[batch])

                ratio = torch.exp(batch_probs_tensor - batch_old_probs_tensor)

                surr1 = ratio * memo_advantages_tensor[batch]
                surr2 = torch.clamp(ratio, 1.0 - self.EPSILON_CLIP, 1.0 + self.EPSILON_CLIP) * memo_advantages_tensor[batch]

                actor_loss = -torch.min(surr1, surr2).mean()

                batch_returns = memo_advantages_tensor[batch] + memo_values_tensor[batch]

                batch_old_values = self.critic(memo_states_tensor[batch])

                critic_loss = nn.MSELoss()(batch_old_values, batch_returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        self.replay_buffer.clear_memory()

    def save_policy(self):
        torch.save(self.actor.state_dict(), f"ppo_policy_pendulum_v1.pare")

