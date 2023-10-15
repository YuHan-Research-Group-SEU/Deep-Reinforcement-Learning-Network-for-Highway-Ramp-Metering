import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

device = torch.device('cpu')

class Actor(nn.Module):
    def __init__(self, N_STATES):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        action = torch.tanh(self.out(x))

        return action

class Critic(nn.Module):
    def __init__(self,N_STATES):
        super(Critic, self).__init__()
        self.fc11 = nn.Linear(N_STATES+1, 128)
        self.fc12 = nn.Linear(128, 64)
        self.fc13 = nn.Linear(64, 32)
        self.fc14 = nn.Linear(32, 16)
        self.out1 = nn.Linear(16, 1)

        self.fc21 = nn.Linear(N_STATES+1, 128)
        self.fc22 = nn.Linear(128, 64)
        self.fc23 = nn.Linear(64, 32)
        self.fc24 = nn.Linear(32, 16)
        self.out2 = nn.Linear(16, 1)

    def forward(self, s, a):
        sa = torch.cat([s,a], 1)
        x = F.relu(self.fc11(sa))
        x = F.relu(self.fc12(x))
        x = F.relu(self.fc13(x))
        x = F.relu(self.fc14(x))
        Q1 = self.out1(x)

        x = F.relu(self.fc21(sa))
        x = F.relu(self.fc22(x))
        x = F.relu(self.fc23(x))
        x = F.relu(self.fc24(x))
        Q2 = self.out2(x)

        return Q1, Q2

class MemoryBuffer(object):
    def __init__(self, MEMORY_SIZE, N_STATES):
        self.buffer = np.zeros([MEMORY_SIZE, 2 * N_STATES + 2])
        self.memory_size = MEMORY_SIZE
        self.n_states = N_STATES
        self.next_idx = 0
        self.memory_counter = 0
        self.mean = None
        self.std = None

    def push(self, state, action, reward, next_state):
        data = np.concatenate((state, action, np.array([reward]), next_state))
        self.buffer[self.next_idx, :] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size
        self.memory_counter += 1
        if self.memory_counter == self.memory_size:
            self.mean = np.mean(self.buffer[:, :self.n_states], axis=0)
            self.std = np.std(self.buffer[:, :self.n_states], axis=0)
            pass

    def sample(self, batch_size):
        sample_index = np.random.choice(self.memory_size, batch_size, replace=False)
        batch_memory = self.buffer[sample_index, :]
        states = torch.tensor(batch_memory[:, :self.n_states], dtype=torch.float32).to(device)
        actions = torch.tensor(batch_memory[:, self.n_states:self.n_states+1], dtype=torch.float32).to(device)
        rewards = torch.tensor(batch_memory[:, self.n_states+1:self.n_states+2], dtype=torch.float32).to(device)
        next_states = torch.tensor(batch_memory[:, self.n_states+2:], dtype=torch.float32).to(device)

        return states, actions, rewards, next_states

class TD3Agent(object):
    def __init__(self, MEMORY_SIZE, N_STATES, LR_ACTOR, LR_CRITIC, BATCH_SIZE, GAMMA, TAU, VAR, POLICY_NOISE, NOISE_CLIP, POLICY_FREQ):
        self.actor_eval = Actor(N_STATES).to(device)
        self.actor_target = Actor(N_STATES).to(device)
        self.critic_eval = Critic(N_STATES).to(device)
        self.critic_target = Critic(N_STATES).to(device)
        self.memory = MemoryBuffer(MEMORY_SIZE, N_STATES)
        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=LR_CRITIC)
        self.critic_crterion = nn.MSELoss(reduction='mean')
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.critic_target.load_state_dict(self.critic_eval.state_dict())
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.tau = TAU
        self.var = VAR
        self.policy_noise = POLICY_NOISE
        self.noise_clip = NOISE_CLIP
        self.policy_freq = POLICY_FREQ
        self.learn_counter = 0
        self.actor_loss = None
        self.critic_loss = None
        self.a_l = []
        self.c_l = []

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = self.actor_eval(state).detach().numpy()
        return action

    def learn(self):
        self.learn_counter += 1
        states, actions, rewards, next_states = self.memory.sample(self.batch_size)

        def critic_learn():
            noise = torch.ones_like(actions).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            actions_next = self.actor_target(next_states).detach() + noise.detach()
            actions_next = actions_next.clamp(-1, 1)
            q1_for_next_state, q2_for_next_state = self.critic_target(next_states, actions_next)
            q_for_next_state = torch.min(q1_for_next_state, q2_for_next_state)
            q_target = rewards + self.gamma * q_for_next_state
            q1, q2 = self.critic_eval(states, actions)
            q_pred = torch.min(q1, q2)
            loss = self.critic_crterion(q_pred, q_target)
            self.c_l.append(loss.item())
            self.critic_loss = np.array(self.c_l)
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

        def actor_learn():
            q1, q2 = self.critic_eval(states, self.actor_eval(states))
            loss = -torch.min(q1, q2).mean()
            self.a_l.append(loss.item())
            self.actor_loss = np.array(self.a_l)
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

        def soft_update(net, net_target):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0- self.tau) + param.data * self.tau)

        critic_learn()
        if self.learn_counter % self.policy_freq == 0:
            actor_learn()
            soft_update(self.critic_eval, self.critic_target)
            soft_update(self.actor_eval, self.actor_target)