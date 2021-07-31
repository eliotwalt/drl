import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from typing import List, Optional
from .models import FCQNetwork
from .utils import ReplayBuffer

class QAgent:
    def __init__(self, num_actions: int, fcs: List[int], num_episodes: int, epsilon: float, lr: float, 
                 gamma: float, batch_size: int, input_dim: int, max_length: int, device: torch.device,
                 dir: str, name):
        '''QAgent constructor
        Inputs:
        -------
        num_actions: int
            number of actions
        fcs: list
            number of neurons in each hidden fc layer (i.e excluding input and output)
        network: torch.Module
            torch network
        num_episodes: int
            number of episodes to train for
        epsilon: float [0, 1]
            probability of random action selection
        lr: float
            learning rate
        gamma: float
            discount factor
        batch_size: int
            batch size sampled from replay buffer
        input_dim: int
            dimensionality of the input
        max_length: int
            max_length of the replay buffer
        device: torch.device
            device to compute with
        name: Optional[str]
            name of the agent
        dir: str
            name of directory to save to
        
        '''
        assert epsilon >= 0 and epsilon <= 1, f'epsilon must be a probability, i.e in [0,1]'
        self.num_actions = num_actions
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.max_length = max_length
        self.device = device
        self.name = name
        self.dirname = os.path.join(dir, name)
        self.fcs = fcs
        self.network = FCQNetwork(input_dim, fcs, num_actions).to(device)
        self.replay_buffer = ReplayBuffer(batch_size, input_dim, max_length, device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr)
    
    def store(self, s, a, r, d):
        self.replay_buffer.store(s, a, r, d)

    def select_action(self, state):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def propagate(self, x, y):
        loss = self.criterion(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def copy_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def save_network(self):
        f = os.path.join(self.dirname, 'network')
        self.network.save(f)

class DQLAgent(QAgent):
    def __init__(self, **qagent_kwargs):
        '''DQLAgent constructor
        Inputs:
        -------
        qagent_kwargs: dict-like
            see QAgent
        '''
        super().__init__(**qagent_kwargs)

    def select_action(self, state):
        '''DQLAgent.select_action'''
        if random.uniform(0, 1) > self.epsilon:
            action = random.choice(range(self.num_actions))
        else:
            action = torch.argmax(self.network(state), dim=1).item()
        return action

    def learn(self):
        '''DQLAgent.learn'''
        states, actions, rewards, states_, dones = self.replay_buffer.sample()
        if states is not None:
            q = torch.zeros_like(rewards)
            q[dones!=1] = self.gamma*torch.max(self.network(states_[dones!=1]), dim=1)[0].detach()
            y = rewards + q
            x = self.network(states)[actions]
            self.propagate(x, y)
            return torch.mean(rewards).detach().item(), torch.mean(x).detach().item()

class DQNAgent(QAgent):
    def __init__(self, C: int, **qagent_kwargs):
        '''DQNAgent constructor
        Inputs:
        -------
        C: int
            frequency of target network update in number of iterations
        qagent_kwargs: dict-like
            see QAgent
        '''
        super().__init__(**qagent_kwargs)
        self.target_network = FCQNetwork(self.input_dim, self.fcs, self.num_actions).to(self.device)
        self.copy_target()
        self.C = C
        self.counter = 0

    def select_action(self, state):
        '''DQNAgent.select_action'''
        if random.uniform(0, 1) > self.epsilon:
            action = random.choice(range(self.num_actions))
        else:
            action = torch.argmax(self.network(state), dim=1).item()
        return action

    def learn(self):
        '''DQNAgent.learn'''
        states, actions, rewards, states_, dones = self.replay_buffer.sample()
        if states is not None:
            q = torch.zeros_like(rewards)
            q[dones!=1] = self.gamma*torch.max(self.target_network(states_[dones!=1]), dim=1)[0].detach()
            y = rewards + q
            x = self.network(states)[actions]
            self.propagate(x, y)
            self.counter += 1
            if self.counter % self.C == 0:
                self.tcopy_target()
            return torch.mean(rewards).detach().item(), torch.mean(x).detach().item()

class DDQNAgent(QAgent):
    def __init__(self, C: int, **qagent_kwargs):
        '''DDQNAgent constructor
        Inputs:
        -------
        C: int
            frequency of target network update in number of iterations
        qagent_kwargs: dict-like
            see QAgent
        '''
        super().__init__(**qagent_kwargs)
        self.target_network = FCQNetwork(self.input_dim, self.fcs, self.num_actions).to(self.device)
        self.copy_target()
        self.C = C
        self.counter = 0

    def select_action(self, state):
        '''DDQNAgent.select_action'''
        if random.uniform(0, 1) > self.epsilon:
            action = random.choice(range(self.num_actions))
        else:
            action = torch.argmax(self.network(state), dim=1).item()
        return action

    def learn(self):
        '''DDQNAgent.learn'''
        states, actions, rewards, states_, dones = self.replay_buffer.sample()
        if states is not None:
            q = torch.zeros_like(rewards)
            vals = self.network(states_[dones!=1])
            targ_vals = self.target_network(states_[dones!=1])
            idx = torch.argmax(vals, dim=1, keepdim=True)
            q[dones!=1] = self.gamma*torch.gather(targ_vals, 1, idx).detach()
            y = rewards + q
            x = torch.gather(self.network(states), 1, actions)
            self.propagate(x, y)
            self.counter += 1
            if self.counter % self.C == 0:
                self.copy_target()
            return torch.mean(rewards).detach().item(), torch.mean(x).detach().item()