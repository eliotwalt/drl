import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from typing import List
from ..models import FCQNetwork, DuelingFCQNetwork
from .utils import ReplayBuffer

class QAgent:
    def __init__(self, num_actions: int, fcs: List[int], num_episodes: int, epsilon: float, lr: float, 
                 gamma: float, batch_size: int, input_dim: int, max_length: int, device: torch.device,
                 dir: str, name: str):
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
        dir: str
            name of directory to save to
        name: str
            name of the agent
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
        self.path = os.path.join(dir, name)
        self.fcs = fcs
        self.network = FCQNetwork(input_dim, fcs, num_actions).to(device)
        self.replay_buffer = ReplayBuffer(batch_size, input_dim, max_length, device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr)
    
    def store(self, s: torch.Tensor, s_: torch.Tensor, a: int, r: float, d: bool):
        '''QAgent.store(s, a, r, d): store transition'''
        self.replay_buffer.store(s, s_, a, r, d)

    def select_action(self, state: torch.Tensor):
        '''QAgent.select_action: epsilon-greedy policy'''
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(range(self.num_actions))
            avg_q = 0
        else:
            q_values = self.network(state)
            action = torch.argmax(q_values, dim=1).item()
            avg_q = torch.mean(q_values, dim=1).item()
        return action, avg_q

    def learn(self):
        raise NotImplementedError

    def propagate(self, x: torch.Tensor, y: torch.Tensor):
        '''QAgent.propagate(x, y): compute and apply loss'''
        loss = self.criterion(x, y).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def copy_target(self):
        '''QAgent.copy_target(): online network into target network'''
        self.target_network.load_state_dict(self.network.state_dict())

    def save_network(self):
        '''QAgent.save_network(): saves network'''
        f = os.path.join(self.path, 'network.pth')
        self.network.save(f)

class DQLAgent(QAgent):
    def __init__(self, *qagent_args, **qagent_kwargs):
        '''DQLAgent constructor
        Inputs:
        -------
        qagent_args, q_agent_wkargs:
            see QAgent
        '''
        super().__init__(*qagent_args, **qagent_kwargs)

    def learn(self):
        '''DQLAgent.learn(): compute and apply update'''
        states, actions, rewards, states_, dones = self.replay_buffer.sample()
        if states is not None:
            q = torch.zeros_like(rewards)
            q[dones!=1] = self.gamma*torch.max(self.network(states_[dones!=1]), dim=1, keepdim=True)[0].detach()
            y = rewards + q
            x = torch.gather(self.network(states), 1, actions)
            self.propagate(x, y)

class DQNAgent(QAgent):
    def __init__(self, C: int, *qagent_args, **qagent_kwargs):
        '''DQNAgent constructor
        Inputs:
        -------
        C: int
            frequency of target network update in number of iterations
        qagent_args, q_agent_wkargs:
            see QAgent
        '''
        super().__init__(*qagent_args, **qagent_kwargs)
        self.target_network = FCQNetwork(self.input_dim, self.fcs, self.num_actions).to(self.device)
        self.copy_target()
        self.C = C
        self.counter = 0

    def learn(self):
        '''DQNAgent.learn(): compute and apply update'''
        states, actions, rewards, states_, dones = self.replay_buffer.sample()
        if states is not None:
            q = torch.zeros_like(rewards)
            q[dones!=1] = self.gamma*torch.max(self.target_network(states_[dones!=1]), dim=1, keepdim=True)[0].detach()
            y = rewards + q
            x = torch.gather(self.network(states), 1, actions)
            self.propagate(x, y)
            self.counter += 1
            if self.counter % self.C == 0:
                self.copy_target()

class DDQNAgent(QAgent):
    def __init__(self, C: int, *qagent_args, **qagent_kwargs):
        '''DDQNAgent constructor
        Inputs:
        -------
        C: int
            frequency of target network update in number of iterations
        qagent_args, q_agent_wkargs:
            see QAgent
        '''
        super().__init__(*qagent_args, **qagent_kwargs)
        self.target_network = FCQNetwork(self.input_dim, self.fcs, self.num_actions).to(self.device)
        self.copy_target()
        self.C = C
        self.counter = 0

    def learn(self):
        '''DDQNAgent.learn(): compute and apply update'''
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

class DuelingDQLAgent(DQLAgent):
    def __init__(self, *dql_args, **dql_kwargs):
        '''DuelingDQLAgent constructor
        Inputs:
        -------
        dql_args, dql_kwargs: 
            see DQLAgent
        '''
        super().__init__(*dql_args, **dql_kwargs)
        self.network = DuelingFCQNetwork(self.input_dim, self.fcs, self.num_actions).to(self.device)

class DuelingDQNAgent(DQNAgent):
    def __init__(self, *dqn_args, **dqn_kwargs):
        '''DuelingDQNAgent constructor
        Inputs:
        -------
        dqn_args, dqn_wkargs:
            see DQNAgent
        '''
        super().__init__(*dqn_args, **dqn_kwargs)
        self.network = DuelingFCQNetwork(self.input_dim, self.fcs, self.num_actions).to(self.device)
        self.target_network = DuelingFCQNetwork(self.input_dim, self.fcs, self.num_actions).to(self.device)

class DuelingDDQNAgent(DDQNAgent):
    def __init__(self, *ddqn_args, **ddqn_kwargs):
        '''DuelingDQNAgent constructor
        Inputs:
        -------
        ddqn_args, ddqn_wkargs:
            see DDQNAgent
        '''
        super().__init__(*ddqn_args, **ddqn_kwargs)
        self.network = DuelingFCQNetwork(self.input_dim, self.fcs, self.num_actions).to(self.device)
        self.target_network = DuelingFCQNetwork(self.input_dim, self.fcs, self.num_actions).to(self.device)
