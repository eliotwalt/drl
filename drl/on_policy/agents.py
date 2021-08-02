import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import random
import os
from typing import List, Optional
from .models import FCACNetwork
import numpy as np

eps = np.finfo(np.float32).eps.item()

class ReinforceAgent:
    def __init__(self, num_actions: int, fcs: List[int], lr: float, gamma: float,
                 input_dim: int, device: torch.device, dir: str, name: str):
        '''ReinforceAgent constructor
        Inputs:
        -------
        num_actions: int
            number of actions
        fcs: list
            number of neurons in each hidden fc layer (i.e excluding input and output)
        num_episodes: int
            number of episodes to train for
        lr: float
            learning rate
        gamma: float
            discount factor
        input_dim: int
            dimensionality of the input
        device: torch.device
            device to compute with
        dir: str
            name of directory to save to
        name: str
            name of the agent
        '''
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.input_dim = input_dim
        self.device = device
        self.name = name
        self.path = os.path.join(dir, name)
        self.fcs = fcs
        self.network = FCACNetwork(input_dim, fcs, num_actions).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr)
        self.empty()

    def save_network(self):
        f = os.path.join(self.path, 'network.pth')
        self.network.save(f)

    def store(self, s, a, r):
        '''Reinforce.store(self, s, a, r): store transition'''
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def empty(self):
        '''Reinforce.empty(self): empty transitions lists'''
        self.states = []
        self.actions = []
        self.rewards = []
    
    def select_action(self, state):
        '''Reinforce.select_action: sample action from policy'''
        action_probs, _ = self.network(state)
        m = Categorical(action_probs)
        return m.sample().item()

    def compute_returns(self):
        '''Reinforce.compute_returns: compute s at each timestep'''
        returns = []
        ret = 0
        for r in self.rewards[::-1]:
            ret= r + self.gamma*ret
            returns.insert(0, ret)
        returns = torch.Tensor(returns).unsqueeze(-1)
        return (returns - returns.mean()) / (returns.std() + eps)
        
    def actor_loss(self, action_probs, returns, baseline):
        '''Reinforce.actor_loss: compute actor loss'''
        m = Categorical(action_probs)
        log_probs = m.log_prob(torch.Tensor(self.actions).long().to(self.device))
        losses = [- log_prob * (return_ - baseline_.item())
                for (log_prob, return_, baseline_) in zip(log_probs, returns, baseline)]
        return torch.stack(losses).sum()

    def baseline_loss(self, baseline, returns):
        '''Reinforce.baseline_loss: compute baseline loss'''
        # return F.mse_loss(baseline, returns, reduction='sum')
        return (returns-baseline)**2
        
    def learn(self):
        '''Reinforce.learn: update actor and critic network'''
        returns = self.compute_returns().to(self.device)
        states = torch.stack(self.states).squeeze(1).to(self.device)
        action_probs, baseline = self.network(states)
        loss = self.actor_loss(action_probs, returns, baseline) + self.baseline_loss(baseline, returns)
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return torch.sum(torch.Tensor(self.rewards)).detach().item(), torch.mean(baseline).detach().item()
