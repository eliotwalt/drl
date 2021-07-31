import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import random
import os
from typing import List, Optional
from .models import FCACNetwork

class ReinforceAgent:
    def __init__(self, num_actions: int, fcs: List[int], lra: float, lrc: float, gamma: float,
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
        lra: float
            learning rate for actor
        lrc: float
            learning rate for critic
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
        self.lra = lra
        self.lrc = lrc
        self.gamma = gamma
        self.input_dim = input_dim
        self.device = device
        self.name = name
        self.dirname = os.path.join(dir, name)
        self.fcs = fcs
        self.network = FCACNetwork(input_dim, fcs, num_actions).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), 1)
        self.empty()

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
        m = Categorical(probs=action_probs)
        return m.sample()

    def compute_returns(self):
        '''Reinforce.compute_returns: compute s at each timestep'''
        T = len(self.rewards)
        returns = torch.empty(T, 1).to(torch.float32)
        for i in range(T):
            r = torch.Tensor(self.rewards[i:]).to(torch.float32)
            g = torch.Tensor([self.gamma**(k-i-1) for k in range(i+1, T+1]).to(torch.float32)
            returns[i] = r.dot(g)
        return returns
    
    def actor_loss(self, action_probs, returns):
        '''Reinforce.actor_loss: compute actor loss'''
        log_probs = torch.log(action_probs)
        return self.lrb * returns*log_probs

    def baseline_loss(self, baseline, returns):
        '''Reinforce.baseline_loss: compute baseline loss'''
        return self.lra * (torch.pow(baseline-returns, 2)

    def learn(self):
        '''Reinforce.learn: update actor and critic network'''
        returns = self.compute_returns().to(self.device)
        states = torch.Tensor(self.states).to(torch.float32).to(self.device)
        action_probs, baseline = self.network(states)
        loss = self.actor_loss(action_probs, returns) + self.baseline_loss(baseline, returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.empty()
