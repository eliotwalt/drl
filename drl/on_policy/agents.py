import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import random
import os
from typing import List, Optional
from ..models import FCACNetwork
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
        '''ReinforceAgent.save_network(): saves network'''
        f = os.path.join(self.path, 'network.pth')
        self.network.save(f)

    def store(self, s: torch.Tensor, s_: torch.Tensor, a: int, r: float):
        '''ReinforceAgent.store(self, s, s_, a, r): store transition'''
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def empty(self):
        '''ReinforceAgent.empty(self): empty transitions lists'''
        self.states = []
        self.actions = []
        self.rewards = []
    
    def select_action(self, state: torch.Tensor):
        '''ReinforceAgent.select_action: sample action from policy'''
        action_probs, value = self.network(state)
        m = Categorical(action_probs)
        return m.sample().item(), value.item()

    def compute_returns(self):
        '''ReinforceAgent.compute_returns: compute s at each timestep'''
        returns = []
        ret = 0
        for r in self.rewards[::-1]:
            ret = r + self.gamma*ret
            returns.insert(0, ret)
        returns = torch.Tensor(returns).unsqueeze(-1)
        return (returns - returns.mean()) / (returns.std() + eps)
        
    def actor_loss(self, action_probs: torch.Tensor, returns: torch.Tensor, value: torch.Tensor):
        '''ReinforceAgent.actor_loss: compute actor loss'''
        m = Categorical(action_probs)
        log_probs = m.log_prob(torch.Tensor(self.actions).long().to(self.device))
        losses = [- log_prob * (return_ - value_.item())
                  for (log_prob, return_, value_) in zip(log_probs, returns, value)]
        return torch.stack(losses).sum()

    def critic_loss(self, value: torch.Tensor, returns: torch.Tensor):
        '''ReinforceAgent.critic_loss: compute value loss'''
        return (returns-value)**2
        
    def learn(self):
        '''ReinforceAgent.learn: update actor and critic network'''
        returns = self.compute_returns().to(self.device)
        states = torch.stack(self.states).squeeze(1).to(self.device)
        action_probs, value = self.network(states)
        loss = self.actor_loss(action_probs, returns, value) + self.critic_loss(value, returns)
        loss = loss.mean().to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class ActorCriticAgent:
    def __init__(self, num_actions: int, fcs: List[int], lr: float, gamma: float,
                 input_dim: int, device: torch.device, dir: str, name: str):
        '''ActorCriticAgent constructor
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
        self.reward = None
        self.done = None
        self.action_log_prob = None
        self.value = None

    def save_network(self):
        '''ActorCriticAgent.save_network(): saves network'''
        f = os.path.join(self.path, 'network.pth')
        self.network.save(f)

    def store(self, s: torch.Tensor, s_: torch.Tensor, a: int, r: float, d: bool):
        '''ActorCriticAgent.store(s, s_, a, r, d): store transition'''
        self.state = s.to(self.device)
        self.state_ = s_.to(self.device)
        self.action = torch.Tensor([a]).long().to(self.device)
        self.reward = torch.Tensor([r]).squeeze().float().to(self.device)
        self.done = d

    def select_action(self, state: torch.Tensor):
        '''ActorCriticAgent.select_action: sample action from policy'''
        action_probs, value = self.network(state)
        m = Categorical(action_probs)
        action = m.sample()
        return action.item(), value.item()

    def learn(self):
        '''ActorCritic.learn(s_): apply online update'''
        action_probs, value = self.network(self.state)
        value_ = 0. if self.done else self.network(self.state_)[1]
        m = Categorical(action_probs)
        action_log_prob = m.log_prob(self.action).to(self.device)
        delta = self.reward + self.gamma * value_ - value
        critic_loss = delta ** 2
        actor_loss = - delta * action_log_prob
        loss = critic_loss + actor_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
