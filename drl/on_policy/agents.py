import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import random
import os
from typing import List, Optional
from ..models import FCACNetwork

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
        '''ReinforceAgent.save_network: saves network'''
        f = os.path.join(self.path, 'network.pth')
        self.network.save(f)

    def store(self, s: torch.Tensor, s_: torch.Tensor, a: int, r: float):
        '''ReinforceAgent.store: store transition'''
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def empty(self):
        '''ReinforceAgent.empty: empty transitions lists'''
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
        self.value = None
        self.empty()

    def save_network(self):
        '''ActorCriticAgent.save_network: saves network'''
        f = os.path.join(self.path, 'network.pth')
        self.network.save(f)

    def store(self, s: torch.Tensor, s_: torch.Tensor, a: int, r: float, d: bool):
        '''ActorCriticAgent.store: store transition'''
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
        '''ActorCritic.learn: apply online update'''
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

class A2CAgent:
    def __init__(self, num_actions: int, fcs: List[int], lr: float, gamma: float, num_envs: int, critic_coeff: float,
                 max_iters: int, beta: float, input_dim: int, device: torch.device, dir: str, name: str):
        '''A2CAgent constructor
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
        num_envs: int
            number of environment the agent received observations from
        critic_coeff: float
            coefficient for critic loss
        max_iters: int
            number of iteration before updating
        beta: float
            entropy term control parameter
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
        self.num_envs = num_envs
        self.critic_coeff = critic_coeff
        self.max_iters = max_iters
        self.beta = beta
        self.input_dim = input_dim
        self.device = device
        self.name = name
        self.path = os.path.join(dir, name)
        self.fcs = fcs
        self.network = FCACNetwork(input_dim, fcs, num_actions).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr)
        self.clear()

    def save_network(self):
        '''ActorCriticAgent.save_network: saves network'''
        f = os.path.join(self.path, 'network.pth')
        self.network.save(f)

    def clear(self):
        '''A2CAgent.clear: clear memories'''
        self.log_probs = []
        self.values = []
        self.next_values = []
        self.rewards = []
        self.dones = []
        self.entropies = []

    def to_torch(self):
        '''A2CAgent.to_torch: turn memories into torch tensors'''
        self.log_probs = torch.cat(self.log_probs).to(self.device)
        self.values = torch.cat(self.values).to(self.device)
        self.next_values = torch.cat(self.next_values).to(self.device)
        self.rewards = torch.cat(self.rewards).to(self.device)
        self.dones = torch.cat(self.dones).to(self.device)
        self.entropies = torch.Tensor(self.entropies).to(self.device)
    
    def store(self, rewards, actions, values, next_values, dones, dists):
        '''A2CAgent.store: store transitions'''
        self.log_probs.append(dists.log_prob(actions))
        self.values.append(values.squeeze())
        self.next_values.append(next_values.squeeze())
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.entropies.append(dists.entropy().mean())


    def select_actions(self, states: torch.Tensor):
        '''A2CAgent.select_actions: select actions based on states'''
        actions_probs, values = self.network(states)
        dists = Categorical(actions_probs)
        actions = dists.sample()
        return actions, values, dists

    def compute_returns(self):
        '''A2CAgent.compute_returns: compute batch of returns'''
        returns = []
        R = self.next_values[-1]*(1-self.dones[-1])
        for t in reversed(range(len(self.rewards))):
            R = self.rewards[t] + self.gamma * R * (1-self.dones[t])
            returns.insert(0,R)
        return torch.cat(returns).to(self.device)

    def learn(self):
        '''A2CAgent.learn: apply update'''
        R = self.compute_returns()
        self.to_torch()
        advantages = R - self.values
        actor_loss = -(self.log_probs*advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy = self.entropies.mean()
        loss = actor_loss + self.critic_coeff*critic_loss - self.beta*entropy
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
        self.optimizer.step()
        self.clear()

class A3CWorker:
    def __init__(self, ):
        pass

class A3CAgent:
    def __init__(self, ):
        pass