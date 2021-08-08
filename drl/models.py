import torch
import torch.nn as nn
from typing import List

class FCQNetwork(nn.Module):
    def __init__(self, input_dim: int, fcs: List[int], num_actions: int):
        '''FCQNetwork constructor
        Inputs:
        -------
        fcs: list
            number of neurons in each fc layer (not including the last one)
        num_actions: int
            number of actions
        '''
        super().__init__()
        layers = []
        fcs = [input_dim]+fcs+[num_actions]
        for i, fc in enumerate(fcs[:-1]):
            layers.append(nn.Linear(fc, fcs[i+1]))
            if i < len(fcs)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor):
        return self.net(states)

    def save(self, f):
        torch.save(self.state_dict(), f)

    def load_weights(self, f):
        self.load_state_dict(torch.load(f))

class FCACNetwork(nn.Module):
    def __init__(self, input_dim: int, fcs: List[int], num_actions: int):
        '''FCACNetwork constructor
        Inputs:
        -------
        fcs: list
            number of neurons in each fc layer (not including the last one)
        num_actions: int
            number of actions
        '''
        super().__init__()
        layers = []
        fcs = [input_dim]+fcs
        for i, fc in enumerate(fcs[:-1]):
            layers.append(nn.Linear(fc, fcs[i+1]))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.pi = nn.Sequential(nn.Linear(fcs[-1], num_actions), nn.Softmax(dim=-1))
        self.b = nn.Linear(fcs[-1], 1)

    def forward(self, states: torch.Tensor):
        x = self.net(states)
        action_probs = self.pi(x)
        baseline = self.b(x)
        return action_probs, baseline

    def save(self, f):
        torch.save(self.state_dict(), f)

    def load_weights(self, f):
        self.load_state_dict(torch.load(f))

class DuelingFCQNetwork(nn.Module):
    def __init__(self, input_dim: int, fcs: List[int], num_actions: int):
        '''DuelingFCQNetwork constructor
        Inputs:
        -------
        fcs: list
            number of neurons in each fc layer (not including the last one)
        num_actions: int
            number of actions
        '''
        super().__init__()
        layers = []
        fcs = [input_dim]+fcs
        for i, fc in enumerate(fcs[:-1]):
            layers.append(nn.Linear(fc, fcs[i+1]))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.advantage = nn.Linear(fcs[-1], num_actions)
        self.value = nn.Linear(fcs[-1], 1)

    def forward(self, states: torch.Tensor):
        x = self.net(states)
        adv = self.advantage(x)
        val = self.value(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def save(self, f):
        torch.save(self.state_dict(), f)

    def load_weights(self, f):
        self.load_state_dict(torch.load(f))
