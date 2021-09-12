import torch

class ReplayBuffer:
    def __init__(self, batch_size: int, dim: int, max_length: int, device: torch.device):
        '''ReplayBuffer constructor
        Inputs:
        -------
        batch_size: int
            size of the batches to sample
        dim: int
            dimensionality of the states
        max_length: int
            maximum size of the replay buffer
        device: torch.device
            device to use
        '''
        self.batch_size = batch_size
        self.dim = dim
        self.max_length = max_length
        self.device = device
        self.states = torch.empty(self.max_length, self.dim).to(torch.float32)
        self.states_ = torch.empty(self.max_length, self.dim).to(torch.float32)
        self.actions = torch.empty(self.max_length, 1).to(torch.long)
        self.rewards = torch.empty(self.max_length, 1).to(torch.float32)
        self.dones = torch.empty(self.max_length).to(bool)
        self.pointer = None

    def store(self, s: torch.Tensor, s_: torch.Tensor, a: int, r: float, d: bool):
        '''ReplayBuffer.store method: store state/action/value
        Inputs:
        -------
        s: torch.Tensor
            state tensor (from which action a is taken)
        s_: torch.Tensor
            next state tensor (resulting from taking action a at state s)
        a: int
            action index
        r: float
            reward value
        '''
        if self.pointer == None or self.pointer == self.max_length:
            self.pointer = 0
        self.states[self.pointer] = s
        self.states_[self.pointer] = s_ 
        self.actions[self.pointer] = a
        self.rewards[self.pointer] = r
        self.dones[self.pointer] = d
        self.pointer += 1

    def sample(self):
        '''ReplayBuffer.sample method: sample a minibatch'''
        if self.pointer == None: # At least, two states
            return None, None, None, None, None
        else:
            idx = torch.randperm(self.pointer)[:self.batch_size]
            s = self.states[idx]
            a = self.actions[idx]
            r = self.rewards[idx]
            s_ = self.states_[idx]
            d = self.dones[idx]
            return s.to(self.device), a.to(self.device), r.to(self.device), s_.to(self.device), d

    def clear(self):
        '''ReplayBuffer.clear method: clear all tensors'''
        self.states = torch.empty(self.max_length)
        self.actions = torch.empty(self.max_length)
        self.rewards = torch.empty(self.max_length)