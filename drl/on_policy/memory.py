import torch

class AdvantageMemoryStack:
    def __init__(self, gamma, maxlen, num_agents, device):
        '''AdvantageMemoryStack constructor
        Inputs:
        -------
        gamma: float
            discount factor
        maxlen: int
            maximum length of memory
        num_agents: int
            number of agent memories
        device: torch.device
            device to send batch to
        '''
        self.num_agents = num_agents
        self.maxlen = maxlen
        self.num_agents = num_agents
        self.batch_size = maxlen*num_agents
        self.device = device
        self.memories = [AdvantageMemory(gamma, maxlen) for _ in range(self.num_agents)]

    def clear(self):
        '''AdvantageMemoryStack.clear: clear all memories'''
        for memory in self.memories:
            memory.clear()
    
    def store(self, rewards, actions, values, dones, dists):
        '''AdvantageMemoryStack.store: store transitions'''
        log_probs = dists.log_prob(actions)
        mean_entropies = dists.entropy().unsqueeze(-1)
        for (reward, value, done, log_prob, mean_entropy, memory) in zip(rewards, values, dones, log_probs, mean_entropies, self.memories):
            memory.store(reward, value, done, log_prob, mean_entropy)

    def get_batch(self):
        '''AdvantageMemoryStack.get_batch: get batch from all memories, stack them and shuffle them'''
        empty = True
        returns_batch = torch.empty(self.batch_size, 1)
        values_batch = torch.empty(self.batch_size, 1)
        log_probs_batch = torch.empty(self.batch_size, 1)
        mean_entropies_batch = torch.empty(self.batch_size, 1)
        for i in range(self.num_agents):
            returns, values, log_probs, mean_entropies = self.memories[i].get_batch()
            if returns is not None:
                empty = False
                returns_batch[i*self.maxlen:(i+1)*self.maxlen] = returns
                values_batch[i*self.maxlen:(i+1)*self.maxlen] = values
                log_probs_batch[i*self.maxlen:(i+1)*self.maxlen] = log_probs
                mean_entropies_batch[i*self.maxlen:(i+1)*self.maxlen] = mean_entropies
        if not empty:
            idx = torch.randperm(self.batch_size)
            return (returns_batch[idx].to(self.device), values_batch[idx].to(self.device),
                    log_probs_batch[idx].to(self.device), mean_entropies_batch[idx].to(self.device))
        else:
            return None, None, None, None

class AdvantageMemory:
    def __init__(self, gamma, maxlen):
        '''AdvantageMemory constructor
        Inputs:
        -------
        gamma: float
            discount factor
        maxlen: int
            maximum length of memory
        '''
        self.gamma = gamma
        self.maxlen = maxlen
        self.clear()

    def clear(self):
        ''''AdvantageMemory.clear: clears all tensor'''
        self.rewards = torch.empty(self.maxlen, 1)
        self.values = torch.empty(self.maxlen, 1)
        self.dones = torch.empty(self.maxlen, 1)
        self.log_probs = torch.empty(self.maxlen, 1)
        self.mean_entropies = torch.empty(self.maxlen, 1)
        self.pointer = 0

    def store(self, reward, value, done, log_prob, mean_entropy):
        '''AdvantageMemory.store: store a token in each tensor'''
        self.rewards[self.pointer] = reward
        self.values[self.pointer] = value
        self.dones[self.pointer] = done
        self.log_probs[self.pointer] = log_prob
        self.mean_entropies[self.pointer] = mean_entropy
        self.pointer += 1

    def get_batch(self):
        '''AdvantageMemory.get_batch: return None if tensors are not full
        and a batch if they are. Tensors are cleared after producing a batch.'''
        if self.pointer == self.maxlen:
            returns = torch.empty(self.maxlen, 1)
            R = self.values[-1]*(1-self.dones[-1]) # ?
            for t in reversed(range(self.maxlen)):
                R = self.rewards[t] + self.gamma*R*(1-self.dones[t]) # ?
                returns[self.maxlen-t-1] = R
            return_tuple = (returns, self.values, self.log_probs, self.mean_entropies) # really for log_probs ?
            self.clear()
            return return_tuple
        else:
            return None, None, None, None