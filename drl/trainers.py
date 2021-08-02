import torch
import gym
import json
import os

class Trainer:
    def __init__(self, env: gym.Env, num_episodes: int, agent, num_seeds: int, pid: int=0):
        '''OnlineTrainer constructor
        Inputs:
        -------
        env: gym.env
            Gym environment
        num_episodes: int
            Number of iterations per epsiode
        agent: QAgent
            QAgent instance
        num_seeds: int
            Number of different seeds to use
        pid: int
            Process id (optional)
        '''
        self.env = env
        self.num_episodes = num_episodes
        self.agent = agent
        self.num_seeds = num_seeds
        self.pid = pid
        self.prid = '[{}]'.format(pid)
        self.avg_rewards = []
        self.avg_values = []
        self.max_reward = -float('inf')

    def run(self):
        '''Trainer.run: run all episodes'''
        for episode in range(self.num_episodes):
            self.run_epsiode()
            if (episode-1) % 10 == 0:
                print('{} episode {}/{} - average reward: {} average val: {}'.format(
                    self.prid, episode, self.num_episodes, 
                    self.avg_rewards[-1], self.avg_values[-1]
                ))
            if self.avg_rewards[-1] > self.max_reward:
                self.agent.save_network()
                self.save_metrics()
                self.max_reward = self.avg_rewards[-1]
        return self.summary()
    
    def summary(self):
        '''Trainer.summary: return dictionary of metrics'''
        return {
            'avg_rewards': self.avg_rewards,
            'avg_value': self.avg_values,
        }

    def save_metrics(self):
        resfile = os.path.join(self.agent.path, 'metrics.json')
        with open(resfile, 'w') as f:
            json.dump(self.summary(), f)

class OnlineTrainer(Trainer):
    def __init__(self, *trainer_args, **trainer_kwargs):
        '''OnlineTrainer constructor
        Inputs:
        -------
        trainer_args, trainer_kwargs: 
            see Trainer
        '''
        super().__init__(*trainer_args, **trainer_kwargs)

    def run_epsiode(self):
        '''OnlineTrainer.run_epsiode: run one episode'''
        state = self.env.reset()
        state = torch.from_numpy(state).to(torch.float32).unsqueeze(0)
        done = False
        episode_reward = 0
        episode_value = 0
        n = 0
        while not done:
            action = self.agent.select_action(state.to(self.agent.device))
            state_, reward, done, _ = self.env.step(action)
            state_ = torch.from_numpy(state_).to(torch.float32).unsqueeze(0)
            self.agent.store(state, action, reward, done)
            state = state_.clone()
            avg_reward, avg_value = self.agent.learn()
            episode_reward += avg_reward
            episode_value += avg_value
            n += 1
        self.avg_rewards.append(episode_reward/n)
        self.avg_values.append(episode_value/n)

class EpisodicTrainer(Trainer):
    def __init__(self, *trainer_args, **trainer_kwargs):
        '''EpisodicTrainer constructor
        Inputs:
        -------
        trainer_args, trainer_kwargs: 
            see Trainer
        '''
        super().__init__(*trainer_args, **trainer_kwargs)
    
    def run_epsiode(self):
        '''EpisodicTrainer.run_episode: run one episode'''
        state = self.env.reset()
        state = torch.from_numpy(state).to(torch.float32).unsqueeze(0)
        done = False
        while not done:
            action = self.agent.select_action(state.to(self.agent.device))
            state_, reward, done, _ = self.env.step(action)
            state_ = torch.from_numpy(state_).to(torch.float32).unsqueeze(0)
            self.agent.store(state, action, reward)
            state = state_.clone()
        avg_reward, avg_value = self.agent.learn()
        self.avg_rewards.append(avg_reward)
        self.avg_values.append(avg_value)
        self.agent.empty()
