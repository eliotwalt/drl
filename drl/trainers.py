import torch
import gym
import json
import os

class Trainer:
    def __init__(self, env: gym.Env, num_episodes: int, agent, num_seeds: int, prid: str=''):
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
        prid: str
            String to identify the trainer's output in terminal
        '''
        self.env = env
        self.num_episodes = num_episodes
        self.agent = agent
        self.num_seeds = num_seeds
        self.prid = prid
        self.total_rewards = []
        self.avg_rewards = []
        self.avg_values = []        
        self.max_reward = -float('inf')

    def verbose_print(self, episode):
        print('{0} episode {1}/{2} - avg reward: {3:.5f} ({4:.5f}), avg value: {5:.5f}'.format(
            self.prid, str(episode).rjust(len(str(self.num_episodes))), 
            self.num_episodes, self.avg_rewards[-1], self.max_reward,
            self.avg_values[-1]
        ))

    def run(self):
        '''Trainer.run: run all episodes'''
        for episode in range(self.num_episodes):
            self.run_epsiode()
            if episode == 0 or (episode+1) % 10 == 0:
                episode_ = episode + 1            
                self.verbose_print(episode_)
            if self.avg_rewards[-1] > self.max_reward:
                self.agent.save_network()
                self.save_metrics()
                self.max_reward = self.avg_rewards[-1]
                if not (episode == 0 or (episode+1) % 10 == 0):
                    self.verbose_print(episode+1)
        return self.summary()
    
    def summary(self):
        '''Trainer.summary: return dictionary of metrics'''
        return {
            'total rewards': self.total_rewards,
            'average rewards': self.avg_rewards,
            'average value': self.avg_values,
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
            action, avg_value = self.agent.select_action(state.to(self.agent.device))
            state_, reward, done, _ = self.env.step(action)
            state_ = torch.from_numpy(state_).to(torch.float32).unsqueeze(0)
            self.agent.store(state, action, reward, done)
            state = state_.clone()
            self.agent.learn()
            episode_reward += reward
            episode_value += avg_value
            n += 1
        self.total_rewards.append(episode_reward)
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
        episode_reward = 0
        episode_value = 0
        n = 0
        while not done:
            action, avg_value = self.agent.select_action(state.to(self.agent.device))
            state_, reward, done, _ = self.env.step(action)
            state_ = torch.from_numpy(state_).to(torch.float32).unsqueeze(0)
            self.agent.store(state, action, reward)
            state = state_.clone()
            episode_reward += reward
            episode_value += avg_value
            n += 1
        self.agent.learn()
        self.total_rewards.append(episode_reward)
        self.avg_rewards.append(episode_reward/n)
        self.avg_values.append(episode_value/n)
        self.agent.empty()
