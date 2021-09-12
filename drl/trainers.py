import torch
import gym
import json
import os

# To automate (for lunarlander-v2)
num_consecutive_episodes_solved = 100
reward_threshold = 200

class Trainer:
    def __init__(self, env: gym.Env, num_episodes: int, agent, num_seeds: int, prid: str=''):
        '''Trainer constructor
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
        self.episode_rewards = []
        self.episode_values = []
        self.avg_rewards = []
        self.avg_values = []        
        self.max_rewards = [-float('inf')]
        self.solved = 'not solved'
        self.solved_at = None

    def verbose_print(self, episode: int):
        print('{0} episode {1}/{2} ({3}) - ep reward: {4:.5f}, avg reward: {5:.5f}, max reward: {6:.5f}'.format(
            self.prid, str(episode).rjust(len(str(self.num_episodes))),
            self.num_episodes, self.solved, self.episode_rewards[-1], 
            self.avg_rewards[-1], self.max_rewards[-1],
        ))

    def run(self):
        '''Trainer.run: run all episodes'''
        for episode in range(self.num_episodes):
            self.run_epsiode()
            if self.solved == 'not solved' and episode >= num_consecutive_episodes_solved and self.avg_rewards[-1] > reward_threshold:
                self.solved = 'solved at {}'.format(episode+1)
                self.solved_at = episode+1
            if episode == 0 or (episode+1) % 10 == 0:
                episode_ = episode + 1            
                self.verbose_print(episode_)
            if self.episode_rewards[-1] > self.max_rewards[-1]:
                self.agent.save_network()
                self.save_metrics()
                self.max_rewards.append(self.episode_rewards[-1])
                if not (episode == 0 or (episode+1) % 10 == 0):
                    self.verbose_print(episode+1)
            else:
                self.max_rewards.append(self.max_rewards[-1])
        self.max_rewards.pop(0)
        return self.summary()
    
    def summary(self):
        '''Trainer.summary: return dictionary of metrics'''
        return {
            'episode reward': self.episode_rewards,
            'episode value': self.episode_values,
            'average reward': self.avg_rewards,
            'average value': self.avg_values,
            'max reward': self.max_rewards,
            'solved at': self.solved_at
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
            self.agent.store(state, state_, action, reward, done)
            state = state_.clone()
            self.agent.learn()
            episode_reward += reward
            episode_value += avg_value
            n += 1
        self.episode_rewards.append(episode_reward)
        self.episode_values.append(episode_value)
        if len(self.episode_rewards) < num_consecutive_episodes_solved:
            self.avg_rewards.append(torch.mean(torch.Tensor(self.episode_rewards)).item())
        else:
            self.avg_rewards.append(torch.mean(torch.Tensor(self.episode_rewards[-num_consecutive_episodes_solved:])).item())
        self.avg_values.append(torch.mean(torch.Tensor(self.episode_values)).item())

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
            self.agent.store(state, state_, action, reward)
            state = state_.clone()
            episode_reward += reward
            episode_value += avg_value
            n += 1
        self.agent.learn()
        self.episode_rewards.append(episode_reward)
        self.episode_values.append(episode_value)
        if len(self.episode_rewards) < num_consecutive_episodes_solved:
            self.avg_rewards.append(torch.mean(torch.Tensor(self.episode_rewards)).item())
        else:
            self.avg_rewards.append(torch.mean(torch.Tensor(self.episode_rewards[-num_consecutive_episodes_solved:])).item())
        self.avg_values.append(torch.mean(torch.Tensor(self.episode_values)).item())
        self.agent.empty()

class MultiEnvsTrainer:
    def __init__(self, env, num_iters: int, agent, num_seeds: int, prid: str=''):
        '''MultiTrainer constructor
        Inputs:
        -------
        env: utils.ParallelEnv
            Gym environment
        num_iters: int
            Number of iterations per epsiode
        agent: QAgent
            QAgent instance
        num_seeds: int
            Number of different seeds to use
        prid: str
            String to identify the trainer's output in terminal
        '''
        self.env = env
        self.num_iters = num_iters
        self.agent = agent
        self.num_seeds = num_seeds
        self.prid = prid
        self.all_rewards = [[] for _ in range(len(env))]
        self.all_values = [[] for _ in range(len(env))]
        self.episode_rewards = [[] for _ in range(len(env))]
        self.episode_values = [[] for _ in range(len(env))]
        self.max_rewards = [[-float('inf')] for _ in range(len(env))]
        self.avg_rewards =  [[] for _ in range(len(env))]
        self.avg_values =  [[] for _ in range(len(env))]
        self.episode_starts = [0 for _ in range(len(env))]
        self.local_solved = [False for _ in range(len(env))]
        self.solved = 'not solved'
        self.solved_at = None

    def verbose_print(self, n_iter: int):
        max_num_episodes = max([len(episode_rewards) for episode_rewards in self.episode_rewards])
        ep_rewards = [episode_rewards[-1] for episode_rewards in self.episode_rewards if len(episode_rewards)>0]
        if max_num_episodes > 0:
            print('{0} iter {1}/{2} ({3}) - mean ep reward: {4:.5f}, max reward: {5:.5f}, max num ep: {6}'.format(
                self.prid, str(n_iter).rjust(len(str(self.num_iters))), self.num_iters, 
                self.solved, sum(ep_rewards)/len(ep_rewards), 
                max([max_rewards[-1] for max_rewards in self.max_rewards]),
                max_num_episodes
            ))

    def pad_metrics(self):
        '''MultiEnvs.pad_metrics: pad metrics lists to equal lengths'''
        max_num_episodes = max([len(episode_rewards) for episode_rewards in self.episode_rewards])
        for lss in [self.episode_rewards, self.episode_values, self.avg_rewards, self.avg_values]:
            for ls in lss:
                if len(ls) != max_num_episodes:
                    ls.extend([ls[-1] for _ in range(max_num_episodes-len(ls))])

    def run(self):
        '''MultiEnvsTrainer.run: run all episodes'''
        states = self.env.reset()
        for n_iter in range(self.num_iters):
            states = self.run_iter(states, n_iter)
            if self.local_solved == [True]*len(self.env): # Solved only when each individual env is solved
                self.solved = 'solved at {}'.format(n_iter)
                self.verbose_print(n_iter+1)
            elif n_iter == 0 or (n_iter+1) % 100 == 0:
                self.verbose_print(n_iter+1)
        for max_rewards in self.max_rewards:
            max_rewards.pop(0)
        self.pad_metrics()
        return self.summary()

    def run_iter(self, states, n_iter):
        '''MultiEnvsTrainer.run_iter: runs a single iteration of learning'''
        actions, values, dists = self.agent.select_actions(torch.from_numpy(states).to(torch.float32).to(self.agent.device))
        states_, rewards, dones, _ = self.env.step(actions.cpu().numpy())
        self.agent.store(
            torch.from_numpy(rewards).to(torch.float32), 
            actions, 
            values.to(torch.float32),
            torch.from_numpy(dones).to(torch.long),
            dists
        )
        self.agent.learn()
        for i, (reward, value, done) in enumerate(zip(rewards, values, dones)):
            self.all_rewards[i].append(reward.item())
            self.all_values[i].append(value.item())
            if done:
                self.episode_rewards[i].append(sum(self.all_rewards[i][self.episode_starts[i]:]))
                avg_start = min(0, self.episode_starts[i])
                avg_window = self.episode_rewards[i][avg_start:]
                self.avg_rewards[i].append(sum(avg_window)/len(avg_window))
                if avg_start >= num_consecutive_episodes_solved and self.avg_rewards[-1]:
                    self.local_solved[i] = True
                if self.episode_rewards[i][-1] > self.max_rewards[i][-1]:
                    self.max_rewards[i].append(self.episode_rewards[i][-1])
                self.episode_values[i].append(sum(self.all_values[i][self.episode_starts[i]:]))
                self.avg_values[i].append(sum(self.episode_values[i])/len(self.episode_values[i]))
                self.episode_starts[i] = n_iter
        return states_
    
    def summary(self):
        '''MultiEnvsTrainer.summary: return dictionary of metrics'''
        return {
            'all_rewards': self.all_rewards,
            'all_values': self.all_values,
            'episode reward': self.episode_rewards,
            'episode value': self.episode_values,
            'average reward': self.avg_rewards,
            'average value': self.avg_values,
            'max reward': self.max_rewards,
            'solved at': self.solved_at
        }

    def save_metrics(self):
        resfile = os.path.join(self.agent.path, 'metrics.json')
        with open(resfile, 'w') as f:
            json.dump(self.summary(), f)
