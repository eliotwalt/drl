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
        agent: Many
            Agent instance
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
            Parallel gym environment
        num_iters: int
            Number of iterations per env
        agent: Many
            Agent instance
        num_seeds: int
            Number of different seeds to use
        prid: str
            String to identify the trainer's output in terminal
        '''
        self.env = env
        self.test_env = gym.make(env.env_name)
        self.num_iters = num_iters
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

    def verbose_print(self, n_iter: int):
        print('{0} iter {1}/{2} ({3}) - ep reward: {4:.5f}, avg reward: {5:.5f}, max reward: {6:.5f}'.format(
            self.prid, str(n_iter).rjust(len(str(self.num_iters))),
            self.num_iters, self.solved, self.episode_rewards[-1], 
            self.avg_rewards[-1], self.max_rewards[-1],
        ))

    def test(self):
        '''MultiEnvsTrainer.run: run a test episode'''
        state = self.test_env.reset()
        state = torch.from_numpy(state).to(torch.float32).unsqueeze(0)
        done = False
        episode_reward = 0
        episode_value = 0
        n = 0
        while not done:
            action, avg_value, _ = self.agent.select_actions(state.to(self.agent.device))
            state, reward, done, _ = self.test_env.step(action.item())
            state = torch.from_numpy(state).to(torch.float32).unsqueeze(0)
            episode_reward += reward
            episode_value += avg_value.item()
            n += 1
        self.episode_rewards.append(episode_reward)
        self.episode_values.append(episode_value)
        if len(self.episode_rewards) < num_consecutive_episodes_solved:
            self.avg_rewards.append(torch.mean(torch.Tensor(self.episode_rewards)).item())
        else:
            self.avg_rewards.append(torch.mean(torch.Tensor(self.episode_rewards[-num_consecutive_episodes_solved:])).item())
        self.avg_values.append(torch.mean(torch.Tensor(self.episode_values)).item())

    def run(self):
        '''MultiEnvsTrainer.run: run all iterations'''
        states = self.env.reset()
        for n_iter in range(self.num_iters):
            states = self.run_iter(states, n_iter)
            if self.solved == 'not solved' and n_iter >= num_consecutive_episodes_solved and self.avg_rewards[-1] > reward_threshold:
                self.solved = 'solved at {}'.format(n_iter+1)
                self.solved_at = n_iter+1
            if n_iter == 0 or (n_iter+1) % 1000 == 0:
                if len(self.episode_rewards) == 0:
                    self.test()
                n_iter_ = n_iter + 1            
                self.verbose_print(n_iter_)
            if self.episode_rewards[-1] > self.max_rewards[-1]:
                self.agent.save_network()
                self.save_metrics()
                self.max_rewards.append(self.episode_rewards[-1])
                if not (n_iter == 0 or (n_iter+1) % 1000 == 0):
                    self.verbose_print(n_iter+1)
            else:
                self.max_rewards.append(self.max_rewards[-1])
        self.max_rewards.pop(0)
        return self.summary()

    def run_iter(self, states, n_iter):
        '''MultiEnvsTrainer.run_iter: runs a single iteration of learning'''
        actions, values, dists = self.agent.select_actions(torch.from_numpy(states).to(torch.float32).to(self.agent.device))
        states_, rewards, dones, _ = self.env.step(actions.cpu().numpy())
        _, next_values = self.agent.network(torch.from_numpy(states_).to(torch.float32).to(self.agent.device))
        self.agent.store(
            torch.from_numpy(rewards).to(torch.float32).to(self.agent.device), 
            actions.to(self.agent.device), 
            values.to(torch.float32).to(self.agent.device),
            next_values.to(torch.float32).to(self.agent.device),
            torch.from_numpy(dones).to(torch.long).to(self.agent.device),
            dists
        )
        if n_iter > 1 and n_iter % self.agent.max_iters == 0:
            self.agent.learn()
            self.test()
        return states_
    
    def summary(self):
        '''MultiEnvsTrainer.summary: return dictionary of metrics'''
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

class SelfContainedTrainer:
    def __init__(self, env, num_episodes: int, agent, num_workers: int, num_seeds: int, prid: str=''):
        '''SelfContainedTrainer constructor
        Inputs:
        -------
        env: str
            Gym environment id
        num_episodes: int
            Number of episodes per agent
        agent: Many
            Agent instance
        num_workers: int
            Number of workers
        num_seeds: int
            Number of different seeds to use
        prid: str
            String to identify the trainer's output in terminal
        '''
        env_name = str(env.spec).split('(')[1].split(')')[0]
        self.envs = [gym.make(env_name) for _ in range(num_workers)]
        self.test_env = env
        self.num_episodes = num_episodes
        self.agent = agent
        self.num_workers = num_workers
        self.num_seeds = num_seeds
        self.prid = prid
        self.episode_rewards = []
        self.episode_values = []
        self.avg_rewards = []
        self.avg_values = []        
        self.max_rewards = [-float('inf')]
        self.solved = 'not solved'
        self.solved_at = None

    def test(self):
        '''SelfContainedTrainer.run: run a test episode'''
        state = self.test_env.reset()
        state = torch.from_numpy(state).to(torch.float32).unsqueeze(0)
        done = False
        episode_reward = 0
        episode_value = 0
        n = 0
        while not done:
            action, avg_value, _ = self.agent.select_action(state.to(self.agent.device))
            state, reward, done, _ = self.test_env.step(action.item())
            state = torch.from_numpy(state).to(torch.float32).unsqueeze(0)
            episode_reward += reward
            episode_value += avg_value.item()
            n += 1
        self.episode_rewards.append(episode_reward)
        self.episode_values.append(episode_value)
        if len(self.episode_rewards) < num_consecutive_episodes_solved:
            self.avg_rewards.append(torch.mean(torch.Tensor(self.episode_rewards)).item())
        else:
            self.avg_rewards.append(torch.mean(torch.Tensor(self.episode_rewards[-num_consecutive_episodes_solved:])).item())
        self.avg_values.append(torch.mean(torch.Tensor(self.episode_values)).item())

    def verbose_print(self, episode: int):
        print('{0} episode {1}/{2} ({3}) - ep reward: {4:.5f}, avg reward: {5:.5f}, max reward: {6:.5f}'.format(
            self.prid, str(episode).rjust(len(str(self.num_episodes))),
            self.num_episodes, self.solved, self.episode_rewards[-1], 
            self.avg_rewards[-1], self.max_rewards[-1],
        ))

    def run(self):
        '''SelfContainedTrainer.run: run all episodes'''
        for episode in range(self.num_episodes):
            self.agent.async_learn(self.envs)
            self.test()
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
        '''SelfContainedTrainer.summary: return dictionary of metrics'''
        return {
            'episode reward': self.episode_rewards,
            'episode value': self.episode_values,
            'average reward': self.avg_rewards,
            'average value': self.avg_values,
            'max reward': self.max_rewards,
            'solved at': self.solved_at
        }
