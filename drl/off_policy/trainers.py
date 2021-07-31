import torch
import gym

class QTrainer:
    def __init__(self, env: gym.Env, num_episodes: int, agent: QAgent, pid: int=0):
        '''QLearningTrainer constructor
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
        '''
        self.env = env
        self.num_episodes = num_episodes
        self.agent = agent
        self.num_seeds = num_seeds
        self.pid = pid
        self.prid = '[{}]'.format(pid)
        self.avg_rewards = []
        self.avg_qs = []
        #### TODO: SEEDS

    def run_epsiode(self):
        '''QLearningTrainer.run_epsiode: run one episode'''
        state = self.env.reset()
        state = torch.from_numpy(state).to(torch.float32).unsqueeze(0)
        done = False
        while not done:
            action = self.agent.select_action(state.to(self.agent.device))
            state_, reward, done, _ = self.env.step(action)
            state_ = torch.from_numpy(state_).to(torch.float32).unsqueeze(0)
            self.agent.store(state, action, reward, done)
            avg_reward, avg_q = self.agent.learn()
            state = state_.clone()
            self.avg_rewards.append(avg_reward)
            self.avg_qs.append(avg_q)

    def run(self):
        '''QLearningTrainer.run: run all episodes'''
        for episode in range(self.num_episodes):
            self.run_epsiode()
            if (episode-1) % 10 == 0:
                print('{} episode {}/{} - average reward: {} average Q: {}'.format(
                    self.prid, episode, self.num_episodes, 
                    self.avg_rewards[-1], self.avg_qs[-1]
                ))
        print(self.summary())

    def summary(self):
        '''QLearningTrainer.summary: return dictionary of metrics'''
        return {
            'avg_rewards': avg_rewards,
            'avg_value': avg_qs
        }