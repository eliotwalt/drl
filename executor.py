import gym
import os
from config import get_config, get_agent, get_trainer

pid = os.getpid()
prid = '[{}]'.format(pid)

def main():
    config = get_config()
    print('{} starting {} environement.'.format(prid, config['env']))
    env = gym.make(config['env'])
    print('{} creating {} agent and trainer.'.format(prid, config['algorithm']))
    agent_config = {k: v for k, v in config.items() if k not in ['algorithm', 'env', 'num_seeds']}
    agent_config['num_actions'] = env.action_space.n
    agent_config['input_dim'] = env.observation_space.shape[0]
    agent = get_agent(config['algorithm'])(**agent_config)
    trainer = get_trainer(config['algorithm'])(env, config['num_episodes'], agent, 'num_seeds', pid)
    print('{} starting training.'.format(prid))
    trainer.run()

if __name__ == '__main__':
    main()   