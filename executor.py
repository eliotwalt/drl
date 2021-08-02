import gym
import os
from config import get_config, get_agent, get_trainer
import json
import matplotlib.pyplot as plt
import numpy as np

pid = os.getpid()
prid = '[{}]'.format(pid)
root = os.path.dirname(__file__)

def make_dirs(d, n):
    savedir = os.path.join(root, d)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    savedir = os.path.join(savedir, n)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

def kwargs_from_config(config, kw_list):
    return {kw: config[kw] for kw in kw_list if kw != 'self'}

def smooth(y, window=100):
    y_smooth = np.convolve(np.array(y), np.ones((100,))/100, mode='valid')
    x_smooth = np.linspace(0, len(y), len(y_smooth))
    return x_smooth, y_smooth

def main():
    config = get_config()
    config['pid'] = pid
    make_dirs(config['dir'], config['name'])

    print('{} starting {} environement.'.format(prid, config['env']))
    env = gym.make(config['env'])
    config['env'] = env
    config['num_actions'] = env.action_space.n
    config['input_dim'] = env.observation_space.shape[0]

    print('{} creating {} agent and trainer.'.format(prid, config['algorithm']))
    agent, agent_kwargs = get_agent(config['algorithm'])
    trainer, trainer_kwargs = get_trainer(config['algorithm'])
    agent = agent(**kwargs_from_config(config, agent_kwargs))
    config['agent'] = agent
    trainer = trainer(**kwargs_from_config(config, trainer_kwargs))

    print('{} starting training.'.format(prid))
    summary = trainer.run()

    print('{} writing results and model.'.format(prid))
    trainer.save_metrics()

    print('{} saving figure.'.format(prid))
    for metric_name, metric_vals in summary.items():
        figfile = os.path.join(agent.path, '{}.png'.format(metric_name))
        plt.plot(np.arange(1,len(metric_vals)+1), metric_vals, c='b', alpha=0.3)
        x_smooth, y_smooth = smooth(metric_vals)
        plt.plot(x_smooth, y_smooth, c='b')
        plt.xlabel('episodes')
        plt.ylabel(metric_name)
        plt.savefig(figfile)
        plt.close()

    print('{} done.'.format(prid))

if __name__ == '__main__':
    main()