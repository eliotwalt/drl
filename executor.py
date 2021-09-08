import gym
import os
from config import get_config, get_agent, get_trainer, get_env_maker
import json
import matplotlib.pyplot as plt
import numpy as np
from utils import make_nested_dir, smooth_curve, config_to_kwargs

pid = os.getpid()
root = os.path.dirname(__file__)

def main():
    config = get_config()
    config['pid'] = pid
    prid = '[{}-{}({})]'.format(config['algorithm'], config['env'], pid)
    config['prid'] = prid
    make_nested_dir(root, *(config['dir'], config['name']))

    print('{} starting {} environement.'.format(prid, config['env']))
    env, env_kwargs = get_env_maker(config['algorithm'])
    env = env(**config_to_kwargs(config, env_kwargs))
    config['env'] = env
    config['num_actions'] = env.action_space.n
    config['input_dim'] = env.observation_space.shape[0]

    print('{} creating {} agent and trainer.'.format(prid, config['algorithm']))
    agent, agent_kwargs = get_agent(config['algorithm'])
    trainer, trainer_kwargs = get_trainer(config['algorithm'])
    agent = agent(**config_to_kwargs(config, agent_kwargs))
    config['agent'] = agent
    trainer = trainer(**config_to_kwargs(config, trainer_kwargs))

    print('{} starting training.'.format(prid))
    summary = trainer.run()
    env.close()

    print('{} writing results and model.'.format(prid))
    trainer.save_metrics()

    print('{} saving figure.'.format(prid))
    for metric_name, metric_vals in summary.items():
        if metric_name != 'solved at':
            figfile = os.path.join(agent.path, '{}.png'.format(metric_name))
            plt.plot(np.arange(1,len(metric_vals)+1), metric_vals, c='b', alpha=0.3)
            x_smooth, y_smooth = smooth_curve(metric_vals)
            plt.plot(x_smooth, y_smooth, c='b')
            plt.xlabel('episodes')
            plt.ylabel(metric_name)
            plt.savefig(figfile)
            plt.close()

    print('{} done.'.format(prid))

if __name__ == '__main__':
    main()