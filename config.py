import argparse
import json
from drl.off_policy.agents import *
from drl.on_policy.agents import *
from drl.trainers import *
import inspect

algo_map = {
    'dql': {
        'agent': DQLAgent,
        'trainer': OnlineTrainer
    },
    'dqn': {
        'agent': DQNAgent,
        'trainer': OnlineTrainer
    },
    'ddqn': {
        'agent': DDQNAgent,
        'trainer': OnlineTrainer
    },
    'dueling-dql': {
        'agent': DuelingDQLAgent,
        'trainer': OnlineTrainer,
    },
    'dueling-dqn': {
        'agent': DuelingDQNAgent,
        'trainer': OnlineTrainer,
    },
    'dueling-ddqn': {
        'agent': DuelingDDQNAgent,
        'trainer': OnlineTrainer,
    },
    'reinforce': {
        'agent': ReinforceAgent,
        'trainer': EpisodicTrainer,
    },
    'actor-critic': {
        'agent': ActorCriticAgent,
        'trainer': OnlineTrainer
    },
}

def make_kwargs(class_):
    baseclasses = class_.__mro__
    kwargs = []
    for baseclass in baseclasses:
        kwargs.extend([arg for arg in inspect.getfullargspec(baseclass).args if arg != 'self'])
    return kwargs

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', help='path to config file', required=True)
    args = parser.parse_args()
    with open(args.cfg) as jsf:
        cfg = json.load(jsf)
    return cfg

def get_agent(algo):
    return algo_map[algo.lower()]['agent'], make_kwargs(algo_map[algo.lower()]['agent'])

def get_trainer(algo):
    return algo_map[algo.lower()]['trainer'], make_kwargs(algo_map[algo.lower()]['trainer'])
    
