import argparse
import json
from drl.off_policy.agents import *
from drl.on_policy.agents import *
from drl.trainers import *
import inspect
from typing import Any
import gym
from drl.on_policy.envs import ParallelEnv

def make_(env: str):
    return gym.make(id=env)

def no_make(env: str): return env

algo_map = {
    'dql': {
        'agent': DQLAgent,
        'trainer': OnlineTrainer,
        'env_make': make_
    },
    'dqn': {
        'agent': DQNAgent,
        'trainer': OnlineTrainer,
        'env_make': make_
    },
    'ddqn': {
        'agent': DDQNAgent,
        'trainer': OnlineTrainer,
        'env_make': make_
    },
    'dueling-dql': {
        'agent': DuelingDQLAgent,
        'trainer': OnlineTrainer,
        'env_make': make_
    },
    'dueling-dqn': {
        'agent': DuelingDQNAgent,
        'trainer': OnlineTrainer,
        'env_make': make_
    },
    'dueling-ddqn': {
        'agent': DuelingDDQNAgent,
        'trainer': OnlineTrainer,
        'env_make': make_
    },
    'reinforce': {
        'agent': ReinforceAgent,
        'trainer': EpisodicTrainer,
        'env_make': make_
    },
    'actor-critic': {
        'agent': ActorCriticAgent,
        'trainer': OnlineTrainer,
        'env_make': make_
    },
    'a2c': {
        'agent': A2CAgent,
        'trainer': MultiEnvsTrainer,
        'env_make': ParallelEnv,
    },
    'a3c': {
        'agent': A3CAgent,
        'trainer': SelfContainedTrainer,
        'env_make': make_,
    }
}

def make_kwargs(class_: Any):
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

def get_agent(algo: str):
    return algo_map[algo.lower()]['agent'], make_kwargs(algo_map[algo.lower()]['agent'])

def get_trainer(algo: str):
    return algo_map[algo.lower()]['trainer'], make_kwargs(algo_map[algo.lower()]['trainer'])

def get_env_maker(algo: str):
    constructor = algo_map[algo.lower()]['env_make']
    if constructor == make_:
        return constructor, ['env']
    else:
        return constructor, make_kwargs(constructor)
    
