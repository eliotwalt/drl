import argparse
import json
from drl.off_policy.agents import *
from drl.off_policy.trainers import *
from drl.on_policy.agents import *
from drl.on_policy.trainers import *

algo_map = {
    'dql': {
        'agent': DQLAgent,
        'trainer': QTrainer
    },
    'dqn': {
        'agent': DQNAgent,
        'trainer': QTrainer
    },
    'ddqn': {
        'agent': DDQNAgent,
        'trainer': QTrainer
    },
    # To complete
}

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', help='path to config file', required=True)
    args = parser.parse_args()
    with open(args.cfg) as jsf:
        cfg = json.load(jsf)
    return cfg

def get_agent(algo):
    return algo_map[algo.lower()]['agent']

def get_trainer(algo):
    return algo_map[algo.lower()]['trainer']
    
