# Deep Reinforcement Learning Overview

This repository contains implementations of different classic Deep Reinforcement Learning algorithms using PyTorch and OpenAI Gym environements. Currently, only vector states environments are supported such as `LunarLander-v2`.

## Structure
The repository is structured as follows:
- `executor.py`: Training execution script
- `utils.py`: Helper functions for execution
- `config.py`: Helper functions for configuration
- `defaults/configs/`: Default JSON configuration files for each algorithms
- `drl/`: Main package's directory
    - `models.py`: PyTorch models
    - `trainers.py`: Training execution classes
    - `off_policy/`: Off-policy algorithms
        - `agents.py`: Off-policy agents
        - `replay_buffer.py`: Replay buffer implementation
    - `on_policy/`: On-policy algorithms
        - `agents.py`: On-policy agents
        - `envs.py`: Multiprocessing environments

## Algorithms

Currently implemented algorithms are:
- Deep Q-Learning: `drl.off_policy.agents.DQLAgent` \[<a href='https://arxiv.org/pdf/1312.5602.pdf'>paper</a>\]
- Deep Q-Network: `drl.off_policy.agents.DQNAgent` \[<a href='https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf'>paper</a>\]
- Double Deep Q-Netwok: `drl.off_policy.agents.DDQNAgent` \[<a href='https://arxiv.org/pdf/1509.06461.pdf'>paper</a>\]
- Reinforce: `drl.on_policy.agents.ReinforceAgent` \[<a href='https://link.springer.com/content/pdf/10.1007%2FBF00992696.pdf'>paper</a>\]
- Actor Critic: `drl.on_policy.agents.ActorCriticAgent` \[<a href='http://www.incompleteideas.net/book/RLbook2020.pdf'>book</a>\]
- Advantage Actor Critic (A2C) with parallel environements: `drl.on_policy.agents.A2CAgent` \[<a href='https://arxiv.org/pdf/1602.01783v2.pdf'>paper</a>\]

## Usage

**todo**: requirements
The execution is handle by `executor.py`. 
```bash
python executor.py -h
usage: executor.py [-h] -c CFG

optional arguments:
  -h, --help         show this help message and exit
  -c CFG, --cfg CFG  path to config file
```
Agent, trainer and envrionements are created based on a JSON configuration file passed as an argument. For reference, see `defaults/configs/`. For instance, training an A2C agent using `defaults/configs/lunarlander-v2.json` is achieved by:
```bash
python exectuor.py -c defaults/configs/lunarlander-v2.json
```

## Todo

- Fix Dueling architectures
- Continuous action space algorithms: ddpg, ppo, sac, ...
- Streamline API

