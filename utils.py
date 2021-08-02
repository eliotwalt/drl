import os
import numpy as np

def _mkdir(path):
    '''create dir if doesn't exist'''
    if not os.path.isdir(path):
        os.mkdir(path)

def make_nested_dir(root, *names):
    '''creates nested directories'''
    names = list(names)
    path = os.path.join(root, names[0])
    _mkdir(path)
    for name in names[1:]:
        path = os.path.join(path, name)
        _mkdir(path)

def smooth_curve(signal, window=100):
    '''smooth signal and return (x,y) for plotting'''
    if len(signal) <= 2*window:
        window = int(window/4)
    y = np.convolve(np.array(signal), np.ones((window,))/window, mode='valid')
    x = np.linspace(0, len(signal), len(y))
    return x, y

def config_to_kwargs(config, kwargs_list):
    '''select key, values in a config dictionary which match an entry
    of the kwargs list'''
    return {kw: config[kw] for kw in kwargs_list if kw != 'self'}
