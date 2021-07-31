import os
import json
import itertools
import queue
import _thread 
import subprocess
from config import get_config

def pwait(popen, desc, qu):
    try:
        popen.wait()
    finally:
        qu.put((desc, popen.returncode))

def run(tasks):
    results = queue.Queue()
    pool_size = 5 # 2*os.cpu_count()
    num_pools = len(tasks) // pool_size + 1
    count = 0
    pids = []
    for i in range(0, num_pools, pool_size):
        for task in tasks[i:i+pool_size]:
            p = subprocess.Popen(['python', os.path.join(os.path.dirname(__file__), 'executor.py'), '-c', task])
            _thread.start_new_thread(pwait, (p, "2 finished", results))
            count += 1
            pids.append(p.pid)
    while count > 0:
        desc, rc = results.get()
        count -= 1

def main():
    tempdir = os.path.join(os.path.dirname(__file__), 'tmp')
    os.makedirs(tempdir, exist_ok=True)
    configs = get_config()
    tasks = []
    for config in configs:
        keys, values = zip(*config['params'].items())
        flat_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for tmp_config in flat_configs:
            tmp_config['algorithm'] = config['algorithm']
            name = [tmp_config['algorithm'], tmp_config['env']]
            name += [
                f'{k}{v}' for (k,v) in tmp_config.items() 
                if k not in ['algorithm', 'env']
            ]
            name = '-'.join(name)
            tmp_dir = os.path.join(tempdir, name)
            tmp_config['dir'] = tmp_dir
            tmp_config['name'] = tmp_dir
            os.makedirs(tmp_dir, exist_ok=True)
            path = os.path.join(tmp_dir, name+'.json')
            with open(path, mode='w') as jsf:
               json.dump(tmp_config, jsf)
            tasks.append(path)
    run(tasks)    

if __name__ == '__main__': 
    main()
