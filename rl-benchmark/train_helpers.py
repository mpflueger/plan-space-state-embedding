import os

import yaml

from garage.experiment import run_experiment

def find_empty_dir(prefix):
    i = 1
    while os.path.exists(prefix + str(i)):
        i += 1
    return prefix + str(i)


def write_config(fn, data):
    with open(fn, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    return fn


"""
This doesn't seem to work since python 3 can't pickle lambda functions,
the dill package might fix it in python 2, didn't find a way in python 3
"""
#def pool_runner(run_task, run_kwargs, group_dir, i):
def pool_runner(run_task, group_dir, i):
    print("***************************************************")
    exp_dir = os.path.join(group_dir, 'exp_') + str(i)
    if os.path.exists(exp_dir):
        print("Skipping experiment dir: ", exp_dir)
        # TODO: resume if necessary?
        return

    print("Starting experiment in dir: ", exp_dir)
    run_experiment(
        run_task,
        log_dir=exp_dir,
        snapshot_mode='last',
        seed=i,
    )
