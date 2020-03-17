import numpy as np
import os
import re
import yaml


def read_traj(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data['path'], data['state_dims']


def get_transitions(data, step=1):
    """
    Get triplet state transitions out of a list of states.
    step: size of steps in the transitions, i.e. for a step of 2 return
          a list of items like [[1,3,5],[2,4,6], ...]
    """
    triplets = []
    for i in range(len(data['path']) - (step * 2)):
        triplets += [data['path'][i:(i+(step*2)+1):step]]
    return triplets


def list_dir(d, patt):
    """
    List the paths of all files in a directory matching a pattern
    """
    return [i.path for i in os.scandir(d)
            if i.is_file and re.search(patt, i.path)]


def read_dir(d, patt='\.json$', step=1):
    """
    Read all data files out of a directory.  
    Default: Expect files with .json file extension.
    Uses yaml loader, which should load json as well.
    """
    transitions = []

    paths = list_dir(d, patt)
    print("Found {} data files in {}".format(len(paths), d))
    # for path in paths:
    for i in range(len(paths)):
        with open(paths[i], 'r') as file:
            data = yaml.safe_load(file)
            transitions += get_transitions(data, step)
        if i % 50 == 49:
            print("  loaded {} files".format(i+1))
            # break # TODO: REMOVE THIS

    print("Read dir to shape ", np.array(transitions).shape)
    return transitions

