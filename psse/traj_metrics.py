import numpy as np

import tensorflow as tf

from data_loader import list_dir, read_traj
from state_net import StateNet, StateNetHypers


def read_traj_dir(d, state_indexes, time_index):
    """
    Read a trajectory data directory into memory

    Input:
        d: directory path
        state_indexes: list of indexes of state values
        time_index: index of the time value

    Output:
        data: [ N x [traj(np.array [steps, state]), time(np.array [steps])] ]
    """
    print("Reading trajectory data from \'{}\'".format(d))
    file_paths = list_dir(d, '\.json$')
    data = []
    # for file_path in file_paths:
    for i in range(len(file_paths)):
        traj, _ = read_traj(file_paths[i])
        traj_arr = np.array(traj)

        traj_state_only = traj_arr[:, state_indexes]
        traj_time = traj_arr[:, time_index]

        data += [[traj_state_only, traj_time]]

        if i % 10 == 0:
            print(".", end='', flush=True)
    print("Done")
    return data


def measure_path_len(traj):
    """
    Measure the distance integrated along a trajectory

    traj: numpy array shape: (steps, state_dim)
    """
    dist = 0
    for i in range(0, traj.shape[0]-1):
        dist += np.linalg.norm(traj[i] - traj[i+1])
    return dist


def measure_traj(traj):
    """
    Measure the start-end distance of a trajecoty and the distance
    along the path

    traj: [state] (numpy array)
    """
    dist = np.linalg.norm(traj[0] - traj[-1])
    path_len = measure_path_len(traj)
    return dist, path_len


def measure_traj_list(data):
    """
    Measure a list of trajectories.

    data: [[traj, timestamps]]
    """
    print("Measuring trajectory stats")
    stats = []
    for t in data:
        dist, path_len = measure_traj(t[0])
        time = t[1][-1] - t[1][0]
        stats += [[dist, path_len, time]]
    return np.array(stats)


def embed_traj(traj, state_net, sess):
    """
    Embed a full trajectory into the embedding space.

    traj: [state]
    state_net: StateNet object
    sess: tensorflow Session object
    """
    embedded_traj = []
    for state in traj:
        # expand_dims to add the batch dimension
        z_mu, z_sigma = state_net.encode(
            sess,
            np.expand_dims(state, axis=0))
        embedded_traj += [np.array(z_mu).ravel()]
    embedded_traj = np.array(embedded_traj)
    return embedded_traj


def embed_traj_list(data, state_net, sess, endpoints=False):
    """
    Embed a list of trajectories into the embedding space.  Optionally only
      embed the start and end.

    data: [[traj, timestamps]]
    state_net: StateNet object
    sess: tensorflow Session
    endpoints: (bool) discard everything except start and end?
    """
    print("Embedding trajectory list")

    if endpoints:
        # Cut all the intermediate states out of data to save time
        print("  (endpoints only)")
        data = [[t[0][[0,-1]], t[1][[0,-1]]] for t in data]

    embedded = []
    for traj in data:
        embedded += [[ embed_traj(traj[0], state_net, sess), traj[1] ]]
    return embedded


def find_scale_factor(metric_x, metric_y):
    """
    Calculate the optimal scale factor to normalize metric_x to metric_y.
    If they measure the same metric, then scale*metric_x = metric_y

    There are multiple ways to do this optimization, the method minimizes
    loss = sum_i((y_i - scale * x_i)^2)
    where scale = sum(x_i * y_i) / sum(x_i^2)
    """
    xy_sum = 0
    x2_sum = 0

    assert(len(metric_x) == len(metric_y))

    for i in range(len(metric_x)):
        xy_sum += metric_x[i] * metric_y[i]
        x2_sum += metric_x[i] * metric_x[i]

    scale = xy_sum / x2_sum
    return scale


def find_metric_error(x, y):
    """
    Compare two distance metrics on a space and calculate errors and
      error statistics.

    Input:
    x,y: lists of corresponding distances

    Output:
    scale: the best scale factor to match x to y
    error: list of errors when matching x to y
    mean_error: average of the absolute value of errors
    std_error: standard deviation of the absolute value of errors
    """
    scale = find_scale_factor(x,y)
    error = scale * x - y
    mean_error = np.mean(np.abs(error))
    std_error = np.std(np.abs(error))
    return (scale, error, mean_error, std_error)

