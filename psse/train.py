__author__ = "Max Pflueger"

import argparse
from multiprocessing import Pool
import numpy as np
import os

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import yaml

import traj_metrics
import data_loader
from state_net import StateNet, StateNetHypers


def print_tf_info():
    print("Using TensorFlow version ", tf.__version__)
    print(" CUDA support: ", tf.test.is_built_with_cuda())


def load_data(data_dir, state_select, steps=[1]):
    """
    Load plan data from data_dir

    state_select: list of indicies of state dimensions to use (for excluding
                  target dimensions or other undesired axes)
    steps: list of step deltas for making state triplets
    """
    data_sets = []
    for step in steps:
        data_sets += data_loader.read_dir(data_dir, step=step)
    print("Read datasets with shape ", np.array(data_sets).shape)
    data_sets = np.array(data_sets)[:, :, state_select]
    return data_sets


def run_training(config, logdir, restore):
    """
    Perform a training run

    config: config parameters for this training run
    logdir: directory path for storing training results
    restore: None or checkpoint to restore from
    """
    # Set up file and logging paths
    writer_path = logdir
    config_fn = None
    config['data_dir'] = os.path.expanduser(config['data_dir'])
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    config_fn = os.path.join(logdir, "config.yaml")
    model_prefix = os.path.join(logdir, "model_latest.ckpt")
    if os.path.exists(config_fn):
        print("Config file already exists at {}, aborting".format(config_fn))
        return

    hypers = StateNetHypers(hypers=config['state_embedding_hypers'])
    batch_size = int(config['batch_size'])

    if config['seed']:
        tf.compat.v1.set_random_seed(config['seed'])
        np.random.seed(config['seed'] + 1)

    # Create model and initialize variables
    state_net = StateNet(config['state_shape'], hypers)

    # save current config
    config['state_embedding_hypers'] = hypers.export()
    with open(config_fn, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    data = load_data(
        config['data_dir'], config['state_select'], config['steps'])
    print("Loaded dataset with shape {}".format(data.shape))

    # Load trajectory data for evaluation
    traj_data = traj_metrics.read_traj_dir(
        config['data_dir'], config['state_select'], config['time_select'])

    with tf.compat.v1.Session(graph=state_net.graph) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # Debug
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        if restore:
            state_net.restore_checkpoint(sess, restore)

        writer = None
        if writer_path:
            writer = tf.summary.FileWriter(
                writer_path, graph=sess.graph, flush_secs=5)

        for epoch in range(config['epochs']):
            print("Starting epoch {}".format(epoch))

            # Sample mini-batch
            # Epoch: shuffle data, pull mini-batches in order
            np.random.shuffle(data)
            for i in range(data.shape[0] // batch_size):
                batch = data[i*batch_size : i*batch_size + batch_size]

                step = state_net.train_step(sess, batch, writer)

                if step % 30 == 0:
                    state_net.eval_step(sess, traj_data, writer)
                if step % 100 == 0:
                    print("  Global Step: ", step)

            # Save model (for import or restore)
            state_net.save_checkpoint(sess, model_prefix)


def pool_runner(config, group_dir, restore, i):
    config['seed'] += i
    exp_dir = os.path.join(group_dir, 'train_' + str(i))
    if os.path.exists(exp_dir):
        print("Skipping training dir: ", exp_dir)
        return

    print("Starting training in dir: ", exp_dir)
    run_training(config, exp_dir, restore)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a state embedding from trajectory data.")
    parser.add_argument('conf', metavar='CONFIG_FILE',
        help="Config file for training")
    parser.add_argument('--restore', '-r', metavar='CKPT',
        help="Restore from checkpoint prefix")
    parser.add_argument('--logdir', metavar='DIR',
        help="Directory to log training results")
    parser.add_argument('--seed', metavar='N', type=int, default=None, 
        help="Seed for random numbers")
    parser.add_argument('--runs', metavar='N', type=int, default=1,
        help="How many training runs to do (default 1)")
    parser.add_argument('--threads', metavar='N', type=int, default=1,
        help="How many processes to start (default 1)")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    args = parse_args()

    print_tf_info()

    # Load config file
    with open(args.conf, 'r') as f:
        config = yaml.safe_load(f)

    # Override config params with command line flags
    if args.seed:
        config['seed'] = args.seed
    if args.logdir:
        config['log_dir'] = args.logdir
    else:
        print("Using log_dir from config: ", config['log_dir'])
    if not os.path.exists(config['log_dir']):
        os.mkdir(config['log_dir'])

    pool = Pool(args.threads)
    pool.starmap(
        pool_runner,
        [[config, config['log_dir'], args.restore, i] for i in range(args.runs)])
    pool.close()
    pool.join()
    print("All workers are done.")

