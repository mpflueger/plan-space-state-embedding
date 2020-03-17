#!/usr/bin/env python3
"""
Use RL to train a policy for the reacher environment
"""

import argparse
from multiprocessing import Pool
import os

import yaml

from garage.experiment import run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import PPO
from garage.tf.algos import TRPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import CategoricalMLPPolicy
from garage.tf.policies import GaussianMLPPolicy

from reacher_env import ReacherEnv
from reacher_embedded_env import ReacherEmbeddedEnv
from garage.envs.base import GarageEnv


def run_task(snapshot_config, *_,
             env_params,
             algo='TRPO',
             algo_params={},
             epochs=1000,
             batch_size=4000,
             policy_hidden_sizes=(32,32),
             embed_state=False,
             model_dir='../models/reacher_limited/train_6',
             augment_embedded_state=False):
    """Run task."""

    embed_config_file = os.path.join(model_dir, 'config.yaml')
    ckpt_path = os.path.join(model_dir, 'model_latest.ckpt')

    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        if embed_state:
            reacher_env = ReacherEmbeddedEnv(
                embed_config_file,
                ckpt_path,
                augment_embedded_state=augment_embedded_state,
                **env_params)
        else:
            reacher_env = ReacherEnv(**env_params)
        env = GarageEnv(reacher_env)

        policy = GaussianMLPPolicy(
            name='policy',
            env_spec=env.spec,
            hidden_sizes=policy_hidden_sizes)
            # hidden_sizes=(32, 32))

        #************** TRPO ***************
        if algo == 'TRPO':
            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TRPO(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        **algo_params)
                        # max_path_length=100,
                        # discount=0.99,
                        # max_kl_step=0.01)

        #**************** PPO *********************
        elif algo == 'PPO':
            baseline = GaussianMLPBaseline(
                env_spec=env.spec,
                regressor_args=dict(
                    hidden_sizes=(32, 32),
                    use_trust_region=True,
                ),
            )

            # NOTE: make sure when setting entropy_method to 'max', set
            # center_adv to False and turn off policy gradient. See
            # tf.algos.NPO for detailed documentation.
            algo = PPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                # max_path_length=100,
                # discount=0.99,
                gae_lambda=0.95,
                lr_clip_range=0.2,
                optimizer_args=dict(
                    batch_size=32,
                    max_epochs=10,),
                stop_entropy_gradient=True,
                entropy_method='max',
                policy_ent_coeff=0.02,
                center_adv=False,
                **algo_params)

        #**************** Other? **********************
        else:
            print("ERROR: requested unrecognized algorithm: ", algo)
            raise NotImplementedError

        runner.setup(algo, env)
        runner.train(n_epochs=epochs, batch_size=batch_size)


def find_empty_dir(prefix):
    i = 1
    while os.path.exists(prefix + str(i)):
        i += 1
    return prefix + str(i)


def write_config(fn, data):
    with open(fn, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    return fn


def pool_runner(config, group_dir, seed):
    print("***************************************************")
    exp_dir = os.path.join(group_dir, 'exp_' + str(seed))
    if os.path.exists(exp_dir):
        print("Skipping experiment dir: ", exp_dir)
        # TODO: resume if necessary?
        return

    print("Starting experiment in dir: ", exp_dir)
    run_experiment(
        #run_task,
        lambda sc, *args: run_task(
            sc, *args,
            env_params=config['env_params'],
            algo=config['algo'],
            algo_params=config['algo_params'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            policy_hidden_sizes=config['policy_hidden_sizes'],
            embed_state=config['use_state_embedding'],
            model_dir=config['model_dir'],
            augment_embedded_state=config['augment_embedded_state']),
        exp_prefix='trpo_reacher_embedded',
        log_dir=exp_dir,
        snapshot_mode='last',
        seed=seed,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an agent")
    parser.add_argument('--logdir', metavar='DIR', default=None, required=True,
        help="Location for logging results")
    parser.add_argument('--config', metavar='CONFIG', default=None,
        help="Use a config file")
    parser.add_argument('--seed', default=None,
        help="Set the random seed")
    parser.add_argument('--embed', default=None, action='store_true',
        help="Use the state embedding")
    parser.add_argument('--model', metavar='DIR',
        help="Specify a directory of the embedding model")
    parser.add_argument('--augment', default=None, action='store_true',
        help="Augment embedded state with raw state")
    parser.add_argument('--runs', metavar='N', type=int, default=1,
        help="How many times to run training, default 1")
    parser.add_argument('--threads', metavar='N', type=int, default=1,
        help="How many experiments to run at a time, default 1")
    return parser.parse_args()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    args = parse_args()

    config = {}
    config['type'] = "trpo_reacher_embedded"
    config['seed'] = 1
    config['use_state_embedding'] = False
    config['augment_embedded_state'] = False
    config['model_dir'] = '../models/reacher_limited/train_6'
    config['env_params'] = {
        'obs_theta': True,
        'obs_trig': False,
        'obs_target': True}
    config['algo'] = 'TRPO'
    config['algo_params'] = {
        'max_path_length': 100,
        'discount': 0.99,
        'max_kl_step': 0.01}
    config['policy_hidden_sizes'] = (32, 32)
    config['epochs'] = 2000
    config['batch_size'] = 4000

    # Load config file overriding defaults
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    # Override config file if options are specified
    if args.seed:
        config['seed'] = args.seed
    if args.embed:
        config['use_state_embedding'] = args.embed
    if args.model:
        config['model_dir'] = args.model
    if args.augment:
        config['augment_embedded_state'] = args.augment

    # Find group name prefix
    if config['algo'] == 'TRPO':
        prefix = 'trpo_reacher_'
    elif config['algo'] == 'PPO':
        prefix = 'ppo_reacher_'
    else:
        prefix = 'reacher_'
    if config['use_state_embedding']:
        prefix += 'embedded_'
    if config['augment_embedded_state']:
        prefix += 'augment_'
    prefix += 'group_'

    # Don't accidentally overwrite results, also allow resume
    if os.path.exists(os.path.join(args.logdir, 'config.yaml')):
        res = input("Log dir \"{}\" already contains a config.yaml, resume?" +
            " [y,N]:".format(args.logdir))
        if res.lower() == y:
            with open(os.path.join(args.logdir, 'config.yaml'), 'r') as f:
                config = yaml.safe_load(f)
            group_dir = args.logdir
        else:
            exit()
    else:
        # Find our group dir
        group_dir = find_empty_dir(os.path.join(args.logdir, prefix))
        os.makedirs(group_dir)
        print("Created new group dir at \"{}\"".format(group_dir))

        # Write config into group dir
        path = write_config(os.path.join(group_dir, 'config.yaml'), config)
        print("Wrote new config file at \"{}\"".format(path))

    # Run the Group
    pool = Pool(args.threads)
    pool.starmap(
        pool_runner,
        [[config, group_dir, config['seed'] + i] for i in range(args.runs)])
    pool.close()
    pool.join()
    print("All workers are done.")

