"""
Environment to use the reacher with a state embedding
"""
import numpy as np
import tensorflow as tf
import yaml

from gym import utils

from reacher_env import ReacherEnv
from state_net import StateNet, StateNetHypers

class ReacherEmbeddedEnv(ReacherEnv):
    def __init__(self,
                 embed_config_file,
                 ckpt_path,
                 augment_embedded_state=False,
                 **kwargs):
        """
        Input:
        embed_config_file: config for the embedding model
        ckpt_path: embedding network checkpoint
        augment_embedded_state: if true normal reacher observation will be
            concatenated to the embedded state
        """
        self.augment_embedded_state = augment_embedded_state

        # Load the embedding model
        print("ESTABLISH THE STATE NET!")
        with open(embed_config_file, 'r') as f:
            embed_config = yaml.safe_load(f)
        state_net_hypers = StateNetHypers(
            embed_config['state_embedding_hypers'])
        self.state_net = StateNet(embed_config['state_shape'], state_net_hypers)

        self.sess = tf.compat.v1.Session(graph=self.state_net.graph)
        self.state_net.restore_checkpoint(self.sess, ckpt_path)
        print("STATE NET ESTABLISHED!")

        ReacherEnv.__init__(self, **kwargs)
        utils.EzPickle.__init__(
            self,
            embed_config_file,
            ckpt_path,
            augment_embedded_state=augment_embedded_state,
            **kwargs)


    def _get_obs(self):
        # Get state from MuJoCo
        # Grabbing state of the reacher joints, ignoring target state
        # Corresponds with state_select: [0,1,4,5] in the training config
        x = np.concatenate([
            self.sim.data.qpos.flat[:2],
            self.sim.data.qvel.flat[:2]
            ])
        x = np.expand_dims(x, axis=0) # add batch dim

        # Get embedded state
        z_mu, z_simga = self.state_net.encode(self.sess, x)

        # Optionally augment with raw state
        r = np.array(z_mu).ravel()
        if self.augment_embedded_state:
            r = np.concatenate([r, super()._get_obs()])

        return r
