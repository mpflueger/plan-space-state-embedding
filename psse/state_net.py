__author__ = "Max Pflueger"

import numpy as np

import tensorflow as tf


def elu_plus(x):
    return tf.nn.elu(x) + 1


class Varnet_2(tf.compat.v1.keras.Model):
    def __init__(self, out_dim, **kwargs):
        super().__init__(**kwargs)

        if type(out_dim) == tuple:
            if len(out_dim) > 1:
                print("Varnet_2 can't handle multi-dim state output")
                raise RuntimeError()
            out_dim = out_dim[0]

        self.fc1 = tf.compat.v1.keras.layers.Dense(
            64, activation=tf.nn.relu, name='fc1')
        self.fc2 = tf.compat.v1.keras.layers.Dense(
            64, activation=tf.nn.relu, name='fc2')
        self.mu = tf.compat.v1.keras.layers.Dense(
            out_dim, activation=None, name='mu')
        self.sigma = tf.compat.v1.keras.layers.Dense(
            out_dim, activation=elu_plus, name='sigma')

    def call(self, x):
        hidden = self.fc2(self.fc1(x))
        mu = self.mu(hidden)
        sigma = self.sigma(hidden)
        return (mu, sigma)


def KL(x_mu, x_sig, y_mu, y_sig):
    """
    Calculate the KL divergence between two normal distributions with 
    diagonal covariance.
    Tensor ops, assuming shape=(batch, dim).
    """
    with tf.compat.v1.variable_scope('KL'):
        a = tf.square(x_sig) / tf.square(y_sig)
        b = tf.square(y_mu - x_mu) / tf.square(y_sig)
        c = 2 * tf.math.log(y_sig) - 2 * tf.math.log(x_sig)
        return tf.multiply(
            tf.reduce_sum(a + b + c - tf.ones(tf.shape(x_mu)), axis=1),
            0.5,
            name="KL_Divergence")

def logPDF(x, mu, sigma):
    """
    Calculate the log density of a datapoint in a multivariate normal
    distribution with diagonal covariance.
    """
    with tf.compat.v1.variable_scope('logPDF'):
        logZ = tf.math.log(tf.reduce_prod(2 * 3.1415 * sigma**2, axis=1)) * 0.5
        # logZ = 0.5 * tf.math.log(
        #     (2 * 3.1415)**tf.cast(tf.shape(sigma)[1], tf.float32)
        #     * tf.reduce_prod(sigma, axis=1)**2)
        logPDF = -0.5 * tf.reduce_sum((x - mu)**2 / sigma**2, axis=1) - logZ
        # logPDF = -0.5 * tf.tensordot((x - mu)**2, sigma**-2, axes=1) - logZ
    return logPDF


class StateNetHypers:
    def __init__(self, hypers=None):
        self.z_dim = 10
        self.increase_zhat_sigma = True
        self.KL_lambda = 0.5
        self.estimator_version = 'B'
        self.learning_rate = 0.001
        self.optimizer = 'adam'

        if hypers:
            self.__dict__.update(hypers)

    def export(self):
        return self.__dict__


# class StateNet(tf.compat.v1.Module):
class StateNet:
    def __init__(self, state_shape, hypers, **kwargs):
        # super().__init__(**kwargs)

        self.state_shape = tuple(state_shape)

        assert isinstance(hypers, StateNetHypers)
        self.hypers = hypers

        self.traj_stats = {}

        self.graph = tf.Graph()
        with self.graph.as_default():

            # Scoping our graph elements so the saver plays nice with other
            # tensorflow stuff
            with tf.compat.v1.variable_scope('StateNet'):
                self._build_graph(self.state_shape, self.hypers)

            # For graph saving and loading
            self.saver = tf.compat.v1.train.Saver(
                var_list=tf.compat.v1.get_collection(
                    key=tf.GraphKeys.GLOBAL_VARIABLES,
                    scope='StateNet'))


    def _build_graph(self, state_shape, hypers):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.x1 = tf.compat.v1.placeholder(tf.float32,
            shape=(None, ) + state_shape,
            name='x_1')
        self.x2 = tf.compat.v1.placeholder(tf.float32,
            shape=(None, ) + state_shape,
            name='x_2')
        self.x3 = tf.compat.v1.placeholder(tf.float32,
            shape=(None, ) + state_shape,
            name='x_3')

        with tf.compat.v1.variable_scope('encoder'):
            self.encoder = Varnet_2(hypers.z_dim, name='encoder')
        with tf.compat.v1.variable_scope('decoder'):
            self.decoder = Varnet_2(state_shape, name='decoder')

        (self.z1_mu, self.z1_sigma) = self.encoder(self.x1)
        (self.z2_mu, self.z2_sigma) = self.encoder(self.x2)
        (self.z3_mu, self.z3_sigma) = self.encoder(self.x3)

        # Use identity to give some names
        z1_mu_ident = tf.identity(self.z1_mu, name="z1_mu")
        z1_sigma_ident = tf.identity(self.z1_sigma, name="z1_sigma")

        # z2 is the mean of z1 and z3
        # z2_hat has the distribution impled by $tilde{q}$
        self.z2_hat_mu = (self.z1_mu + self.z3_mu) / 2
        self.z2_hat_sigma = tf.sqrt( 0.25
            * (tf.square(self.z1_sigma) + tf.square(self.z3_sigma)))
        if hypers.increase_zhat_sigma:
            # This is (theortically) to account for the fact that z2 is drawn
            # from the same distribution as z1 and z3, but z2_hat will tend
            # to have a tighter distribution than z1 and z3 because of the
            # math of averaging random variables
            self.z2_hat_sigma = tf.sqrt( 0.5
                * (tf.square(self.z1_sigma) + tf.square(self.z3_sigma)))

        # To get sampled values for z we need the random variable per the
        # reparameterization trick
        self.epsilon1 = tf.random.normal(
            tf.shape(self.z1_mu), mean=0.0, stddev=1.0, name='epsilon_1')
        self.epsilon2 = tf.random.normal(
            tf.shape(self.z2_mu), mean=0.0, stddev=1.0, name='epsilon_2')
        self.epsilon3 = tf.random.normal(
            tf.shape(self.z3_mu), mean=0.0, stddev=1.0, name='epsilon_3')

        # Sample z
        # TODO: allow more than 1 sample per datapoint, this is optional as long
        #       as the minibatches are sufficiently large
        self.z1 = self.z1_mu + self.epsilon1 * self.z1_sigma
        self.z2 = self.z2_hat_mu + self.epsilon2 * self.z2_hat_sigma
        self.z3 = self.z3_mu + self.epsilon3 * self.z3_sigma

        (self.x1_mu, self.x1_sigma) = self.decoder(self.z1)
        (self.x2_mu, self.x2_sigma) = self.decoder(self.z2)
        (self.x3_mu, self.x3_sigma) = self.decoder(self.z3)

        # avoid log of exp to avoid numerical instability
        self.x1_logPDF = logPDF(self.x1, self.x1_mu, self.x1_sigma)
        self.x2_logPDF = logPDF(self.x2, self.x2_mu, self.x2_sigma)
        self.x3_logPDF = logPDF(self.x3, self.x3_mu, self.x3_sigma)

        # Estimate the evidence lower bound, which we wish to maximize
        # SGVB estimator B
        q_mu = tf.concat([self.z1_mu, self.z3_mu], axis=1)
        q_sig = tf.concat([self.z1_sigma, self.z3_sigma], axis=1)
        KL_prior = KL(
            q_mu,
            q_sig,
            tf.zeros(tf.shape(q_mu)),
            tf.ones(tf.shape(q_mu)))
        KL_z2 = KL(
            self.z2_hat_mu,
            self.z2_hat_sigma,
            self.z2_mu,
            self.z2_sigma)
        self.L_B = (
            - KL_prior
            + self.x1_logPDF
            + self.x2_logPDF
            + self.x3_logPDF
            - hypers.KL_lambda * KL_z2
            )
        self.L_B = tf.reduce_mean(self.L_B)

        if self.hypers.optimizer == 'adam':
            opt = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.hypers.learning_rate)
        elif self.hypers.optimizer == 'rmsprop':
            opt = tf.compat.v1.train.RMSPropOptimizer(
                learning_rate=self.hypers.learning_rate)
        else:
            print("ERROR: unrecognized optimizer: ", self.hypers.optimizer)

        # Choose our training op
        if self.hypers.estimator_version == 'B':
            # Note: we want to *maximize* L_B
            self.train_op = opt.minimize(-self.L_B,
                global_step=self.global_step)
        else:
            print("Training with estimator_version \"{}\" is not implemented"
                .format(self.hypers.estimator_version))
            raise NotImplementedError

        # Logging
        tf.compat.v1.summary.scalar("L_B", self.L_B)
        tf.compat.v1.summary.scalar("x1_logPDF (batch mean)",
            tf.reduce_mean(self.x1_logPDF))
        tf.compat.v1.summary.scalar("KL encoding vs prior (batch mean)",
            tf.reduce_mean(KL_prior))
        tf.compat.v1.summary.scalar("KL z2_hat vs z2 (batch mean)",
            tf.reduce_mean(KL_z2))
        # for v in tf.compat.v1.get_collection(
        #         tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES):
        #     tf.compat.v1.summary.histogram(v.name, v)
        self.merged_summaries = tf.compat.v1.summary.merge_all()


    def encode(self, sess, x):
        feed = {self.x1: x}
        [mu, sigma] = sess.run(
            [self.z1_mu, self.z1_sigma],
            feed_dict=feed)
        return mu, sigma


    def save_checkpoint(self, sess, path):
        save_path = self.saver.save(sess, path)
        print("Model checkpoint saved to: {}".format(save_path))


    def restore_checkpoint(self, sess, path):
        self.saver.restore(sess, path)
        print("Model restored from: {}".format(path))


    def train_step(self, sess, batch, writer=None):
        feed = {self.x1: batch[:, 0, :],
                self.x2: batch[:, 1, :],
                self.x3: batch[:, 2, :],
                }

        [_, summary, step] = sess.run(
            [self.train_op, self.merged_summaries, self.global_step],
            feed_dict=feed)

        if writer:
            writer.add_summary(summary, step)

        return step

    def eval_step(self, sess, traj_data, writer, dataset='train'):
        # Evaluate trajectory metrics for embedding space on traj_data
        print("Called eval_step()")

        global_step = sess.run(self.global_step)

        # Calculate evaluation statistics
        if not 'raw' in self.traj_stats:
            self.traj_stats['raw'] = traj_metrics.measure_traj_list(traj_data)
            self.traj_stats['norm'] = self.traj_stats['raw'][:,1] \
                                      / np.mean(self.traj_stats['raw'][:,1])

        embed_data = traj_metrics.embed_traj_list(
            traj_data, self, sess, endpoints=True)
        embed_stats = traj_metrics.measure_traj_list(embed_data)

        scale_factor, _, mean_error, stddev_error \
            = traj_metrics.find_metric_error(
                embed_stats[:,0], self.traj_stats['norm'])

        # Write stats to the log
        summary = tf.compat.v1.Summary(value=[
            tf.compat.v1.Summary.Value(
                tag="Metric_Error/" + dataset + "_mean",
                simple_value=mean_error)])
        writer.add_summary(summary, global_step)
        summary = tf.compat.v1.Summary(value=[
            tf.compat.v1.Summary.Value(
                tag="Metric_Error/" + dataset + "_stddev",
                simple_value=stddev_error)])
        writer.add_summary(summary, global_step)
        summary = tf.compat.v1.Summary(value=[
            tf.compat.v1.Summary.Value(
                tag="Metric_Error/" + dataset + "_scalefactor",
                simple_value=scale_factor)])
        writer.add_summary(summary, global_step)

# Dealing with a circular dependency
import traj_metrics
