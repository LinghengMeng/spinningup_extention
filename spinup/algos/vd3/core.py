import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli

class VariationalDense:
    """Variational Dense Layer Class"""
    def __init__(self, n_in, n_out, model_prob=0.9, model_lam=1e-2, name="hidden"):
        self.model_prob = model_prob    # probability to keep units
        self.model_lam = model_lam      # l^2 / 2*tau
        self.model_bern = Bernoulli(probs=self.model_prob, dtype=tf.float32)
        # with tf.variable_scope("variational_dense"):
        self.model_M = tf.get_variable("{}_M".format(name), initializer=tf.truncated_normal([n_in, n_out], stddev=0.01))
        self.model_m = tf.get_variable("{}_b".format(name), initializer=tf.zeros([n_out]))
        self.model_W = tf.matmul(
            tf.diag(self.model_bern.sample((n_in, ))), self.model_M
        )

    def __call__(self, X, activation=tf.identity):
        if activation is None:
            activation = tf.identity
        output = activation(tf.matmul(X, self.model_W) + self.model_m)
        # if self.model_M.shape[1] == 1:
        #     output = tf.squeeze(output)
        return output

    @property
    def regularization(self):
        return self.model_lam * (
            self.model_prob * tf.reduce_sum(tf.square(self.model_M)) +
            tf.reduce_sum(tf.square(self.model_m))
        )

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def mlp_variational(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    # Hidden layers
    regularization = 0#tf.placeholder(dtype=tf.float32, shape=(1,))
    for l, h in enumerate(hidden_sizes[:-1]):
        hidden_layer = VariationalDense(n_in=x.shape.as_list()[1],
                             n_out=h,
                             model_prob=0.9,
                             model_lam=1e-2,
                             name="h{}".format(l+1))
        x = hidden_layer(x, activation)
        regularization += hidden_layer.regularization
    # Output layer
    out_layer = VariationalDense(n_in=x.shape.as_list()[1],
                                 n_out=hidden_sizes[-1],
                                 model_prob=0.9,
                                 model_lam=1e-2,
                                 name="Out")
    x = out_layer(x, output_activation)
    regularization += out_layer.regularization
    return x, regularization

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

"""
Variational Actor-Critics
"""
def mlp_actor_critic_variational(x, a, hidden_sizes=(400,300), activation=tf.nn.relu,
                                 output_activation=tf.tanh, action_space=None):
    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]
    with tf.variable_scope('pi'):
        # pi, pi_reg = mlp_variational(x, list(hidden_sizes)+[act_dim], activation, output_activation)
        # pi = pi * act_limit
        pi = act_limit * mlp(x, list(hidden_sizes) + [act_dim], activation, output_activation)
    with tf.variable_scope('q'):
        q, q_reg = mlp_variational(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None)
        q = tf.squeeze(q, axis=1)
    with tf.variable_scope('q', reuse=tf.AUTO_REUSE):
        q_pi, q_pi_reg = mlp_variational(tf.concat([x,pi], axis=-1), list(hidden_sizes)+[1], activation, None)
        q_pi = tf.squeeze(q_pi, axis=1)
    return pi, q, q_reg, q_pi, q_pi_reg