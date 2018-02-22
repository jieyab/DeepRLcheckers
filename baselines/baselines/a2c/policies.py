import numpy as np
import tensorflow as tf
from baselines.a2c.utils import sample_without_exploration, normalized_columns_initializer
import tensorflow.contrib.slim as slim


class CnnPolicy_slim_TTT(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space
        X = tf.placeholder(tf.float32, ob_shape) #obs
        TEMP = tf.placeholder(tf.float32, 1)
        with tf.variable_scope("model", reuse=reuse):
            conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                inputs=X, num_outputs=32,
                                kernel_size=[3, 3], stride=[1, 1], padding='VALID')
            #self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
            #                         inputs=self.conv1, num_outputs=32,
            #                         kernel_size=[4, 4], stride=[2, 2], padding='VALID')

            hidden = slim.fully_connected(slim.flatten(conv1), 512, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.001)
)
            pi = slim.fully_connected(hidden, nact,
                                      activation_fn = None,
                                      weights_initializer = normalized_columns_initializer(0.01),
                                      biases_initializer = None)
            vf = slim.fully_connected(hidden, 1,
                                         activation_fn=None,
                                         weights_initializer=normalized_columns_initializer(1.0),
                                         biases_initializer=None)
        pi = tf.nn.softmax(pi/TEMP)

        v0 = vf[:, 0]
        p0 = [pi]
        a0 = sample_without_exploration(pi)
        self.initial_state = [] #not stateful
        def step(ob, temp, *_args, **_kwargs):
            a, v, pi = sess.run([a0, v0, p0], {X: ob, TEMP: temp})
            return a, v, [], pi

        def value(ob, temp, *_args, **_kwargs):
            return sess.run(v0, {X: ob, TEMP: temp})

        self.X = X
        self.TEMP = TEMP
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CnnPolicy_slim_TTT_2(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space
        X = tf.placeholder(tf.float32, ob_shape) #obs
        TEMP = tf.placeholder(tf.float32, 1)
        with tf.variable_scope("model_2", reuse=reuse):
            conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=X, num_outputs=32,
                                     kernel_size=[3, 3], stride=[1, 1], padding='VALID')
            #self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
            #                         inputs=self.conv1, num_outputs=32,
            #                         kernel_size=[4, 4], stride=[2, 2], padding='VALID')

            hidden = slim.fully_connected(slim.flatten(conv1), 512, activation_fn=tf.nn.relu,
                                          weights_regularizer=slim.l2_regularizer(0.001)
                                          )
            pi = slim.fully_connected(hidden, nact,
                                      activation_fn=None,
                                      weights_initializer=normalized_columns_initializer(0.01),
                                      biases_initializer=None)
            vf = slim.fully_connected(hidden, 1,
                                      activation_fn=None,
                                      weights_initializer=normalized_columns_initializer(1.0),
                                      biases_initializer=None)
            pi = tf.nn.softmax(pi)

        v0 = vf[:, 0]
        p0 = [pi]
        a0 = sample_without_exploration(pi/TEMP)
        self.initial_state = []  # not stateful

        def step(ob, temp, *_args, **_kwargs):
            a, v, pi = sess.run([a0, v0, p0], {X: ob, TEMP: temp})
            return a, v, [], pi

        def value(ob, temp, *_args, **_kwargs):
            return sess.run(v0, {X: ob, TEMP: temp})

        self.X = X
        self.TEMP = TEMP
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CnnPolicy_slim(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space
        X = tf.placeholder(tf.float32, ob_shape) #obs
        TEMP = tf.placeholder(tf.float32, 1)
        with tf.variable_scope("model", reuse=reuse):
            conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=X, num_outputs=32,
                                     kernel_size=[4, 4], stride=[1, 1], padding='VALID')
            conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=conv1, num_outputs=64,
                                     kernel_size=[2, 2], stride=[1, 1], padding='VALID')

            hidden = slim.fully_connected(slim.flatten(conv2), 512, activation_fn=tf.nn.relu)
            pi = slim.fully_connected(hidden, nact,
                                      activation_fn = None,
                                      weights_initializer = normalized_columns_initializer(0.01),
                                      biases_initializer = None)
            vf = slim.fully_connected(hidden, 1,
                                         activation_fn=None,
                                         weights_initializer=normalized_columns_initializer(1.0),
                                         biases_initializer=None)
        pi = tf.nn.softmax(pi/TEMP)
        v0 = vf[:, 0]
        p0 = [pi]
        a0 = sample_without_exploration(pi)
        self.initial_state = [] #not stateful
        def step(ob, temp, *_args, **_kwargs):
            a, v, pi = sess.run([a0, v0, p0], {X: ob, TEMP: temp})
            return a, v, [], pi

        def value(ob, temp, *_args, **_kwargs):
            return sess.run(v0, {X: ob, TEMP: temp})

        def get_pi(ob, *_args, **_kwargs):
            a, v, pi = sess.run([a0, v0, p0], {X:ob})
            return a, v, [], pi

        self.X = X
        self.TEMP = TEMP
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy_slim_2(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space
        X = tf.placeholder(tf.float32, ob_shape) #obs
        TEMP = tf.placeholder(tf.float32, 1)
        with tf.variable_scope("model_2", reuse=reuse):
            conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=X, num_outputs=32,
                                     kernel_size=[4, 4], stride=[1, 1], padding='VALID')
            conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=conv1, num_outputs=64,
                                     kernel_size=[2, 2], stride=[1, 1], padding='VALID')

            hidden = slim.fully_connected(slim.flatten(conv2), 512, activation_fn=tf.nn.relu)
            pi = slim.fully_connected(hidden, nact,
                                      activation_fn = None,
                                      weights_initializer = normalized_columns_initializer(0.01),
                                      biases_initializer = None)
            vf = slim.fully_connected(hidden, 1,
                                         activation_fn=None,
                                         weights_initializer=normalized_columns_initializer(1.0),
                                         biases_initializer=None)
        pi = tf.nn.softmax(pi/TEMP)
        v0 = vf[:, 0]
        p0 = [pi]
        a0 = sample_without_exploration(pi)
        self.initial_state = [] #not stateful
        def step(ob, temp, *_args, **_kwargs):
            a, v, pi = sess.run([a0, v0, p0], {X: ob, TEMP: temp})
            return a, v, [], pi

        def value(ob, temp, *_args, **_kwargs):
            return sess.run(v0, {X: ob, TEMP: temp})

        def get_pi(ob, *_args, **_kwargs):
            a, v, pi = sess.run([a0, v0, p0], {X:ob})
            return a, v, [], pi

        self.X = X
        self.TEMP = TEMP
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

