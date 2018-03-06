import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from AlphaGomoku.core.utils_2 import sample_without_exploration, normalized_columns_initializer


class policy_4x4_2x2(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, scope, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        TEMP = tf.placeholder(tf.float32, 1)
        with tf.variable_scope(scope, reuse=reuse):
            conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=X, num_outputs=64,
                                kernel_size=[4, 4], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.0005))
            conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=conv1, num_outputs=128,
                                kernel_size=[2, 2], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.0005))

            hidden = slim.fully_connected(slim.flatten(conv2), 512, activation_fn=tf.nn.relu)
            pi = slim.fully_connected(hidden, nact,
                                      activation_fn=None,
                                      weights_initializer=normalized_columns_initializer(0.01),
                                      biases_initializer=None,
                                      weights_regularizer=slim.l2_regularizer(0.0005))
            vf = slim.fully_connected(hidden, 1,
                                      activation_fn=None,
                                      weights_initializer=normalized_columns_initializer(1.0),
                                      biases_initializer=None,
                                      weights_regularizer=slim.l2_regularizer(0.0005))
        # pi = tf.nn.softmax(pi / TEMP)
        v0 = vf[:, 0]
        p0 = [pi]
        a0 = sample_without_exploration(pi)
        self.initial_state = []  # not stateful

        def step(ob, temp, *_args, **_kwargs):
            a, v, pi = sess.run([a0, v0, p0], {X: ob, TEMP: temp})
            return a, v, [], pi

        def value(ob, temp, *_args, **_kwargs):
            return sess.run(v0, {X: ob, TEMP: temp})

        def get_pi(ob, *_args, **_kwargs):
            a, v, pi = sess.run([a0, v0, p0], {X: ob})
            return a, v, [], pi

        self.X = X
        self.TEMP = TEMP
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class policy_4x4_2x2_1x1(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, scope, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        TEMP = tf.placeholder(tf.float32, 1)
        with tf.variable_scope(scope, reuse=reuse):
            conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=X, num_outputs=64,
                                kernel_size=[4, 4], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.0005))
            conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=conv1, num_outputs=128,
                                kernel_size=[2, 2], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.0005))
            conv3 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=conv2, num_outputs=64,
                                kernel_size=[1, 1], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.0005))


            hidden = slim.fully_connected(slim.flatten(conv3), 512, activation_fn=tf.nn.relu)
            pi = slim.fully_connected(hidden, nact,
                                      activation_fn=None,
                                      weights_initializer=normalized_columns_initializer(0.01),
                                      biases_initializer=None,
                                      weights_regularizer=slim.l2_regularizer(0.0005))
            vf = slim.fully_connected(hidden, 1,
                                      activation_fn=None,
                                      weights_initializer=normalized_columns_initializer(1.0),
                                      biases_initializer=None,
                                      weights_regularizer=slim.l2_regularizer(0.0005))
        # pi = tf.nn.softmax(pi / TEMP)
        v0 = vf[:, 0]
        p0 = [pi]
        a0 = sample_without_exploration(pi)
        self.initial_state = []  # not stateful

        def step(ob, temp, *_args, **_kwargs):
            a, v, pi = sess.run([a0, v0, p0], {X: ob, TEMP: temp})
            return a, v, [], pi

        def value(ob, temp, *_args, **_kwargs):
            return sess.run(v0, {X: ob, TEMP: temp})

        def get_pi(ob, *_args, **_kwargs):
            a, v, pi = sess.run([a0, v0, p0], {X: ob})
            return a, v, [], pi

        self.X = X
        self.TEMP = TEMP
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class policy_4x4_1x1_2x2_1x1(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, scope, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        TEMP = tf.placeholder(tf.float32, 1)
        with tf.variable_scope(scope, reuse=reuse):
            conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=X, num_outputs=128,
                                kernel_size=[4, 4], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.0005))
            conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=conv1, num_outputs=64,
                                kernel_size=[1, 1], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.0005))
            conv3 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=conv2, num_outputs=128,
                                kernel_size=[2, 2], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.0005))
            conv4 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=conv3, num_outputs=64,
                                kernel_size=[1, 1], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.0005))

            hidden = slim.fully_connected(slim.flatten(conv4), 512, activation_fn=tf.nn.relu)
            pi = slim.fully_connected(hidden, nact,
                                      activation_fn=None,
                                      weights_initializer=normalized_columns_initializer(0.01),
                                      biases_initializer=None,
                                      weights_regularizer=slim.l2_regularizer(0.0005))
            vf = slim.fully_connected(hidden, 1,
                                      activation_fn=None,
                                      weights_initializer=normalized_columns_initializer(1.0),
                                      biases_initializer=None,
                                      weights_regularizer=slim.l2_regularizer(0.0005))
        # pi = tf.nn.softmax(pi / TEMP)
        v0 = vf[:, 0]
        p0 = [pi]
        a0 = sample_without_exploration(pi)
        self.initial_state = []  # not stateful

        def step(ob, temp, *_args, **_kwargs):
            a, v, pi = sess.run([a0, v0, p0], {X: ob, TEMP: temp})
            return a, v, [], pi


        def value(ob, temp, *_args, **_kwargs):
            return sess.run(v0, {X: ob, TEMP: temp})

        def get_pi(ob, *_args, **_kwargs):
            a, v, pi = sess.run([a0, v0, p0], {X: ob})
            return a, v, [], pi

        self.X = X
        self.TEMP = TEMP
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class policy_4x4_1x1_2x2_1x1_features(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, scope, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        TEMP = tf.placeholder(tf.float32, 1)
        with tf.variable_scope(scope, reuse=reuse):
            conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=X, num_outputs=256,
                                kernel_size=[4, 4], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.0005))
            conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=conv1, num_outputs=128,
                                kernel_size=[1, 1], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.0005))
            conv3 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=conv2, num_outputs=256,
                                kernel_size=[2, 2], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.0005))
            conv4 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=conv3, num_outputs=128,
                                kernel_size=[1, 1], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.0005))

            hidden = slim.fully_connected(slim.flatten(conv4), 512, activation_fn=tf.nn.relu)
            pi = slim.fully_connected(hidden, nact,
                                      activation_fn=None,
                                      weights_initializer=normalized_columns_initializer(0.01),
                                      biases_initializer=None,
                                      weights_regularizer=slim.l2_regularizer(0.0005))
            vf = slim.fully_connected(hidden, 1,
                                      activation_fn=None,
                                      weights_initializer=normalized_columns_initializer(1.0),
                                      biases_initializer=None,
                                      weights_regularizer=slim.l2_regularizer(0.0005))
        # pi = tf.nn.softmax(pi / TEMP)
        v0 = vf[:, 0]
        p0 = [pi]
        a0 = sample_without_exploration(pi)
        self.initial_state = []  # not stateful

        def step(ob, temp, *_args, **_kwargs):
            a, v, pi = sess.run([a0, v0, p0], {X: ob, TEMP: temp})
            return a, v, [], pi

        def value(ob, temp, *_args, **_kwargs):
            return sess.run(v0, {X: ob, TEMP: temp})

        def get_pi(ob, *_args, **_kwargs):
            a, v, pi = sess.run([a0, v0, p0], {X: ob})
            return a, v, [], pi

        self.X = X
        self.TEMP = TEMP
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value