import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from AlphaGomoku.core.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample
from AlphaGomoku.core.utils_2 import sample_without_exploration, normalized_columns_initializer


class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, size, nlstm=256, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = size * size  # ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32), 'c1', nf=16, rf=2, stride=1, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=32, rf=2, stride=1, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h2)
            h4 = fc(h3, 'fc1', nh=32, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x: x)
            vf = fc(h5, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = pi
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X: ob, S: state, M: mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, size, nlstm=64, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = size * size  # ac_space.n
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32), 'c1', nf=32, rf=2, stride=1, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=2, stride=1, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=32, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x: x)
            vf = fc(h5, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X: ob, S: state, M: mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, size, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = size * size  # ac_space.n
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        with tf.variable_scope("model", reusse=reuse):
            h = conv(tf.cast(X, tf.float32), 'c1', nf=32, rf=2, stride=1, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=2, stride=1, init_scale=np.sqrt(2))
            # h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h2)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x: x)
            vf = fc(h4, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = sample_without_exploration(pi)
        noise = tf.random_uniform(tf.shape(pi), tf.reduce_max(pi)) / 5
        pi = noise + pi
        p0 = pi
        self.initial_state = []  # not stateful

        def step(ob, *_args, **_kwargs):
            a, v, prob = sess.run([a0, v0, p0], {X: ob})
            # print('sum', (prob + 1) / (np.sum(prob) + len(prob[0])))
            return a, v, [], (prob + 1) / (np.sum(prob) + len(prob[0]))

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CnnPolicy2(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, size, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = size * size  # ac_space.n
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32), 'c1m', nf=32, rf=2, stride=1, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2m', nf=64, rf=2, stride=1, init_scale=np.sqrt(2))
            # h3 = conv(h2, 'c3m', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h2)
            h4 = fc(h3, 'fc1m', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pim', nact, act=lambda x: x)
            vf = fc(h4, 'vm', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = sample_without_exploration(pi)
        noise = tf.random_uniform(tf.shape(pi), tf.reduce_max(pi)) / 5
        pi = noise + pi
        p0 = pi
        self.initial_state = []  # not stateful

        def step(ob, *_args, **_kwargs):
            a, v, prob = sess.run([a0, v0, p0], {X: ob})
            # print('sum', (prob + 1) / (np.sum(prob) + len(prob[0])))
            return a, v, [], (prob + 1) / (np.sum(prob) + len(prob[0]))

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CnnPolicy_slim_TTT(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        TEMP = tf.placeholder(tf.float32, 1)
        with tf.variable_scope("model", reuse=reuse):
            conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                inputs=X, num_outputs=32,
                                kernel_size=[3, 3], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.01))
            # conv2 = slim.conv2d(activation_fn=tf.nn.elu,
            #                    inputs=conv1, num_outputs=64,
            #                    kernel_size=[2, 2], stride=[1, 1], padding='VALID')

            hidden = slim.fully_connected(slim.flatten(conv1), 512, activation_fn=tf.nn.relu)
            pi = slim.fully_connected(hidden, nact,
                                      activation_fn=None,
                                      weights_initializer=normalized_columns_initializer(0.01),
                                      biases_initializer=None,
                                      weights_regularizer=slim.l2_regularizer(0.01))
            vf = slim.fully_connected(hidden, 1,
                                      activation_fn=None,
                                      weights_initializer=normalized_columns_initializer(1.0),
                                      biases_initializer=None,
                                      weights_regularizer=slim.l2_regularizer(0.01))
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


class CnnPolicy_slim_TTT_2(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        TEMP = tf.placeholder(tf.float32, 1)
        with tf.variable_scope("model_2", reuse=reuse):
            conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                inputs=X, num_outputs=32,
                                kernel_size=[3, 3], stride=[1, 1], padding='VALID')
            # self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
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
        a0 = sample_without_exploration(pi / TEMP)
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
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space
        X = tf.placeholder(tf.float32, ob_shape)  # obs
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
                                      activation_fn=None,
                                      weights_initializer=normalized_columns_initializer(0.01),
                                      biases_initializer=None)
            vf = slim.fully_connected(hidden, 1,
                                      activation_fn=None,
                                      weights_initializer=normalized_columns_initializer(1.0),
                                      biases_initializer=None)
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


class CnnPolicy_slim_2(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        TEMP = tf.placeholder(tf.float32, 1)
        with tf.variable_scope("model_2", reuse=reuse):
            conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                inputs=X, num_outputs=32,
                                kernel_size=[4, 4], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.00005))
            conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                inputs=conv1, num_outputs=64,
                                kernel_size=[2, 2], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.00005))

            hidden = slim.fully_connected(slim.flatten(conv2), 512, activation_fn=tf.nn.relu)
            pi = slim.fully_connected(hidden, nact,
                                      activation_fn=None,
                                      weights_initializer=normalized_columns_initializer(0.01),
                                      biases_initializer=None,
                                      weights_regularizer=slim.l2_regularizer(0.00005))
            vf = slim.fully_connected(hidden, 1,
                                      activation_fn=None,
                                      weights_initializer=normalized_columns_initializer(1.0),
                                      biases_initializer=None,
                                      weights_regularizer=slim.l2_regularizer(0.00005))
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


class CnnPolicy_slim_scope5x5(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, scope, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        TEMP = tf.placeholder(tf.float32, 1)
        with tf.variable_scope(scope, reuse=reuse):
            conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=X, num_outputs=32,
                                kernel_size=[4, 4], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.0005))
            conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=conv1, num_outputs=64,
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


class CnnPolicy_slim_scope5x5_1x1(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, scope, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        print(ob_shape)
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
                                inputs=conv2, num_outputs=128,
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


class CnnPolicy_ni(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, scope, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        # ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space

        self.X = tf.placeholder(tf.float32, shape=[None, 4, nh, nw])
        self.X_reshaped = tf.reshape(self.X, [-1, nh, nw, 4])
        self.TEMP = tf.placeholder(tf.float32, 1)

        with tf.variable_scope(scope, reuse=reuse):
            conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=self.X_reshaped, num_outputs=32,
                                kernel_size=[3, 3], stride=[1, 1], padding='same')
            conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=conv1, num_outputs=64,
                                kernel_size=[3, 3], stride=[1, 1], padding='same')
            conv3 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=conv2, num_outputs=128,
                                kernel_size=[3, 3], stride=[1, 1], padding='same')

            action_conv = slim.conv2d(activation_fn=tf.nn.relu,
                                      inputs=conv3, num_outputs=4,
                                      kernel_size=[1, 1], stride=[1, 1], padding='same')

            action_conv_flat = tf.reshape(action_conv, [-1, 4 * nh * nw])
            pi = tf.layers.dense(action_conv_flat, nh * nw, activation=tf.nn.softmax)

            eval_conv = tf.layers.conv2d(inputs=conv3, filters=2, kernel_size=[1, 1],
                                         padding="same", activation=tf.nn.relu)
            eval_conv_flat = tf.reshape(eval_conv, [-1, 2 * nh * nw])
            conv_fc = tf.layers.dense(inputs=eval_conv_flat, units=nh * nw, activation=tf.nn.relu)
            vf = tf.layers.dense(inputs=conv_fc, units=1, activation=tf.nn.tanh)

            # pi = slim.fully_connected(pi, nact,
            #                           activation_fn=None,
            #                           weights_initializer=normalized_columns_initializer(0.01),
            #                           biases_initializer=None,
            #                           weights_regularizer=slim.l2_regularizer(0.0005))
            # vf = slim.fully_connected(pi, 1,
            #                           activation_fn=None,
            #                           weights_initializer=normalized_columns_initializer(1.0),
            #                           biases_initializer=None,
            #                           weights_regularizer=slim.l2_regularizer(0.0005))
        # pi = tf.nn.softmax(pi / TEMP)
        v0 = vf[:, 0]
        p0 = [pi]
        a0 = sample_without_exploration(pi)
        self.initial_state = []  # not stateful

        def step(ob, temp, *_args, **_kwargs):
            a, v, pi = sess.run([a0, v0, p0], {self.X: ob, self.TEMP: temp})
            return a, v, [], pi

        def value(ob, temp, *_args, **_kwargs):
            return sess.run(v0, {self.X: ob, self.TEMP: temp})

        def get_pi(ob, *_args, **_kwargs):
            a, v, pi = sess.run([a0, v0, p0], {self.X: ob})
            return a, v, [], pi

        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CnnPolicy_slim_scope9x9(object):

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
                                kernel_size=[5, 5], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.0005))
            conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=conv1, num_outputs=128,
                                kernel_size=[2, 2], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.0005))

            conv3 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=conv2, num_outputs=256,
                                kernel_size=[2, 2], stride=[1, 1], padding='VALID',
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


class CnnPolicy_slim_scope9x9_1x1(object):

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
                                kernel_size=[5, 5], stride=[1, 1], padding='VALID',
                                weights_regularizer=slim.l2_regularizer(0.0005))
            conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                                inputs=conv1, num_outputs=128,
                                kernel_size=[2, 2], stride=[1, 1], padding='VALID',
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
