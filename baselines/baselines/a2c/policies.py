import numpy as np
import tensorflow as tf

from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample, \
    sample_without_exploration, sample_K


class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = 9  # ac_space.n
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

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=64, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = 9  # ac_space.n
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

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = 9  # ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32) / 255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x: x)
            vf = fc(h4, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = []  # not stateful

        def step(ob, *_args, **_kwargs):
            print('what is ob', np.mean(ob))
            a, v = sess.run([a0, v0], {X: ob})
            return a, v, []  # dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space*ac_space
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32), 'c1', nf=32, rf=3, stride=1, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h2)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x: x)
            vf = fc(h4, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = sample(pi,nact)
        p0 = pi
        self.initial_state = []  # not stateful

        def step(ob, *_args, **_kwargs):
            a, v,prob = sess.run([a0, v0, p0], {X: ob})
            return a, v, [],prob  # dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})


        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy_VS(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space*ac_space
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32), 'c1_B', nf=32, rf=3, stride=1, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2_B', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3_B', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h2)
            h4 = fc(h3, 'fc1_B', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi_B',nact, act=lambda x: x)
            vf = fc(h4, 'v_B', 1,act=lambda x: x)

        v0 = vf[:, 0]
        a0 = sample_K(pi,nact)
        p0 = pi
        self.initial_state = []  # not stateful

        def step(ob, *_args, **_kwargs):
            a, v,prob = sess.run([a0, v0, p0], {X: ob})
            return a, v, [],prob  # dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})


        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy_TTT(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space*ac_space
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32), 'c1', nf=32, rf=2, stride=1, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=2, stride=1, init_scale=np.sqrt(2))
            #h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h2)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x: x)
            vf = fc(h4, 'v',1, act=lambda x: x)

        v0 = vf[:, 0]
        noise = tf.random_uniform(tf.shape(pi), tf.reduce_max(pi)) / 2
        pi = noise + pi
        a0 = sample_K(pi,nact)
        p0 = [pi]
        self.initial_state = []  # not stateful

        def step(ob, *_args, **_kwargs):
            a, v,prob = sess.run([a0, v0, p0], {X: ob})
            return a, v, [],prob  # dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})


        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy_VS_TTT(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space*ac_space
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32), 'c1_B', nf=32, rf=2, stride=1, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2_B', nf=64, rf=2, stride=1, init_scale=np.sqrt(2))
            #h3 = conv(h2, 'c3_B', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h)
            h4 = fc(h3, 'fc1_B', nh=128, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi_B', nact, act=lambda x: x)
            vf = fc(h4, 'v_B', 1,act=lambda x: x)

        v0 = vf[:, 0]
        noise = tf.random_uniform(tf.shape(pi), tf.reduce_max(pi)) / 5
        pi = noise + pi
        a0 = sample_K(pi,nact)
        p0 = pi
        self.initial_state = []  # not stateful

        def step(ob, *_args, **_kwargs):
            a, v,prob = sess.run([a0, v0, p0], {X: ob})
            return a, v, [],prob  # dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})


        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
