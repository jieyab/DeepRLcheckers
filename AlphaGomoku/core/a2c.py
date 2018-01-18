import time

import joblib
import numpy as np
import tensorflow as tf

from AlphaGomoku.common import logger
from AlphaGomoku.common.math_util import explained_variance
from AlphaGomoku.common.misc_util import set_global_seeds
from AlphaGomoku.core.tree_search import MonteCarlo
from AlphaGomoku.core.utils import Scheduler, find_trainable_variables
from AlphaGomoku.core.utils import cat_entropy, mse
from AlphaGomoku.core.utils import discount_with_dones


class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nact = 9  # ac_space.n
        nbatch = nenvs * nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X: obs, A: actions, ADV: advs, R: rewards, LR: cur_lr}
            if states:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            # make_path(save_path)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)


class Runner(object):

    def __init__(self, env, model, nsteps=5, nstack=4, gamma=0.7):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.nenv = env.num_envs
        self.nh = nh
        self.nw = nw
        self.nc = nc
        self.nstack = nstack
        self.batch_ob_shape = (nenv * nsteps, nh, nw, nc * nstack)
        self.obs = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.float32)
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

        self.mcts = MonteCarlo(env, self.model)
        self.list_board = []
        self.list_ai_board = []

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-1, axis=3)
        self.obs[:, :, :, -1] = obs[:, :, :, 0]

    def run(self):
        logger.debug('- ' * 20 + 'run' + ' -' * 20)
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_states = self.states
        mb_dones.append(np.zeros(1, dtype=bool))

        node, reward = self.mcts.self_play()
        plus_state, minus_state, plus_action, minus_action = self.mcts.get_state(node)

        for i in range(len(plus_state)):
            mb_obs.append(np.copy(plus_state[i]))
            mb_actions.append(np.ones(1, dtype=int) * plus_action[i])

            if i == len(plus_state) - 1:
                value = self.model.value(plus_state[i], self.states, np.ones(1, dtype=bool))
                mb_values.append(value)
                mb_dones.append(np.ones(1, dtype=bool))
                if reward:
                    mb_rewards.append(np.ones(1))
                else:
                    mb_rewards.append(np.zeros(1))
            else:
                value = self.model.value(plus_state[i], self.states, np.zeros(1, dtype=bool))
                mb_values.append(value)
                mb_dones.append(np.zeros(1, dtype=bool))
                mb_rewards.append(np.zeros(1))

        mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0).reshape(
            (self.nenv * len(plus_state), self.nh, self.nw, self.nc * self.nstack))
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        last_values = self.model.value(plus_state[-1], self.states, self.dones).tolist()
        logger.info('Last_values: ', last_values)

        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()

        for ob in mb_obs:
            print(ob.tolist())
        print('actions', mb_actions)
        print('values', mb_values)
        print('rewards', mb_rewards)
        print('masks', mb_masks)
        print('status', mb_states)

        dim_necessary = self.nsteps - mb_obs.shape[0]
        mb_obs = np.concatenate((mb_obs, np.zeros((dim_necessary, 3, 3, 1))), axis=0)
        mb_rewards = np.concatenate((mb_rewards, np.zeros(dim_necessary)), axis=0)
        mb_masks = np.concatenate((mb_masks, np.full(dim_necessary, True, dtype=bool)), axis=0)
        mb_actions = np.concatenate((mb_actions, np.zeros(dim_necessary)), axis=0)
        mb_values = np.concatenate((mb_values, np.zeros(dim_necessary)), axis=0)

        policy_loss, value_loss, policy_entropy = self.model.train(mb_obs, mb_states, mb_rewards, mb_masks, mb_actions,
                                                                   mb_values)
        logger.debug('* ' * 20 + 'run' + ' *' * 20)
        return policy_loss, value_loss, policy_entropy


def learn(policy, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01,
          max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=20,
          load_model=False, model_path=''):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes)  # HACK

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack,
                  num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                  lrschedule=lrschedule)

    if load_model:
        model.load(model_path)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs * nsteps
    tstart = time.time()

    for update in range(1, total_timesteps // nbatch + 1):
        policy_loss, value_loss, policy_entropy = runner.run()

        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.dump_tabular()
        if (update % (log_interval * 10)) == 0:
            model.save('../models/gomoku.cpkt')

    runner.mcts.visualization()
    env.close()


if __name__ == '__main__':
    import AlphaGomoku.games.tic_tac_toe_x as tt

    from gym import spaces

    observation_space = spaces.Box(low=-1, high=1, shape=(3, 3, 1))
    nh, nw, nc = observation_space.shape
    print(nh, nw, nc)

    mb_obs = [tt.new_board(3), tt.new_board(3), tt.new_board(3), tt.new_board(3), tt.new_board(3)]
    mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0).reshape(
        (1 * 3, nh, nw, nc * 5))
    print(mb_obs)
