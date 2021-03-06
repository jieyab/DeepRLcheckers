import time
from statistics import mean

import joblib
import numpy as np
import tensorflow as tf
import copy
import csv

from AlphaGomoku.common import logger
from AlphaGomoku.common.misc_util import set_global_seeds
from AlphaGomoku.core.tree_search import MonteCarlo
from AlphaGomoku.core.utils import Scheduler, find_trainable_variables
from AlphaGomoku.core.utils import cat_entropy, mse
from AlphaGomoku.core.utils import discount_with_dones


class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs, size,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nbatch = nenvs * nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, size, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, size, reuse=True)

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


class Model2(object):
    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
                 ent_coef=0.1, vf_coef=0.5, max_grad_norm=0.05, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear', summary_writter=''):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nact = ac_space
        nbatch = nenvs * nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)
        q = tf.one_hot(A, nact, dtype=tf.float32)
        neglogpac = -tf.reduce_sum(tf.log((train_model.pi) + 1e-10) * q, [1])

        # neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
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

        def train(obs, states, rewards, masks, actions, values, temp):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X: obs, A: actions, ADV: advs, R: rewards, LR: cur_lr, train_model.TEMP: temp}
            if states != []:
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
        summary_writter.add_graph(sess.graph)


class Runner(object):

    def __init__(self, env, model, model2, nsteps=5, nstack=4, gamma=0.7):
        self.env = env
        self.model = model
        self.model2 = model2
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
        if model is None:
            self.states = None
        else:
            self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

        self.mcts = MonteCarlo(env, self.model, self.model2)
        self.list_board = []
        self.list_ai_board = []
        self.matrix_actions = np.zeros((nh, nh))

        for i in range(nh * nh):
            actions = i
            xy = [int(actions % nh),
                  int(actions / nh)]
            self.matrix_actions[xy[0], xy[1]] = i

    def exchange_models(self):
        tmp = self.model
        self.model = self.model2
        self.model2 = tmp

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-1, axis=3)
        self.obs[:, :, :, -1] = obs[:, :, :, 0]

    def pad_training_data(self, list_state, list_action, reward, model):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_states = self.states
        mb_dones.append(np.zeros(1, dtype=bool))

        for i in range(len(list_state)):
            mb_obs.append(np.copy(list_state[i]))
            mb_actions.append(np.ones(1, dtype=int) * list_action[i])

            if i == len(list_state) - 1:
                value = model.value(list_state[i], self.states, np.ones(1, dtype=bool))
                mb_values.append(value)
                mb_dones.append(np.ones(1, dtype=bool))
                if reward == 1:
                    mb_rewards.append(np.ones(1))
                elif reward == -1:
                    mb_rewards.append(np.ones(1) * -1)
                else:
                    mb_rewards.append(np.zeros(1))
            else:
                value = model.value(list_state[i], self.states, np.zeros(1, dtype=bool))
                mb_values.append(value)
                mb_dones.append(np.zeros(1, dtype=bool))
                mb_rewards.append(np.zeros(1))

        mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0).reshape(
            (self.nenv * len(list_state), self.nh, self.nw, self.nc * self.nstack))
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        last_values = model.value(list_state[-1], self.states, self.dones).tolist()
        logger.info('Last_values: ', str(last_values))

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

        print('- ' * 50)
        # state = str(mb_obs[-1]).replace(']]', ']').replace(']]', '').replace('[[[[', '').replace('\n', '') \
        #     .replace('[[', '\n').replace('[', '').replace(' ', '').replace(']', ' | ').replace(' | \n', '\n')
        # print(state)
        # self.mcts.show_state(mb_obs[-1])
        # for ob in mb_obs:
        #     print(ob.tolist())
        actions = []
        for action in mb_actions:
            actions.append([int(action % self.env.get_board_size()), int(action / self.env.get_board_size())])
        print('actions', actions)
        print('values', mb_values)
        print('rewards', mb_rewards)
        # print('masks', mb_masks)
        # print('status', mb_states)

        dim_necessary = self.nsteps - mb_obs.shape[0]
        mb_obs = np.concatenate(
            (mb_obs, np.zeros((dim_necessary, self.env.get_board_size(), self.env.get_board_size(), 1))), axis=0)
        mb_rewards = np.concatenate((mb_rewards, np.zeros(dim_necessary)), axis=0)
        mb_masks = np.concatenate((mb_masks, np.full(dim_necessary, True, dtype=bool)), axis=0)
        mb_actions = np.concatenate((mb_actions, np.zeros(dim_necessary)), axis=0)
        mb_values = np.concatenate((mb_values, np.zeros(dim_necessary)), axis=0)

        policy_loss, value_loss, policy_entropy = model.train(mb_obs, mb_states, mb_rewards, mb_masks, mb_actions,
                                                              mb_values)
        return float(policy_loss), float(value_loss), float(policy_entropy)

    def run(self):
        logger.debug('- ' * 20 + 'run' + ' -' * 20)
        node, reward = self.mcts.self_play()
        plus_state, minus_state, plus_action, minus_action = self.mcts.get_state(node)
        policy_loss, value_loss, policy_entropy = [], [], []
        policy_loss_2, value_loss_2, policy_entropy_2 = [], [], []

        # Train normal
        p1, v1, e1 = self.pad_training_data(plus_state, plus_action, reward, self.model)
        p2_2, v2_2, e2_2 = self.pad_training_data(minus_state, minus_action, -reward, self.model2)

        policy_loss.append(p1)
        policy_loss_2.append(p2_2)
        value_loss.append(v1)
        value_loss_2.append(v2_2)
        policy_entropy.append(e1)
        policy_entropy_2.append(e2_2)

        # Rotation ACLW 180

        rot_plus_action = copy.copy(plus_action)
        rot_minus_action = copy.copy(minus_action)

        rot_matrix = np.rot90(self.matrix_actions, 2)
        rot_plus_state = copy.copy(plus_state)
        rot_minus_state = copy.copy(minus_state)
        print(len(plus_state), len(minus_state))
        for i in range(len(plus_action)):
            rot_plus_state[i] = np.rot90(plus_state[i], 2)
            rot_plus_action[i] = rot_matrix[int(plus_action[i] % self.nh),
                                            int(plus_action[i] / self.nh)]

        for i in range(len(minus_action)):
            rot_minus_state[i] = np.rot90(minus_state[i], 2)
            rot_minus_action[i] = rot_matrix[int(minus_action[i] % self.nh),
                                             int(minus_action[i] / self.nh)]

        p1, v1, e1 = self.pad_training_data(rot_plus_state, rot_plus_action, reward, self.model)
        p2_2, v2_2, e2_2 = self.pad_training_data(rot_minus_state, rot_minus_action, -reward, self.model2)

        policy_loss.append(p1)
        policy_loss_2.append(p2_2)
        value_loss.append(v1)
        value_loss_2.append(v2_2)
        policy_entropy.append(e1)
        policy_entropy_2.append(e2_2)

        # Rotation ACLW 90
        rot_matrix = np.rot90(self.matrix_actions, 1)

        for i in range(len(plus_action)):
            rot_plus_state[i][0, :, :, 0] = np.rot90(plus_state[i][0, :, :, 0], 1)
            rot_plus_action[i] = rot_matrix[int(plus_action[i] % self.nh),
                                            int(plus_action[i] / self.nh)]

        for i in range(len(minus_action)):
            rot_minus_state[i][0, :, :, 0] = np.rot90(minus_state[i][0, :, :, 0], 1)
            rot_minus_action[i] = rot_matrix[int(minus_action[i] % self.nh),
                                             int(minus_action[i] / self.nh)]

        p1, v1, e1 = self.pad_training_data(rot_plus_state, rot_plus_action, reward, self.model)
        p2_2, v2_2, e2_2 = self.pad_training_data(rot_minus_state, rot_minus_action, -reward, self.model2)

        policy_loss.append(p1)
        policy_loss_2.append(p2_2)
        value_loss.append(v1)
        value_loss_2.append(v2_2)
        policy_entropy.append(e1)
        policy_entropy_2.append(e2_2)

        # Rotation ACLW 270
        rot_matrix = np.rot90(self.matrix_actions, 3)

        for i in range(len(plus_action)):
            rot_plus_state[i][0, :, :, 0] = np.rot90(plus_state[i][0, :, :, 0], 3)
            rot_plus_action[i] = rot_matrix[int(plus_action[i] % self.nh),
                                            int(plus_action[i] / self.nh)]

        for i in range(len(minus_action)):
            rot_minus_state[i][0, :, :, 0] = np.rot90(minus_state[i][0, :, :, 0], 3)
            rot_minus_action[i] = rot_matrix[int(minus_action[i] % self.nh),
                                             int(minus_action[i] / self.nh)]

        p1, v1, e1 = self.pad_training_data(rot_plus_state, rot_plus_action, reward, self.model)
        p2_2, v2_2, e2_2 = self.pad_training_data(rot_minus_state, rot_minus_action, -reward, self.model2)

        policy_loss.append(p1)
        policy_loss_2.append(p2_2)
        value_loss.append(v1)
        value_loss_2.append(v2_2)
        policy_entropy.append(e1)
        policy_entropy_2.append(e2_2)
        self.exchange_models()
        self.mcts.exchange_models()

        logger.debug('* ' * 20 + 'run' + ' *' * 20)
        return mean(policy_loss), mean(value_loss), mean(policy_entropy), mean(policy_loss_2), mean(value_loss_2), mean(
            policy_entropy_2)


def learn(policy, policy2, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01,
          max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=20,
          load_model=False, model_path='', model_path2=''):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes)  # HACK

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack,
                  num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef, size=env.get_board_size(),
                  max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                  lrschedule=lrschedule)

    model2 = Model(policy=policy2, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack,
                   num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef, size=env.get_board_size(),
                   max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                   lrschedule=lrschedule)

    if load_model:
        model.load(model_path)
        model2.load(model_path2)
    runner = Runner(env, model, model2, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs * nsteps
    tstart = time.time()
    policy_loss_saver, value_loss_saver, policy_entropy_saver = [], [], []
    policy_loss_saver_2, value_loss_saver_2, policy_entropy_saver_2 = [], [], []
    for update in range(1, total_timesteps // nbatch + 1):
        policy_loss, value_loss, policy_entropy, policy_loss_2, value_loss_2, policy_entropy_2 = runner.run()

        policy_loss_saver.append(str(policy_loss))
        value_loss_saver.append(value_loss)
        policy_entropy_saver.append(policy_entropy)
        policy_loss_saver_2.append(str(policy_loss_2))
        value_loss_saver_2.append(value_loss_2)
        policy_entropy_saver_2.append(policy_entropy_2)

        nseconds = time.time() - tstart
        fps = float((update * nbatch) / nseconds)

        if update % log_interval == 0 or update == 1:
            runner.mcts.print_statistic()
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("policy_entropy_2", float(policy_entropy_2))
            logger.record_tabular("value_loss_2", float(value_loss_2))
            logger.dump_tabular()
        if (update % (log_interval * 10)) == 0:
            logger.warn('Try to save cpkt file.')
            model.save(model_path)
            model2.save(model_path2)

            PolicyLossFile = "../statistics/policy_loss.csv"
            with open(PolicyLossFile, 'w') as f:
                writer = csv.writer(f, lineterminator='\n', delimiter=',')
                writer.writerow(float(val) for val in policy_loss_saver)

            ValueLossFile = "../statistics/value_loss.csv"
            with open(ValueLossFile, 'w') as f:
                writer = csv.writer(f, lineterminator='\n', delimiter=',')
                writer.writerow(float(val) for val in value_loss_saver)

            PolicyEntropyFile = "../statistics/policy_entropy_loss.csv"
            with open(PolicyEntropyFile, 'w') as f:
                writer = csv.writer(f, lineterminator='\n', delimiter=',')
                writer.writerow(float(val) for val in policy_entropy_saver)

            # Second model
            PolicyLossFile = "../statistics/policy_loss_2.csv"
            with open(PolicyLossFile, 'w') as f:
                writer = csv.writer(f, lineterminator='\n', delimiter=',')
                writer.writerow(float(val) for val in policy_loss_saver_2)

            ValueLossFile = "../statistics/value_loss_2.csv"
            with open(ValueLossFile, 'w') as f:
                writer = csv.writer(f, lineterminator='\n', delimiter=',')
                writer.writerow(float(val) for val in value_loss_saver_2)

            PolicyEntropyFile = "../statistics/policy_entropy_loss_2.csv"
            with open(PolicyEntropyFile, 'w') as f:
                writer = csv.writer(f, lineterminator='\n', delimiter=',')
                writer.writerow(float(val) for val in policy_entropy_saver_2)

    runner.mcts.visualization()
    env.close()


def play_with_policy(policy, policy2, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=0.5,
                     ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99,
                     gamma=0.99, log_interval=20, model_path='', model_path2=''):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes)  # HACK
    statistics_path = ('./stadistics_random')

    runner = Runner(env, None, None, nsteps=nsteps, nstack=nstack, gamma=gamma)
    runner.mcts.self_play_with_simple_policy()
    env.close()


def play(policy, policy2, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01,
         max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=20,
         model_path='', model_path2=''):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes)  # HACK
    statistics_path = ('./stadistics_random')
    summary_writer = tf.summary.FileWriter(statistics_path)

    # model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack,
    #                num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef, size=env.get_board_size(),
    #                max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
    #                lrschedule=lrschedule)
    model = Model2(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack,
                   num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
                   max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                   lrschedule=lrschedule, summary_writter=summary_writer)

    # model2 = Model(policy=policy2, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack,
    #                num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef, size=env.get_board_size(),
    #                max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
    #                lrschedule=lrschedule)
    model2 = Model2(policy=policy2, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack,
                    num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                    lrschedule=lrschedule, summary_writter=summary_writer)

    model.load(model_path)
    model2.load(model_path2)
    runner = Runner(env, model, model2, nsteps=nsteps, nstack=nstack, gamma=gamma)
    runner.mcts.start_play()
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
