import time
import time

import joblib
import numpy as np
import tensorflow as tf

from AlphaGomoku.common import logger
from AlphaGomoku.common.misc_util import set_global_seeds, explained_variance
from AlphaGomoku.core.utils_2 import Scheduler
from AlphaGomoku.core.utils_2 import cat_entropy, mse
from AlphaGomoku.core.utils_2 import discount_with_dones


class Model(object):
    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
                 ent_coef=0.1, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):
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

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model')
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
        self.player = 'A'


class Model_2(object):
    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
                 ent_coef=0.1, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):
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

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model_2')
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
        self.player = 'B'


class Runner(object):
    def __init__(self, env, model, model_2, nsteps=5, nstack=4, gamma=0.7):
        self.env = env
        self.model = model
        self.model_2 = model_2

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
        self.dones_B = [False for _ in range(nenv)]

        self.batch = {}

        self.reward_winning = self.env.reward_winning
        self.reward_lossing = self.env.reward_lossing
        self.reward_illegal_move = self.env.reward_illegal_move
        self.reward_draw = self.env.reward_draw

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-1, axis=3)
        self.obs[:, :, :, -1] = obs[:, :, :, 0]

    def update_obs_B(self, obs_B):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs_B = np.roll(self.obs_B, shift=-1, axis=3)
        self.obs_B[:, :, :, -1] = obs_B[:, :, :, 0]

    def get_legal_moves(self, probs):
        illegal_moves = self.env.get_illegal_moves()
        for i in illegal_moves:
            probs[i] = 0
        return probs

    def softmax_b(self, x):
        illegal_moves = self.env.get_illegal_moves()
        x = np.clip(x, 1e-20, 80.0)
        x = np.delete(x, illegal_moves)
        # print(x)
        x = np.exp(x) / np.sum(np.exp(x), axis=0)
        for i in range(len(illegal_moves)):
            x = np.insert(x, illegal_moves[i], 0)
        return x

    def softmax(self, x):
        x = np.clip(x, 1e-20, 80.0)
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def run(self, temp):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_obs_B, mb_rewards_B, mb_actions_B, mb_values_B, mb_dones_B = [], [], [], [], []

        mb_states = self.states
        self.obs = self.obs * 0
        self.obs_B = self.obs * 0

        for n in range(self.nsteps):
            counter = n
            self.obs = self.obs_B
            obs_play = self.obs_B * -1
            _, values, states, probs = self.model.step(obs_play, temp, [], [])
            a_dist = np.squeeze(probs)
            a_dist = np.clip(a_dist, 1e-20, 1)
            a_dist = self.get_legal_moves(a_dist)
            a_dist = a_dist / np.sum(a_dist)
            a = np.random.choice(a_dist, p=a_dist)
            actions = [np.argmax(a_dist == a)]

            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _, illegal = self.env.step_vs(actions, self.model_2.player)

            obs_play = obs * -1

            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n] * 0
            self.update_obs(self.obs)

            _, values_B, states_B, probs = self.model_2.step(obs_play, temp, [], [])
            a_dist = np.squeeze(probs)
            a_dist = np.clip(a_dist, 1e-20, 1)
            a_dist = self.get_legal_moves(a_dist)
            a_dist = a_dist / np.sum(a_dist)
            a = np.random.choice(a_dist, p=a_dist)
            actions_B = [np.argmax(a_dist == a)]

            mb_obs_B.append(np.copy(self.obs))
            mb_actions_B.append(actions_B)
            mb_values_B.append(values_B)
            mb_dones_B.append(self.dones_B)
            self.obs_B, rewards_B, dones_B, _, illegal_B = self.env.step_vs(actions_B, self.model.player)

            self.obs_B = self.obs_B * -1
            self.states = states
            self.dones = dones

            self.states_B = states_B
            self.dones_B = dones_B

            for n, done in enumerate(dones_B):
                if done:
                    self.obs_B[n] = self.obs_B[n] * 0
            self.update_obs_B(self.obs_B)

            if rewards == self.reward_winning:
                # mb_dones_B = mb_dones
                rewards_B = [self.reward_lossing]

            elif rewards_B == self.reward_winning:
                # mb_dones = mb_dones_B
                rewards = [self.reward_lossing]

            elif (rewards_B == self.reward_draw):
                # mb_dones = mb_dones_B
                rewards_B = [self.reward_draw]
                rewards = [self.reward_draw]
                self.dones = np.ones((1, 1), dtype=bool)
                self.dones_B = np.ones((1, 1), dtype=bool)

            elif (rewards == self.reward_draw):
                # mb_dones_B = mb_dones
                rewards_B = [self.reward_draw]
                rewards = [self.reward_draw]
                self.dones = np.ones((1, 1), dtype=bool)
                self.dones_B = np.ones((1, 1), dtype=bool)

            mb_rewards.append(rewards)
            mb_rewards_B.append(rewards_B)

            if rewards != 0 or rewards_B != 0:
                break

        mb_dones.append(self.dones)
        mb_dones_B.append(self.dones_B)

        if rewards[0] >= 1:
            mb_dones_B = mb_dones
        elif rewards_B[0] <= 1:
            mb_dones = mb_dones_B

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0).reshape(
            (self.nenv * (counter + 1), self.nh, self.nw, self.nc * self.nstack))
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, temp, [], []).tolist()

        # batch of steps to batch of rollouts _B
        mb_obs_B = np.asarray(mb_obs_B, dtype=np.float32).swapaxes(1, 0).reshape(
            (self.nenv * (counter + 1), self.nh, self.nw, self.nc * self.nstack))
        mb_rewards_B = np.asarray(mb_rewards_B, dtype=np.float32).swapaxes(1, 0)
        mb_actions_B = np.asarray(mb_actions_B, dtype=np.int32).swapaxes(1, 0)
        mb_values_B = np.asarray(mb_values_B, dtype=np.float32).swapaxes(1, 0)
        mb_dones_B = np.asarray(mb_dones_B, dtype=np.bool).swapaxes(1, 0)
        mb_masks_B = mb_dones_B[:, :-1]
        mb_dones_B = mb_dones_B[:, 1:]
        last_values_B = self.model_2.value(self.obs_B, temp, [], []).tolist()

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

        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards_B, mb_dones_B, last_values_B)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            # print('Player A',mb_rewards, mb_dones, last_values)
            mb_rewards_B[n] = rewards
        mb_rewards_B = mb_rewards_B.flatten()
        mb_actions_B = mb_actions_B.flatten()
        mb_values_B = mb_values_B.flatten()
        mb_masks_B = mb_masks_B.flatten()

        return mb_obs, [], mb_rewards, mb_masks, mb_actions, mb_values, mb_obs_B, [], mb_rewards_B, mb_masks_B, mb_actions_B, mb_values_B

    def test(self, temp):
        self.obs = self.obs * 0
        for n in range(self.nsteps):
            self.obs = self.obs_B
            obs_play = self.obs_B * -1
            actions, values, states, probs = self.model.step(obs_play, temp, [], [])
            a_dist = np.squeeze(probs)
            a_dist = np.clip(a_dist, 1e-20, 1)
            a_dist = self.get_legal_moves(a_dist)
            a_dist = a_dist / np.sum(a_dist)

            actions = [np.argmax(a_dist)]

            obs, rewards, dones, _, illegal = self.env.step_vs(actions, 'A')

            if rewards != 0:
                break

            obs_play = obs * -1

            actions_B, values_B, states_B, probs = self.model_2.step(obs_play, temp, [], [])

            a_dist = np.squeeze(probs)
            a_dist = np.clip(a_dist, 1e-20, 1)
            a_dist = self.get_legal_moves(a_dist)
            a_dist = a_dist / np.sum(a_dist)
            actions_B = [np.argmax(a_dist)]

            self.obs_B, rewards_B, dones_B, _, illegal_B = self.env.step_vs(actions_B, 'B')

            if rewards_B != 0:
                break

    def put_in_batch(self, obs, states, reward, masks, actions, values, obs_B, states_B, rewards_B, masks_B,
                     actions_B, values_B):

        size = len(self.batch)
        self.batch.update({size: [obs, states, reward, masks, actions, values, obs_B, states_B, rewards_B, masks_B,
                                  actions_B, values_B]})
        return size

    def size_batch(self):
        return len(self.batch)

    def get_batch(self):
        return self.batch

    def empty_batch(self):
        self.batch.clear()


def redimension_results(obs, states, rewards, masks, actions, values, env, nsteps):
    dim_total = nsteps
    dim = obs.shape[0]
    dim_necesaria = dim_total - dim
    obs = np.concatenate((obs, np.zeros((dim_necesaria, env.dimensions(), env.dimensions(), 1))), axis=0)
    rewards = np.concatenate((rewards, np.zeros((dim_necesaria))), axis=0)
    masks = np.concatenate((masks, np.full((dim_necesaria), True, dtype=bool)), axis=0)
    actions = np.concatenate((actions, np.zeros(dim_necesaria)), axis=0)
    values = np.concatenate((values, np.zeros(dim_necesaria)), axis=0)

    return obs, states, rewards, masks, actions, values


def train_data_augmentation(obs, states, rewards, masks, actions, values, model, temp):
    size = obs.shape[1]
    policy_loss, value_loss, policy_entropy = [], [], []
    actions_in_board = np.array([np.zeros((len(obs[0]), len(obs[0]))) for _ in range(len(actions))])
    for i in range(len(actions)):
        actions_in_board[i, int(actions[i] % len(obs[0])), int(actions[i] / len(obs[0]))] = 1
    new_actions = np.array([0 for _ in range(len(actions))])
    # print(actions_in_board)

    for i in [1, 2, 3, 4]:
        # rotate counterclockwise
        rot_obs = np.array([np.rot90(s, i) for s in obs])
        rot_actions = np.array([np.rot90(s, i) for s in actions_in_board])
        new_actions.fill(0)
        for i in range(len(actions)):
            import itertools
            for x, y in itertools.product(range(len(obs[0])), range(len(obs[0]))):
                if rot_actions[i, x, y] == 1:
                    new_actions[i] = size * y + x

        pl, vl, pe = model.train(rot_obs, states, rewards, masks, new_actions,
                                 values, temp)
        policy_loss.append(pl)
        value_loss.append(vl)
        policy_entropy.append(pe)

        flip_obs = np.array([np.fliplr(s) for s in rot_obs])
        flip_actions = np.array([np.fliplr(s) for s in rot_actions])
        new_actions.fill(0)
        for i in range(len(actions)):
            import itertools
            for x, y in itertools.product(range(len(obs[0])), range(len(obs[0]))):
                if flip_actions[i, x, y] == 1:
                    new_actions[i] = size * y + x

        pl, vl, pe = model.train(flip_obs, states, rewards, masks, new_actions,
                                 values, temp)

        policy_loss.append(pl)
        value_loss.append(vl)
        policy_entropy.append(pe)

    return np.mean(pl), np.mean(vl), np.mean(pe)


def train_without_data_augmentation(obs, states, rewards, masks, actions, values, model, temp):
    pl, vl, pe = model.train(obs, states, rewards, masks, actions, values, temp)
    return pl, vl, pe


def learn(policy, policy_2, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01,
          max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=1000,
          load_model=False, model_path='', data_augmentation=True, TRAINING_BATCH=10):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes)  # HACK
    statistics_path = ('./stadistics')
    summary_writer = tf.summary.FileWriter(statistics_path)
    run_test = 2000
    temp = np.ones(1)
    CHANGE_PLAYER = 100

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack,
                  num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                  lrschedule=lrschedule)

    model_2 = Model_2(policy=policy_2, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps,
                      nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
                      max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon,
                      total_timesteps=total_timesteps, lrschedule=lrschedule)

    if load_model:
        # model_A.load('./models/model_A.cpkt')
        # model_B.load('./models/model_B.cpkt')
        print('Model loaded')

    runner = Runner(env, model, model_2, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs * nsteps
    tstart = time.time()
    for update in range(0, total_timesteps // nbatch + 1):
        if update % 999 == 0:
            print('update', update)
            env.print_stadistics_vs()
            games_A, games_B, games_finish_in_draw, illegal_games = env.get_stadistics_vs()

        if (update % run_test < 1000) and (update % run_test > 0):
            runner.test(temp)

            if ((update % run_test) == 999):
                summary = tf.Summary()
                summary.value.add(tag='test/games_A', simple_value=float(games_A))
                summary.value.add(tag='test/games_B', simple_value=float(games_B))
                summary.value.add(tag='test/games_finish_in_draw', simple_value=float(games_finish_in_draw))
                summary.value.add(tag='test/illegal_games', simple_value=float(illegal_games))
                summary_writer.add_summary(summary, update)

                summary_writer.flush()
        else:

            if update % CHANGE_PLAYER == 0 and update != 0:
                aux = model_2
                model_2 = model
                model = aux
                runner = Runner(env, model, model_2, nsteps=nsteps, nstack=nstack, gamma=gamma)

            obs, states, rewards, masks, actions, values, obs_B, states_B, rewards_B, masks_B, actions_B, values_B = runner.run(
                temp)

            obs, states, rewards, masks, actions, values = redimension_results(obs, states, rewards, masks, actions,
                                                                               values, env, nsteps)
            obs_B, states_B, rewards_B, masks_B, actions_B, values_B = redimension_results(obs_B, states_B, rewards_B,
                                                                                           masks_B, actions_B, values_B,
                                                                                           env, nsteps)

            size_batch = runner.put_in_batch(obs, states, rewards, masks, actions, values, obs_B, states_B, rewards_B,
                                             masks_B, actions_B, values_B)

            if size_batch == TRAINING_BATCH:

                batch = runner.get_batch()
                for i in range(len(batch)):
                    obs, states, rewards, masks, actions, values, obs_B, states_B, rewards_B, masks_B, actions_B, values_B = batch.get(
                        i)
                    if data_augmentation:
                        policy_loss, value_loss, policy_entropy = train_data_augmentation(obs, states, rewards, masks,
                                                                                          actions,
                                                                                          values, model, temp)
                        policy_loss_B, value_loss_B, policy_entropy_B = train_data_augmentation(obs_B, states_B,
                                                                                                rewards_B,
                                                                                                masks_B, actions_B,
                                                                                                values_B,
                                                                                                model_2, temp)
                    else:
                        policy_loss, value_loss, policy_entropy = train_without_data_augmentation(obs, states, rewards,
                                                                                                  masks,
                                                                                                  actions,
                                                                                                  values, model, temp)
                        policy_loss_B, value_loss_B, policy_entropy_B = train_without_data_augmentation(obs_B, states_B,
                                                                                                        rewards_B,
                                                                                                        masks_B,
                                                                                                        actions_B,
                                                                                                        values_B,
                                                                                                        model_2, temp)
                runner.empty_batch()

                nseconds = time.time() - tstart
                fps = int((update * nbatch) / nseconds)
                ev = explained_variance(values, rewards)
                ev_B = explained_variance(values_B, rewards_B)
                if update % (TRAINING_BATCH * 100) == 0:
                    print('update:', update)
                    logger.record_tabular("nupdates", update)
                    logger.record_tabular("total_timesteps", update * nbatch)
                    logger.record_tabular("fps", fps)

                    logger.record_tabular("policy_entropy", float(policy_entropy))
                    logger.record_tabular("policy_loss", float(policy_loss))
                    logger.record_tabular("value_loss", float(value_loss))
                    logger.record_tabular("explained_variance", float(ev))

                    logger.record_tabular("policy_entropy_B", float(policy_entropy_B))
                    logger.record_tabular("policy_loss_B", float(policy_loss_B))
                    logger.record_tabular("value_loss_B", float(value_loss_B))
                    logger.record_tabular("explained_variance_B", float(ev_B))
                    logger.dump_tabular()

                summary = tf.Summary()
                summary.value.add(tag='train_A/policy_entropy', simple_value=float(policy_entropy))
                summary.value.add(tag='train_loss/policy_loss_A', simple_value=float(policy_loss))
                summary.value.add(tag='train_A/explained_variance', simple_value=float(ev))
                summary.value.add(tag='train_loss/value_loss_A', simple_value=float(value_loss))

                summary.value.add(tag='train_B/policy_entropy', simple_value=float(policy_entropy_B))
                summary.value.add(tag='train_loss/policy_loss', simple_value=float(policy_loss_B))
                summary.value.add(tag='train_B/explained_variance_B', simple_value=float(ev_B))
                summary.value.add(tag='train_loss/value_loss_B', simple_value=float(value_loss_B))

                summary.value.add(tag='train/wan_A', simple_value=float(games_A))
                summary.value.add(tag='train/wan_B', simple_value=float(games_B))
                summary.value.add(tag='train/games_finish_in_draw', simple_value=float(games_finish_in_draw))
                summary_writer.add_summary(summary, update)

                summary_writer.flush()

            if (update % (log_interval * 1)) == 0:
                model.save('../models/tic_tac_toe.cpkt')


if __name__ == '__main__':
    pass
