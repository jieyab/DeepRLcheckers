import time
import time

import joblib
import numpy as np
import tensorflow as tf
import datetime
import os
import random
import csv

from AlphaGomoku.common import logger
from AlphaGomoku.common.misc_util import set_global_seeds, explained_variance
from AlphaGomoku.core.utils_2 import Scheduler
from AlphaGomoku.core.utils_2 import cat_entropy, mse
from AlphaGomoku.core.utils_2 import discount_with_dones

class Model(object):
    def __init__(self, policy,scope, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
                 ent_coef=0.1, vf_coef=0.5, max_grad_norm=0.5, l_rate=1e-4,
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

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, scope=scope, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, scope=scope, reuse=True)
        #q = tf.one_hot(A, nact, dtype=tf.float32)
        #neglogpac = -tf.reduce_sum(tf.log((train_model.pi) + 1e-10) * q, [1])

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        #trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=
        
        print(l_rate)
        trainer = tf.train.AdamOptimizer(l_rate)
        _train = trainer.apply_gradients(grads)
        lr = l_rate
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
        self.scope = scope


class Runner(object):
    def __init__(self, env, model, player, model_2, player_2, nsteps=5, nstack=4, gamma=0.7):
        self.env = env
        self.model = model
        self.model_2 = model_2

        self.player=player
        self.player_2 = player_2

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

    def softmax_b(self, x, temp):
        illegal_moves = self.env.get_illegal_moves()
        x = np.clip(x, 1e-20, 80.0)
        x = np.delete(x, illegal_moves)
        # print(x)
        x = x/temp
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

        for n in range(self.obs.shape[1] * self.obs.shape[1]):
            counter = n
            self.obs = self.obs_B
            obs_play = self.obs_B * -1
            _, values, states, probs = self.model.step(obs_play,np.ones(1), [], [])
            a_dist = np.squeeze(probs)
            a_dist = np.clip(a_dist, 1e-20, 1)
            a_dist = self.softmax_b(a_dist, temp)
            a_dist = a_dist / np.sum(a_dist)
            a = np.random.choice(a_dist, p=a_dist)
            actions = [np.argmax(a_dist == a)]

            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _, illegal = self.env.step_vs(actions, self.player)

            obs_play = obs * -1

            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n] * 0
            self.update_obs(self.obs)

            _, values_B, states_B, probs = self.model_2.step(obs_play, np.ones(1), [], [])
            a_dist = np.squeeze(probs)
            a_dist = np.clip(a_dist, 1e-20, 1)
            a_dist = self.softmax_b(a_dist, temp)
            a_dist = a_dist / np.sum(a_dist)
            a = np.random.choice(a_dist, p=a_dist)
            actions_B = [np.argmax(a_dist == a)]

            mb_obs_B.append(np.copy(self.obs))
            mb_actions_B.append(actions_B)
            mb_values_B.append(values_B)
            mb_dones_B.append(self.dones_B)
            self.obs_B, rewards_B, dones_B, _, illegal_B = self.env.step_vs(actions_B, self.player_2)

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
        elif rewards[0] <= 1:
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

    def test(self, temp, model, NUMBER_TEST, summary_writer, env, update):
        for i in range(NUMBER_TEST):
            self.obs = self.obs * 0
            for n in range(self.obs.shape[1]*self.obs.shape[1]):
                actions, values, states, probs = model.step(self.obs, temp, self.states, self.dones)

                a_dist = np.squeeze(probs)
                a_dist = np.clip(a_dist, 1e-20, 1)
                a_dist = self.get_legal_moves(a_dist)
                #a_dist = a_dist / np.sum(a_dist)
                actions = [np.argmax(a_dist)]
                # print(actions, self.env.get_illegal_moves())
                obs, rewards, dones, _, illegal = self.env.step(actions)

                # print(illegal, )
                self.obs = obs

                if dones[0] or illegal:
                    break

        won_AI, won_random, draws, illegal_games = env.get_stadistics()
        print('Result test random'+ model.scope)
        env.print_stadistics('AI_vs_AI mode')
        summary = tf.Summary()
        #summary.value.add(tag='test'+ self.model.scope+ '/won_AI' , simple_value=float(won_AI))
        summary.value.add(tag='test_random/won_' + model.scope, simple_value=float(won_random))
        #summary.value.add(tag='test/draws/'+ model.scope+ '/illegal_games', simple_value=float(illegal_games))
        summary_writer.add_summary(summary, update)
        summary_writer.flush()

        for i in range(NUMBER_TEST):
            self.obs = self.obs * 0
            for n in range(self.obs.shape[1]*self.obs.shape[1]):
                actions, values, states, probs = model.step(self.obs, temp, self.states, self.dones)

                a_dist = np.squeeze(probs)
                a_dist = np.clip(a_dist, 1e-20, 1)
                a_dist = self.get_legal_moves(a_dist)
                #a_dist = a_dist / np.sum(a_dist)
                actions = [np.argmax(a_dist)]
                # print(actions, self.env.get_illegal_moves())
                obs, rewards, dones, _, illegal = self.env.step_smart(actions,False)

                # print(illegal, )
                self.obs = obs

                if dones[0] or illegal:
                    break

        won_AI, won_random, draws, illegal_games = env.get_stadistics()
        print('Result test false expert '+ model.scope)
        env.print_stadistics('AI_vs_AI mode')
        summary = tf.Summary()
        #summary.value.add(tag='test'+ self.model.scope+ '/won_AI' , simple_value=float(won_AI))
        summary.value.add(tag='test_false_expert/won_' + model.scope, simple_value=float(won_random))
        #summary.value.add(tag='test/draws/'+ model.scope+ '/illegal_games', simple_value=float(illegal_games))
        summary_writer.add_summary(summary, update)
        summary_writer.flush()

        for i in range(NUMBER_TEST):
            self.obs = self.obs * 0
            for n in range(self.obs.shape[1]*self.obs.shape[1]):
                actions, values, states, probs = model.step(self.obs, temp, self.states, self.dones)

                a_dist = np.squeeze(probs)
                a_dist = np.clip(a_dist, 1e-20, 1)
                a_dist = self.get_legal_moves(a_dist)
                #a_dist = a_dist / np.sum(a_dist)
                actions = [np.argmax(a_dist)]
                # print(actions, self.env.get_illegal_moves())
                obs, rewards, dones, _, illegal = self.env.step_smart(actions, True)

                # print(illegal, )
                self.obs = obs

                if dones[0] or illegal:
                    break

        won_AI, won_random, draws, illegal_games = env.get_stadistics()
        print('Result test true True '+ model.scope)
        env.print_stadistics('AI_vs_AI mode')
        summary = tf.Summary()
        #summary.value.add(tag='test'+ self.model.scope+ '/won_AI' , simple_value=float(won_AI))
        summary.value.add(tag='test_true_expert/won_' + model.scope, simple_value=float(won_random))
        #summary.value.add(tag='test/draws/'+ model.scope+ '/illegal_games', simple_value=float(illegal_games))
        summary_writer.add_summary(summary, update)


        summary_writer.flush()


    def put_in_batch(self, obs, states, reward, masks, actions, values, obs_B, states_B, rewards_B, masks_B,
                     actions_B, values_B):
        size = len(self.batch)
        number_slice = obs.shape[1]

        obs_slice = np.vsplit(obs, number_slice)
        reward_slice = np.hsplit(reward, number_slice)
        masks_slice = np.hsplit(masks, number_slice)
        actions_slice = np.hsplit(actions, number_slice)
        values_slice = np.hsplit(values, number_slice)

        obs_slice_B = np.vsplit(obs_B, number_slice)
        reward_slice_B = np.hsplit(rewards_B, number_slice)
        masks_slice_B = np.hsplit(masks_B, number_slice)
        actions_slice_B = np.hsplit(actions_B, number_slice)
        values_slice_B = np.hsplit(values_B, number_slice)

        for i in range(len(obs_slice)):
            self.batch.update(
                {size + i: [np.asarray(obs_slice[i]), [], np.asarray(reward_slice[i]), np.asarray(masks_slice[i]),
                            np.asarray(actions_slice[i]), np.asarray(values_slice[i]),
                            np.asarray(obs_slice_B[i]), [], np.asarray(reward_slice_B[i]), np.asarray(masks_slice_B[i]),
                            np.asarray(actions_slice_B[i]), np.asarray(values_slice_B[i])
                            ]})
            if (masks_slice[i][0] == True and masks_slice[i][1] == True):
                break
        return size

    def size_batch(self):
        return len(self.batch)

    def get_batch(self):
        return self.batch

    def empty_batch(self):
        self.batch.clear()

    def save_csv(self, file, data):
        with open(file, 'w') as f:
            writer = csv.writer(f, lineterminator='\n', delimiter=',')
            writer.writerow(float(val) for val in data)


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
    return np.mean(policy_loss), np.mean(value_loss), np.mean(policy_entropy)


def train_without_data_augmentation(obs, states, rewards, masks, actions, values, model, temp):
    pl, vl, pe = model.train(obs, states, rewards, masks, actions, values, temp)
    return pl, vl, pe


def print_logger(update, policy_entropy, policy_loss, value_loss, policy_entropy_B, policy_loss_B, value_loss_B):
    #nbatch = 1
    #nseconds = time.time() - tstart
    #fps = int((update * nbatch) / nseconds)
    #ev = explained_variance(values, rewards)
    #ev_B = explained_variance(values_B, rewards_B)

    print('update:', update)
    logger.record_tabular("nupdates", update)
    #logger.record_tabular("fps", fps)

    logger.record_tabular("policy_entropy", float(policy_entropy))
    logger.record_tabular("policy_loss", float(policy_loss))
    logger.record_tabular("value_loss", float(value_loss))
    #logger.record_tabular("explained_variance", float(ev))

    logger.record_tabular("policy_entropy_B", float(policy_entropy_B))
    logger.record_tabular("policy_loss_B", float(policy_loss_B))
    logger.record_tabular("value_loss_B", float(value_loss_B))
    #logger.record_tabular("explained_variance_B", float(ev_B))
    logger.dump_tabular()

def print_tensorboard_training(summary_writer,update,policy_entropy, policy_loss, value_loss, policy_entropy_B,
                                      policy_loss_B, value_loss_B, temp):
    summary = tf.Summary()
    summary.value.add(tag='train_loss/policy_entropy', simple_value=float(policy_entropy))
    summary.value.add(tag='train_loss/policy_loss', simple_value=float(policy_loss))
    summary.value.add(tag='train_loss/value_loss', simple_value=float(value_loss))

    summary.value.add(tag='train_loss/policy_entropy_B', simple_value=float(policy_entropy_B))
    summary.value.add(tag='train_loss/policy_loss_B', simple_value=float(policy_loss_B))
    summary.value.add(tag='train_loss/value_loss_B', simple_value=float(value_loss_B))
    summary.value.add(tag='train/temp', simple_value=float(temp))

    summary_writer.add_summary(summary, update)

    summary_writer.flush()

def print_tensorboard_training_score(summary_writer,update,env):
    games_A, games_B, games_finish_in_draw, illegal_games = env.get_stadistics_vs()

    summary = tf.Summary()
    summary.value.add(tag='train/wan_A', simple_value=float(games_A))
    summary.value.add(tag='train/wan_B', simple_value=float(games_B))
    summary.value.add(tag='train/games_finish_in_draw', simple_value=float(games_finish_in_draw))
    summary_writer.add_summary(summary, update)

def create_models(NUMBER_OF_MODELS,policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs, ent_coef, vf_coef,
                           max_grad_norm, lr, alpha, epsilon, total_timesteps, lrschedule):
    models = []

    for i in range(0,NUMBER_OF_MODELS):

        model_name = 'model_holder_' + str(i)

        models.append(Model(policy=policy, scope=model_name, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=np.sqrt(nsteps), nstack=nstack,
                    num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, l_rate=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                    lrschedule=lrschedule))


    return models

def change_player(models,env):
    pl1, pl2 = random.sample(range(0, len(models)), 2)
    runner = Runner(env, models[pl1], 'A', models[pl2], 'B', nsteps=5, nstack=1, gamma=0.99)
    return runner, models[pl1], models[pl2]

def change_player_keep_one(models,env):
    aux = random.sample(range(1, len(models)), 1)
    pl1, pl2 = random.sample([[0],aux],2)
    runner = Runner(env, models[pl1[0]], 'A', models[pl2[0]], 'B', nsteps=5, nstack=1, gamma=0.99)
    return runner, models[pl1[0]], models[pl2[0]]

def save_csv(file, data):
    with open(file, 'w') as f:
        writer = csv.writer(f, lineterminator='\n', delimiter=',')
        writer.writerow(float(val) for val in data)

def train(temp, runner,model, model_2, data_augmentation,BATCH_SIZE,env,summary_writer, update, counter_stadistics,tstart,nsteps=5,):
    obs, states, rewards, masks, actions, values, obs_B, states_B, rewards_B, masks_B, actions_B, values_B = runner.run(
        temp)

    obs, states, rewards, masks, actions, values = redimension_results(obs, states, rewards, masks, actions,
                                                                       values, env, nsteps)
    obs_B, states_B, rewards_B, masks_B, actions_B, values_B = redimension_results(obs_B, states_B, rewards_B,
                                                                                   masks_B, actions_B, values_B,
                                                                                   env, nsteps)

    size_batch = runner.put_in_batch(obs, states, rewards, masks, actions, values, obs_B, states_B, rewards_B,
                                     masks_B, actions_B, values_B)

    if size_batch >= BATCH_SIZE:
        batch = runner.get_batch()
        policy_loss_sv, value_loss_sv, policy_entropy_sv = [], [], []
        policy_loss_sv_B, value_loss_sv_B, policy_entropy_sv_B = [], [], []
        for i in range(len(batch)):
            obs, states, rewards, masks, actions, values, obs_B, states_B, rewards_B, masks_B, actions_B, values_B = batch.get(
                i)


            if data_augmentation:
                pl, vl, pe = train_data_augmentation(obs, states, rewards, masks,
                                                                                  actions,
                                                                                  values, model, temp)
                pl_B, vl_B, pe_B = train_data_augmentation(obs_B, states_B,
                                                                                        rewards_B,
                                                                                        masks_B, actions_B,
                                                                                        values_B,
                                                                                        model_2, temp)

                policy_loss_sv.append(pl)
                value_loss_sv.append(vl)
                policy_entropy_sv.append(pe)

                policy_loss_sv_B.append(pl_B)
                value_loss_sv_B.append(vl_B)
                policy_entropy_sv_B.append(pe_B)
            else:
                pl, vl, pe = train_without_data_augmentation(obs, states, rewards,
                                                                                          masks,
                                                                                          actions,
                                                                                          values, model, temp)
                pl_B, vl_B, pe_B = train_without_data_augmentation(obs_B, states_B,
                                                                                                rewards_B,
                                                                                                masks_B,
                                                                                                actions_B,
                                                                                                values_B,
                                                                                                model_2, temp)
                policy_loss_sv.append(pl)
                value_loss_sv.append(vl)
                policy_entropy_sv.append(pe)

                policy_loss_sv_B.append(pl_B)
                value_loss_sv_B.append(vl_B)
                policy_entropy_sv_B.append(pe_B)

        runner.empty_batch()
        policy_loss, value_loss, policy_entropy = np.mean(policy_loss_sv), np.mean(value_loss_sv), np.mean(
            policy_entropy_sv)

        policy_loss_B, value_loss_B, policy_entropy_B = np.mean(policy_loss_sv_B), np.mean(value_loss_sv_B), np.mean(
            policy_entropy_sv_B)

        print_tensorboard_training(summary_writer, update, policy_entropy, policy_loss, value_loss,
                              policy_entropy_B, policy_loss_B, value_loss_B, temp)
        #        print_logger(update, policy_entropy, policy_loss, value_loss, policy_entropy_B, policy_loss_B, value_loss_B)







def learn(policy, env, seed, nsteps, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01,
          max_grad_norm=0.5, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=1000,
          load_model=False, model_path='', data_augmentation=True, BATCH_SIZE=100,NUMBER_OF_MODELS=3, CF=0.01,learning_rate=0.001):
    tf.reset_default_graph()
    set_global_seeds(seed)
    print('CF',CF)
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes)  # HACK
    now = datetime.datetime.now()

    CHANGE_PLAYER = 4000
    NUMBER_TEST = 1000
    TEMP_CTE = 10000
    counter_stadistics = 0
    temp = np.ones(1)

    parameters = now.strftime("%d-%m-%Y_%H-%M-%S") + "_seed_" + str(
        seed) + "_BATCH_" + str(BATCH_SIZE) + "_TEMP_" + str(TEMP_CTE) + "_DA_" + str(data_augmentation) + str(
        np.sqrt(nsteps)) + 'x' + str(np.sqrt(nsteps)) + '_num_players_' + str(NUMBER_OF_MODELS) + str(
        policy) + 'works' + '_CF_' + str(CF) + '_lr_' + str(learning_rate)
    statistics_path = ('../statistics/AI_vs_AI/' + parameters )

    models_path= statistics_path + '/model/'
    statistics_csv = statistics_path + "/csv/"


    summary_writer = tf.summary.FileWriter(statistics_path)

    models = create_models(NUMBER_OF_MODELS,policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs, ent_coef, vf_coef,
                           max_grad_norm, learning_rate, alpha, epsilon, total_timesteps, lrschedule)

    BATCH_SIZE = np.sqrt(nsteps) * BATCH_SIZE


    if load_model:
        # model_A.load('./models/model_A.cpkt')
        # model_B.load('./models/model_B.cpkt')
        print('Model loaded')

    runner, model, model_2 = change_player_keep_one(models, env)
    print('Loaded players', model.scope, 'A', model_2.scope, 'B')

    nbatch = nenvs * nsteps
    tstart = time.time()
    try:
        os.stat(statistics_path)
    except:
        os.mkdir(statistics_path)
    try:
        os.stat(models_path)
    except:
        os.mkdir(models_path)
    for update in range(0, total_timesteps // nbatch + 1):


        # if update % TEMP_COUNTER == 0:
        #     temp_count += 1
        #     temp = temp * (TEMP_CTE / (temp_count + TEMP_CTE)) + 0.2
        #     print('temp:', temp)

        if update % CHANGE_PLAYER == 0 and update != 0:
            env.print_stadistics_vs()
            print_tensorboard_training_score(summary_writer, update, env)
            temp = ((1-CF) * np.exp(-(update / TEMP_CTE)) + CF) * np.ones(1)
            print('Testing players, update:', update)
            runner.test(temp, model,NUMBER_TEST,summary_writer, env, update)
            runner.test(temp, model_2,NUMBER_TEST,summary_writer, env, update)

            runner, model, model_2 = change_player_keep_one(models,env)
            print('Change players, new players', model.scope, 'A',model_2.scope,'B')


        else:
            train(temp, runner, model, model_2, data_augmentation, BATCH_SIZE, env, summary_writer, update,
                  counter_stadistics, tstart, nsteps=nsteps)


        if (update % (log_interval * 10)) == 0 and update != 0:
             print('Save check point')
             for mod in models:
                 mod.save(models_path + parameters + '_' + str(mod.scope) )


if __name__ == '__main__':
    pass
