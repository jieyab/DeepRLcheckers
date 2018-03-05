import time

import joblib
import numpy as np
import tensorflow as tf

from AlphaGomoku.common import logger
from AlphaGomoku.common.misc_util import set_global_seeds, explained_variance
from AlphaGomoku.core.utils_2 import Scheduler, find_trainable_variables
from AlphaGomoku.core.utils_2 import cat_entropy, mse
from AlphaGomoku.core.utils_2 import discount_with_dones
import datetime
import os
import csv


class Model(object):
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

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, scope="model",reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack,scope="model", reuse=True)
        #q = tf.one_hot(A, nact, dtype=tf.float32)
        #neglogpac = -tf.reduce_sum(tf.log((train_model.pi) + 1e-10) * q, [1])

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
        #trainer = tf.train.AdamOptimizer()
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

        self.batch = {}

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-1, axis=3)
        self.obs[:, :, :, -1] = obs[:, :, :, 0]

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

    def run(self, temp,expert):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_states = self.states
        self.obs = self.obs * 0
        for n in range(self.nsteps):
            counter = n
            actions, values, states, probs = self.model.step(self.obs, np.ones(1), [], [])
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
            obs, rewards, dones, _, illegal = self.env.step_smart(actions, expert)
            if illegal:
                counter = 0
                mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
                mb_obs.append(np.copy(self.obs))
                mb_actions.append(actions)
                mb_values.append(values)
                mb_dones.append(self.dones)
                # print(mb_obs, mb_rewards, mb_actions, mb_values, mb_dones)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n] * 0
            self.update_obs(obs)
            mb_rewards.append(rewards)
            if dones[0] or illegal:
                break
        mb_dones.append(self.dones)
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
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values

    def test(self, temp, expert):
        self.obs = self.obs * 0
        for n in range(self.nsteps):
            actions, values, states, probs = self.model.step(self.obs, np.ones(1), self.states, self.dones)

            a_dist = np.squeeze(probs)
            a_dist = np.clip(a_dist, 1e-20, 1)
            a_dist = self.softmax_b(a_dist, temp)
            a_dist = a_dist / np.sum(a_dist)
            actions = [np.argmax(a_dist)]
            # print(actions, self.env.get_illegal_moves())
            obs, rewards, dones, _, illegal = self.env.step_smart(actions, expert)

            # print(illegal, )
            self.obs = obs

            if dones[0] or illegal:
                break


    def put_in_batch(self, obs, states, reward, masks, actions, values):
        size = len(self.batch)

        number_slice = obs.shape[1]
        obs_slice =np.vsplit(obs,number_slice)
        reward_slice =np.hsplit(reward,number_slice)
        masks_slice =np.hsplit(masks,number_slice)
        actions_slice =np.hsplit(actions,number_slice)
        values_slice=np.hsplit(values,number_slice)

        for i in range(len(obs_slice)):
            self.batch.update({size + i: [np.asarray(obs_slice[i]), [], np.asarray(reward_slice[i]), np.asarray(masks_slice[i]),
                                          np.asarray(actions_slice[i]), np.asarray(values_slice[i])]})
            if (masks_slice[i][0] == True and masks_slice[i][1] == True):
                break

        return size

    def size_batch(self):
        return len(self.batch)

    def get_batch(self):
        return self.batch

    def empty_batch(self):
        self.batch.clear()


def save_csv(file, data):
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
    policy_loss, value_loss, policy_entropy = [], [], []
    size = obs.shape[1]

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


def learn(policy, env, seed, nsteps, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01,
          max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=1000,
          load_model=False, model_path='', data_augmentation=True, BATCH_SIZE=10,
          TEMP_CTE=30000, RUN_TEST=5000, expert=True):


    tf.reset_default_graph()
    set_global_seeds(seed)
    print('Data augmentation', data_augmentation)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes)  # HACK
    now = datetime.datetime.now()
    temp = np.ones(1)
    counter_stadistics = 0
    parameters = now.strftime("%d-%m-%Y_%H-%M-%S") + "_seed_" + str(
        seed) + "_BATCH_" + str(BATCH_SIZE) + "_TEMP_" + str(TEMP_CTE) + "_DA_" + str(data_augmentation) + "_VF_" + str(
        vf_coef) + str(policy) + str(np.sqrt(nsteps))+ 'x'+ str(np.sqrt(nsteps)) + 'expert' + str(expert)
    statistics_path = "../statistics/random/"
    BATCH_SIZE = np.sqrt(nsteps) * BATCH_SIZE
    model_path_load = model_path
    try:
        os.stat("../statistics/")
    except:
        os.mkdir("../statistics/")
    try:
        os.stat(statistics_path)
    except:
        os.mkdir(statistics_path)
    statistics_path = "../statistics/random/" + parameters
    model_path = statistics_path + "/model/"
    statistics_csv = statistics_path + "/csv/"

    games_wonAI_test_saver, games_finish_in_draw_test_saver, games_wonRandom_test_saver, illegal_test_games_test_saver = [], [], [], []
    games_wonAI_train_saver, games_finish_in_draw_train_saver, games_wonRandom_train_saver, illegal_test_games_train_saver = [], [], [], []
    update_test, update_train = [], []
    policy_entropy_saver, policy_loss_saver, explained_variance_saver, value_loss_saver, ev_saver = [], [], [], [], []

    try:
        os.stat(statistics_path)
    except:
        os.mkdir(statistics_path)

    try:
        os.stat(statistics_csv)
    except:
        os.mkdir(statistics_csv)

    try:
        os.stat(model_path)
    except:
        os.mkdir(model_path)

    summary_writer = tf.summary.FileWriter(statistics_path)
    temp = np.ones(1)

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=np.sqrt(nsteps), nstack=nstack,
                  num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                  lrschedule=lrschedule, summary_writter=summary_writer)

    if load_model:
        model.load(model_path_load)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs * nsteps
    tstart = time.time()

    for update in range(0, total_timesteps // nbatch + 1):
        if update % 1000 == 0:
            print('update: ', update)
            import threading
            env.print_stadistics(threading.get_ident())
        if (update % RUN_TEST < 1000) and (update % RUN_TEST > 0) and (update != 0):
            # print("Aqui")
            runner.test(np.ones(1), expert)
            temp = (0.8 * np.exp(-(update / TEMP_CTE)) + 0.2) * np.ones(1)


            if ((update % RUN_TEST) == 999):
                games_wonAI, games_wonRandom, games_finish_in_draw, illegal_games = env.get_stadistics()
                summary = tf.Summary()
                summary.value.add(tag='test/games_wonAI', simple_value=float(games_wonAI))
                summary.value.add(tag='test/games_wonRandom', simple_value=float(games_wonRandom))
                summary.value.add(tag='test/games_finish_in_draw', simple_value=float(games_finish_in_draw))
                summary.value.add(tag='test/illegal_games', simple_value=float(illegal_games))
                summary_writer.add_summary(summary, update)

                games_wonAI_test_saver.append(games_wonAI)
                games_wonRandom_test_saver.append(games_wonRandom)
                games_finish_in_draw_test_saver.append(games_finish_in_draw)
                illegal_test_games_test_saver.append(illegal_games)
                update_test.append(update)

                save_csv(statistics_csv + 'games_wonAI_test.csv', games_wonAI_test_saver)
                save_csv(statistics_csv + 'games_wonRandom_test.csv', games_wonRandom_test_saver)
                save_csv(statistics_csv + 'games_finish_in_draw_test.csv', games_finish_in_draw_test_saver)
                save_csv(statistics_csv + 'illegal_games_test.csv', illegal_test_games_test_saver)
                save_csv(statistics_csv + 'update_test.csv', update_test)

                summary_writer.flush()
        else:
            obs, states, rewards, masks, actions, values = runner.run(temp,expert)
            # print('obs',obs,'actions',actions)
            # print('values',values,'rewards',rewards,)

            obs, states, rewards, masks, actions, values = redimension_results(obs, states, rewards, masks, actions,
                                                                               values, env, nsteps)

            size_batch = runner.put_in_batch(obs, states, rewards, masks, actions, values)
            if size_batch >= BATCH_SIZE:
                # print('Training batch')
                batch = runner.get_batch()
                policy_loss_sv, value_loss_sv, policy_entropy_sv = [], [], []

                for i in range(len(batch)):
                    obs, states, rewards, masks, actions, values = batch.get(i)
                    if data_augmentation:
                        pl, vl, pe = train_data_augmentation(obs, states, rewards, masks,
                                                                                          actions,
                                                                                          values, model, temp)
                        policy_loss_sv.append(pl)
                        value_loss_sv.append(vl)
                        policy_entropy_sv.append(pe)
                    else:

                        pl, vl, pe= train_without_data_augmentation(obs, states, rewards,
                                                                                                  masks,
                                                                                                  actions, values,
                                                                                                  model,
                                                                                                  temp)
                        policy_loss_sv.append(pl)
                        value_loss_sv.append(vl)
                        policy_entropy_sv.append(pe)

                runner.empty_batch()
                policy_loss, value_loss, policy_entropy = np.mean(policy_loss_sv), np.mean(value_loss_sv), np.mean(policy_entropy_sv)
                # print('batch trained')
                nseconds = time.time() - tstart
                fps = int((update * nbatch) / nseconds)
                ev = explained_variance(values, rewards)

                counter_stadistics += 1
                if counter_stadistics % 10 == 0:
                    counter_stadistics = 0
                    logger.record_tabular("nupdates", update)
                    logger.record_tabular("total_timesteps", update * nbatch)
                    logger.record_tabular("fps", fps)
                    logger.record_tabular("policy_entropy", float(policy_entropy))
                    logger.record_tabular("policy_loss", float(policy_loss))
                    logger.record_tabular("value_loss", float(value_loss))
                    logger.record_tabular("explained_variance", float(ev))
                    #logger.dump_tabular()

                    games_wonAI, games_wonRandom, games_finish_in_draw, illegal_games = env.get_stadistics()

                    summary = tf.Summary()
                    summary.value.add(tag='train/policy_entropy', simple_value=float(policy_entropy))
                    summary.value.add(tag='train/policy_loss', simple_value=float(policy_loss))
                    summary.value.add(tag='train/explained_variance', simple_value=float(ev))
                    summary.value.add(tag='train/value_loss', simple_value=float(value_loss))
                    summary.value.add(tag='train/games_wonAI', simple_value=float(games_wonAI))
                    summary.value.add(tag='train/games_wonRandom', simple_value=float(games_wonRandom))
                    summary.value.add(tag='train/games_finish_in_draw', simple_value=float(games_finish_in_draw))
                    summary.value.add(tag='train/illegal_games', simple_value=float(illegal_games))
                    summary.value.add(tag='train/temp', simple_value=float(temp))
                    summary_writer.add_summary(summary, update)
                    summary_writer.flush()

                    games_wonAI_train_saver.append(games_wonAI)
                    games_wonRandom_train_saver.append(games_wonRandom)
                    games_finish_in_draw_train_saver.append(games_finish_in_draw)
                    illegal_test_games_train_saver.append(illegal_games)
                    update_train.append(update)

                    save_csv(statistics_csv + 'games_wonAI_train.csv', games_wonAI_train_saver)
                    save_csv(statistics_csv + 'games_wonRandom_train.csv', games_wonRandom_train_saver)
                    save_csv(statistics_csv + 'games_finish_in_draw_train.csv', games_finish_in_draw_train_saver)
                    save_csv(statistics_csv + 'illegal_games_train.csv', illegal_test_games_train_saver)
                    save_csv(statistics_csv + 'update_train.csv', update_train)

                    policy_entropy_saver.append(policy_entropy)
                    policy_loss_saver.append(policy_loss)
                    ev_saver.append(ev)
                    value_loss_saver.append(value_loss)

                    save_csv(statistics_csv + 'policy_entropy.csv', policy_entropy_saver)
                    save_csv(statistics_csv + 'policy_loss.csv', policy_loss_saver)
                    save_csv(statistics_csv + 'ev.csv', ev_saver)
                    save_csv(statistics_csv + 'value_loss.csv', value_loss_saver)

            if (update % (log_interval * 10)) == 0:
                print('Save check point')
                model.save(model_path +  parameters + '.cpkt')

    env.close()

if __name__ == '__main__':
    pass