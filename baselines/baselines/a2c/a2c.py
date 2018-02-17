import os.path as osp
import gym
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.policies import CnnPolicy
from baselines.a2c.utils import cat_entropy, mse

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
            ent_coef=0.1, vf_coef=0.5, max_grad_norm=0.05, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nact = ac_space
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])


        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)
        q = tf.one_hot(A, 9, dtype=tf.float32)
        neglogpac = -tf.reduce_sum(tf.log((train_model.pi) + 1e-10) * q, [1])

        #neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

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
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
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
            #make_path(save_path)
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
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc*nstack)
        self.obs = np.zeros((nenv, nh, nw, nc*nstack), dtype=np.float32)
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

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


    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        self.obs = self.obs * 0
        for n in range(self.nsteps):
            counter = n
            actions, values, states,probs = self.model.step(self.obs, [], [])
            probs = np.squeeze(probs)
            a_dist = self.softmax_b(probs)
            # print(probs)
            # probs = self.get_legal_moves(probs)
            # print(a_dist)
            # print('+-+-+-+-+-+-+-+-+-+-+-+-+-')
            # a_dist = probs / np.sum(probs)
            a = np.random.choice(a_dist, p=a_dist)
            actions = [np.argmax(a_dist == a)]

            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _, illegal = self.env.step(actions)
            if illegal:
                counter = 0
                mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
                mb_obs.append(np.copy(self.obs))
                mb_actions.append(actions)
                mb_values.append(values)
                mb_dones.append(self.dones)
                #print(mb_obs, mb_rewards, mb_actions, mb_values, mb_dones)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.update_obs(obs)
            mb_rewards.append(rewards)
            if dones[0] or illegal:
                break
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0).reshape((self.nenv*(counter+1), self.nh, self.nw, self.nc*self.nstack))
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values


    def test(self):
        self.obs = self.obs * 0
        for n in range(self.nsteps):
            actions, values, states, probs = self.model.step(self.obs, self.states, self.dones)

            probs = np.squeeze(probs)
            a_dist = self.softmax(probs)
            #print(probs)
            #print(a_dist)
            a_dist = self.get_legal_moves(a_dist)
            #print(a_dist)
            # print('+-+-+-+-+-+-+-+-+-+-+-+-+-')
            #a_dist = probs / np.sum(probs)
            #a = np.random.choice(a_dist, p=a_dist)
            actions = [np.argmax(a_dist)]
            #print(actions, self.env.get_illegal_moves())
            obs, rewards, dones, _, illegal = self.env.step(actions)

            #print(illegal, )
            self.obs = obs

            if dones[0] or illegal:
                break

def learn(policy, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=1000, load_model=False,model_path=''):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes) # HACK
    statistics_path = ('./stadistics')
    summary_writer = tf.summary.FileWriter(statistics_path)
    run_test = 5000
    policy_entropy = 10

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)

    if load_model:
        model.load(model_path)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs*nsteps
    tstart = time.time()
    for update in range(0, total_timesteps//nbatch+1):
        if update % 1000 == 0:
            print('update', update)
            env.print_stadistics()
        if (update % run_test < 1000)  and (update % run_test > 0):
            #print("Aqui")
            runner.test()
            games_wonAI, games_wonRandom, games_finish_in_draw, illegal_games = env.get_stadistics()
            if ((update % run_test) == 999):
                summary = tf.Summary()
                summary.value.add(tag='test/games_wonAI', simple_value=float(games_wonAI))
                summary.value.add(tag='test/games_wonRandom', simple_value=float(games_wonRandom))
                summary.value.add(tag='test/games_finish_in_draw', simple_value=float(games_finish_in_draw))
                summary.value.add(tag='test/illegal_games', simple_value=float(illegal_games))
                summary_writer.add_summary(summary, update)

                summary_writer.flush()
        else:
            obs, states, rewards, masks, actions, values = runner.run()
            # print('obs',obs,'actions',actions)
            # print('values',values,'rewards',rewards,)

            dim_total = nsteps
            dim = obs.shape[0]
            dim_necesaria = dim_total - dim
            obs = np.concatenate((obs, np.zeros((dim_necesaria, env.dimensions(), env.dimensions(), 1))), axis=0)
            rewards = np.concatenate((rewards, np.zeros((dim_necesaria))), axis=0)
            masks = np.concatenate((masks, np.full((dim_necesaria), True, dtype=bool)), axis=0)
            actions = np.concatenate((actions, np.zeros(dim_necesaria)), axis=0)
            values = np.concatenate((values, np.zeros(dim_necesaria)), axis=0)

            policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)

            nseconds = time.time() - tstart
            fps = int((update * nbatch) / nseconds)
            if update % log_interval == 0 or update == 1:
                ev = explained_variance(values, rewards)
                logger.record_tabular("nupdates", update)
                logger.record_tabular("total_timesteps", update * nbatch)
                logger.record_tabular("fps", fps)
                logger.record_tabular("policy_entropy", float(policy_entropy))
                logger.record_tabular("policy_loss", float(policy_loss))
                logger.record_tabular("value_loss", float(value_loss))
                logger.record_tabular("explained_variance", float(ev))
                logger.dump_tabular()

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
                summary_writer.add_summary(summary, update)
                summary_writer.flush()

            if (update % (log_interval * 10)) == 0:
                #model.save('./models/tic_tac_toe.cpkt')
                pass

    env.close()

if __name__ == '__main__':
    main()
