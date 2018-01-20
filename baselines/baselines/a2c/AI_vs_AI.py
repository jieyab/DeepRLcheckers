import time
import time

import joblib
import numpy as np
import tensorflow as tf

from baselines import logger
from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.utils import cat_entropy, mse
from baselines.a2c.utils import discount_with_dones
from baselines.common import set_global_seeds, explained_variance


class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
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


class Runner(object):

    def __init__(self, env, model_A, model_B, nsteps=5, nstack=4, gamma=0.7):
        self.env = env
        self.model_A = model_A
        self.model_B = model_B
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.nenv = env.num_envs
        self.nh = nh
        self.nw = nw
        self.nc = nc
        self.nstack = nstack
        self.batch_ob_shape = (nenv * nsteps, nh, nw, nc * nstack)
        self.obs = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.float32)
        self.obs_B = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.float32)
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model_A.initial_state
        self.states_B = model_B.initial_state
        self.dones = [False for _ in range(nenv)]
        self.dones_B = [False for _ in range(nenv)]


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

    def get_legal_moves(self,actions):
        illegal_moves = self.env.get_illegal_moves()
        moves_nn = actions[0].tolist()
        for i in illegal_moves:
            if i in moves_nn:
                moves_nn.remove(i)
        actions = [np.asarray(moves_nn[0])]
        return actions

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_obs_B, mb_rewards_B, mb_actions_B, mb_values_B, mb_dones_B = [],[],[],[],[]
        mb_states = self.states
        mb_states_B = self.states_B
        self.obs = self.obs*0
        self.obs_B = self.obs*0

        for n in range(self.nsteps):
            counter = n
            self.obs = self.obs_B
            obs_play = self.obs_B *-1
            actions, values, states, prob_A = self.model_A.step(obs_play, [], [])
            save_actions_A = actions
            actions = [np.asarray(actions[0][0])]
            #actions = [actions[0]]
            #actions = self.get_legal_moves(actions)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            self.obs, rewards, dones, _, illegal = self.env.step_vs(actions, 'A')
            #Vs AI

            obs_play = self.obs * -1

            actions_B, values_B, states_B , prob_B= self.model_B.step(obs_play, [], [])
            save_actions_B = actions_B
            actions_B = [np.asarray(actions_B[0][0])]
            #actions_B = [actions_B[0]]
            #actions_B = self.get_legal_moves(actions_B)
            mb_obs_B.append(np.copy(self.obs))
            mb_actions_B.append(actions_B)
            mb_values_B.append(values_B)
            mb_dones_B.append(self.dones_B)
            self.obs_B, rewards_B, dones_B, _, illegal_B = self.env.step_vs(actions_B, 'B')

            if illegal or illegal_B:
                counter = 0
                mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
                mb_obs.append(np.copy(self.obs))
                mb_actions.append(actions)
                mb_values.append(values)
                mb_dones.append(self.dones)

                counter = 0
                mb_obs_B, mb_rewards_B, mb_actions_B, mb_values_B, mb_dones_B = [], [], [], [], []
                mb_obs_B.append(np.copy(self.obs_B))
                mb_actions_B.append(actions_B)
                mb_values_B.append(values_B)
                mb_dones_B.append(self.dones_B)

            self.obs_B = self.obs_B*-1
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n] * 0
            self.update_obs(self.obs)

            self.states_B = states_B
            self.dones_B = dones_B
            for n, done in enumerate(dones_B):
                if done:
                    self.obs_B[n] = self.obs_B[n] * 0
            self.update_obs_B(self.obs_B)

            if rewards == 1:
                #mb_dones_B = mb_dones
                rewards_B = [-0.5]
            elif rewards_B == 1:
                #mb_dones = mb_dones_B
                rewards = [-0.5]
            elif (rewards_B == 0.5):
                #mb_dones = mb_dones_B
                rewards_B = [0.5]
                rewards = [0.5]
                self.dones = np.ones((1, 1), dtype=bool)
                self.dones_B = np.ones((1, 1), dtype=bool)
            elif (rewards == 0.5):
                #mb_dones_B = mb_dones
                rewards_B = [0.5]
                rewards = [0.5]
                self.dones = np.ones((1, 1), dtype=bool)
                self.dones_B = np.ones((1, 1), dtype=bool)

            mb_rewards.append(rewards)
            mb_rewards_B.append(rewards_B)
            if rewards != 0 or rewards_B != 0 or illegal_B or illegal:
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
        last_values = self.model_A.value(self.obs, self.states, self.dones).tolist()

        # batch of steps to batch of rollouts _B

        mb_obs_B = np.asarray(mb_obs_B, dtype=np.float32).swapaxes(1, 0).reshape(
            (self.nenv * (counter + 1), self.nh, self.nw, self.nc * self.nstack))
        mb_rewards_B = np.asarray(mb_rewards_B, dtype=np.float32).swapaxes(1, 0)
        mb_actions_B = np.asarray(mb_actions_B, dtype=np.int32).swapaxes(1, 0)
        mb_values_B = np.asarray(mb_values_B, dtype=np.float32).swapaxes(1, 0)
        mb_dones_B = np.asarray(mb_dones_B, dtype=np.bool).swapaxes(1, 0)
        mb_masks_B = mb_dones_B[:, :-1]
        mb_dones_B = mb_dones_B[:, 1:]
        last_values_B = self.model_B.value(self.obs_B, self.states_B, self.dones_B).tolist()

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

        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, mb_obs_B, mb_states_B, mb_rewards_B, mb_masks_B, mb_actions_B, mb_values_B ,illegal, illegal_B


def learn(policy,policy_b, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01,
          max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=1000,
          load_model=False, model_path=''):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.dimensions()
    num_procs = len(env.remotes)  # HACK

    model_A = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack,
                    num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                    lrschedule=lrschedule)
    model_B = Model(policy=policy_b, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps,
                    nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon,
                    total_timesteps=total_timesteps, lrschedule=lrschedule)

    if load_model:
        #model_A.load('./models/model_A.cpkt')
        #model_B.load('./models/model_B.cpkt')
        print('Model loaded')

    runner = Runner(env, model_A, model_B, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs * nsteps
    tstart = time.time()
    for update in range(1, total_timesteps // nbatch + 1):
        obs, states, rewards, masks, actions, values, obs_B, states_B, rewards_B, masks_B, actions_B, values_B, illegal, illegal_B= runner.run()
        # print('obs',obs,'actions',actions)
        # print('values',values,'rewards',rewards,)

        if update % 1000000 == 0:
            aux = model_A
            model_A = model_B
            model_B = aux
            runner = Runner(env, model_A, model_B, nsteps=nsteps, nstack=nstack, gamma=gamma)
        if illegal_B == False:
            dim_total = nsteps
            dim = obs.shape[0]
            dim_necesaria = dim_total - dim
            obs = np.concatenate((obs, np.zeros((dim_necesaria, env.dimensions(), env.dimensions(), 1))), axis=0)
            rewards = np.concatenate((rewards, np.zeros((dim_necesaria))), axis=0)
            masks = np.concatenate((masks, np.full((dim_necesaria), True, dtype=bool)), axis=0)
            actions = np.concatenate((actions, np.zeros(dim_necesaria)), axis=0)
            values = np.concatenate((values, np.zeros(dim_necesaria)), axis=0)

            policy_loss, value_loss, policy_entropy = model_A.train(obs, states, rewards, masks, actions, values)
        if illegal == False:
            dim_total = nsteps
            dim = obs_B.shape[0]
            dim_necesaria = dim_total - dim
            obs_B = np.concatenate((obs_B, np.zeros((dim_necesaria, env.dimensions(), env.dimensions(), 1))), axis=0)
            rewards_B = np.concatenate((rewards_B, np.zeros((dim_necesaria))), axis=0)
            masks_B = np.concatenate((masks_B, np.full((dim_necesaria), True, dtype=bool)), axis=0)
            actions_B = np.concatenate((actions_B, np.zeros(dim_necesaria)), axis=0)
            values_B = np.concatenate((values_B, np.zeros(dim_necesaria)), axis=0)

            policy_loss_B, value_loss_B, policy_entropy_B = model_B.train(obs_B, states_B, rewards_B, masks_B, actions_B, values_B)


        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0:
            ev = explained_variance(values, rewards)
            ev_B = explained_variance(values_B, rewards_B)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))

            logger.record_tabular("policy_entropy_B", float(policy_entropy_B))
            logger.record_tabular("value_loss_B", float(value_loss_B))
            logger.record_tabular("explained_variance_B", float(ev_B))
            logger.dump_tabular()
        if (update % (log_interval * 1)) == 0:
            model_A.save('./models/model_A.cpkt')
            model_B.save('./models/model_B.cpkt')
            print('Saved model')


    env.close()
