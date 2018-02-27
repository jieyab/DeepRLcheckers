# !/usr/bin/env python

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import threading
import time
import configparser
import sys

from AlphaGomoku.core.a2c_2 import learn
from AlphaGomoku.core.policies import CnnPolicy_slim
from AlphaGomoku.games.tic_tac_toe_x_2 import TicTacToeXGameSpec


def worker(policy, sb, wl, seed, total_times, lm, model_path, da, bs, tc):
    env = TicTacToeXGameSpec(sb, wl)
    learn(policy, env, nsteps=sb * sb, nstack=1, seed=seed, total_timesteps=total_times,
          load_model=lm, model_path=model_path, data_augmentation=da,
          BATCH_SIZE=bs, TEMP_CTE=tc)


if __name__ == '__main__':
    policy_fn = CnnPolicy_slim

    config = configparser.RawConfigParser()
    str_read = str(sys.argv[1])
    config_read = 'configuration/' + str_read
    config.read(config_read)
    print(config_read)
    threads = []
    for task in config.sections():
        sb = config.getint(task, 'size_board')
        wl = config.getint(task, 'winning_length')
        seed = config.getint(task, 'seed')
        total_times = config.getint(task, 'total_times')
        lm = config.getboolean(task, 'load_model')
        model_path = config.get(task, 'model_path')
        da = config.getboolean(task, 'data_augmentation')
        bs = config.getint(task, 'BATCH_SIZE')
        tc = config.getint(task, 'TEMP_CTE')
        t = threading.Thread(target=worker, args=(policy_fn, sb, wl, seed, total_times, lm, model_path, da,
                                                  bs, tc))
        threads.append(t)
        time.sleep(3)
        t.start()

    print(threads)
