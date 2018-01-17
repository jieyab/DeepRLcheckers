#!/usr/bin/env python
import os
import sys

from AlphaGomoku.core.a2c import learn
from AlphaGomoku.core.policies import CnnPolicy
from AlphaGomoku.games.tic_tac_toe_x import TicTacToeXGameSpec
from AlphaGomoku.common import logger

is_mute = 1

if __name__ == '__main__':
    logger.set_level(logger.WARN)

    if is_mute:
        sys.stdout = open(os.devnull, 'w')

    policy_fn = CnnPolicy
    env = TicTacToeXGameSpec(3, 3)
    learn(policy_fn, env, nsteps=10,
          nstack=1, seed=0, total_timesteps=10000000, load_model=False, model_path='../models/gomoku.cpkt')
    env.close()
    # sys.stdout = sys.__stdout__
