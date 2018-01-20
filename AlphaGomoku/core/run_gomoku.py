#!/usr/bin/env python

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from AlphaGomoku.common import logger
from AlphaGomoku.core.a2c import learn, play
from AlphaGomoku.core.policies import CnnPolicy
from AlphaGomoku.games.tic_tac_toe_x import TicTacToeXGameSpec

if __name__ == '__main__':
    logger.set_level(logger.INFO)

    policy_fn = CnnPolicy
    env = TicTacToeXGameSpec(5, 4)
    learn(policy_fn, env, nsteps=25, log_interval=5,
          nstack=1, seed=0, total_timesteps=1000000000, load_model=False, model_path='../models/gomoku.cpkt')
    play(policy_fn, env, nsteps=36, nstack=1, seed=0, model_path='../models/gomoku.cpkt')
    env.close()
