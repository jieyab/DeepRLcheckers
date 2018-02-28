#!/usr/bin/env python

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from AlphaGomoku.common import logger
from AlphaGomoku.core.a2c import *
from AlphaGomoku.core.policies import CnnPolicy, CnnPolicy2, CnnPolicy_slim, CnnPolicy_slim_2
from AlphaGomoku.games.tic_tac_toe_x import TicTacToeXGameSpec

if __name__ == '__main__':
    #logger.set_level(logger.CURRENT)

    board_size = 5
    winning_length = 4
    env = TicTacToeXGameSpec(board_size, winning_length)
    # learn(CnnPolicy, CnnPolicy2, env, nsteps=board_size * board_size, log_interval=5,
    #       nstack=1, seed=0, total_timesteps=1000000000, load_model=False, model_path='../models/gomoku_p1.cpkt',
    #       model_path2='../models/gomoku_p2.cpkt')
    # play(CnnPolicy, CnnPolicy2, env, nsteps=board_size * board_size, nstack=1, seed=0, model_path='../models/gomoku_p1.cpkt',
    #      model_path2='../models/gomoku_p2.cpkt')
    # play(CnnPolicy_slim, CnnPolicy_slim_2, env, nsteps=board_size * board_size, nstack=1, seed=0,
    #      model_path='../models/tic_tac_toe.cpkt',
    #      model_path2='../models/tic_tac_toe.cpkt')
    play_with_policy(CnnPolicy_slim, CnnPolicy_slim_2, env, nsteps=board_size * board_size, nstack=1, seed=0)
    env.close()
