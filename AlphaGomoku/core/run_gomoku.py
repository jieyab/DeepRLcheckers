#!/usr/bin/env python

from AlphaGomoku.common import logger
from AlphaGomoku.core.a2c import learn, play
from AlphaGomoku.core.policies import CnnPolicy
from AlphaGomoku.games.tic_tac_toe_x import TicTacToeXGameSpec

if __name__ == '__main__':
    logger.set_level(logger.WARN)

    policy_fn = CnnPolicy
    env = TicTacToeXGameSpec(3, 3)
    # learn(policy_fn, env, nsteps=10,
    #       nstack=1, seed=0, total_timesteps=200, load_model=False, model_path='../models/gomoku.cpkt')
    play(policy_fn, env, nsteps=10, nstack=1, seed=0, model_path='../models/gomoku.cpkt')
    env.close()
