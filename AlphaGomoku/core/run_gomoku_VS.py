#!/usr/bin/env python

from AlphaGomoku.core.AI_vs_AI_2 import learn
from AlphaGomoku.core.policies import CnnPolicy_slim_scope5x5, CnnPolicy_slim_scope5x5_1x1,CnnPolicy_slim_scope9x9_1x1, CnnPolicy_slim_scope9x9_1x1
from AlphaGomoku.games.tic_tac_toe_x_2 import TicTacToeXGameSpec


if __name__ == '__main__':
    policy_fn = CnnPolicy_slim_scope9x9_1x1
    size_board = 9
    winning_length = 5
    NUMBER_OF_MODELS = 2
    BATCH_SIZE = 256
    seed = 10
    print(size_board,winning_length,NUMBER_OF_MODELS,policy_fn)
    env = TicTacToeXGameSpec(size_board, winning_length)
    learn(policy_fn, env, nsteps=size_board * size_board
          , nstack=1, seed=seed, total_timesteps=10000000, load_model=False, model_path='./models/tic_tac_toe.cpkt',
          data_augmentation=True, BATCH_SIZE=BATCH_SIZE,NUMBER_OF_MODELS = NUMBER_OF_MODELS)
    env.close()
