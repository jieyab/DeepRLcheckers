#!/usr/bin/env python
from AlphaGomoku.core.a2c_2 import learn
from AlphaGomoku.core.policies import CnnPolicy_slim_scope5x5, CnnPolicy_slim_scope5x5_1x1, CnnPolicy_slim_scope5x5_3, \
    CnnPolicy_slim_scope5x5_3_1x1
from AlphaGomoku.games.tic_tac_toe_x_2 import TicTacToeXGameSpec

if __name__ == '__main__':

    policy_fn = CnnPolicy_slim_scope5x5_1x1
    size_board = 5  # int(sys.argv[2])
    winning_length = 4

    seed = 5

    BATCH_SIZE = 128
    TEMP_CTE = 5000

    data_augmentation = True

    # depth conv1 and depth conv2 and weights fc
    # data agumentation
    model_path = "../statistics/random/model.cpkt"
    print(size_board,winning_length,seed,BATCH_SIZE,policy_fn)
    env = TicTacToeXGameSpec(size_board, winning_length)
    learn(policy_fn, env, nsteps=size_board * size_board, nstack=1, seed=seed, total_timesteps=10000000,
          load_model=False, model_path=model_path, data_augmentation=data_augmentation,
          BATCH_SIZE=BATCH_SIZE, TEMP_CTE=TEMP_CTE, expert=True)
