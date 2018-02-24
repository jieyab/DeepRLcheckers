#!/usr/bin/env python
from AlphaGomoku.core.a2c_2 import learn
from AlphaGomoku.core.policies import CnnPolicy_slim
from AlphaGomoku.games.tic_tac_toe_x_2 import TicTacToeXGameSpec

if __name__ == '__main__':



    policy_fn = CnnPolicy_slim
    size_board= 5 #int(sys.argv[2])
    winning_length = 4

    seed = 1

    BATCH_SIZE = 500
    TEMP_CTE = 2
    TEMP_COUNTER = 10000




    data_augmentation = True

    #depth conv1 and depth conv2 and weights fc
    # data agumentation
    model_path ='../models/tic_tac_toe_23-02-2018_19-35-56_seed_0_BATCH_100_TEMP_2_DA_True_VF_0.5.cpkt'
    env = TicTacToeXGameSpec(size_board, winning_length)
    learn(policy_fn, env, nsteps=size_board*size_board
          , nstack=1, seed=seed, total_timesteps=10000000, load_model=False, model_path=model_path,
          data_augmentation=data_augmentation, BATCH_SIZE=BATCH_SIZE,TEMP_CTE=TEMP_CTE,TEMP_COUNTER=TEMP_COUNTER)
