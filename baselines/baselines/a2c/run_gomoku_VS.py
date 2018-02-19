#!/usr/bin/env python
from baselines.a2c.AI_vs_AI import learn
from baselines.a2c.policies import CnnPolicy, CnnPolicy_slim, CnnPolicy_slim_2, CnnPolicy_slim_TTT, CnnPolicy_slim_TTT_2
from AlphaGomoku.games.tic_tac_toe_x import TicTacToeXGameSpec

if __name__ == '__main__':

    policy_fn = CnnPolicy_slim
    policy_fn_2 = CnnPolicy_slim_2
    size_board= 5
    winning_length = 4
    env = TicTacToeXGameSpec(size_board, winning_length)
    learn(policy_fn,policy_fn_2, env, nsteps=size_board*size_board
          ,nstack=1,  seed= 0,total_timesteps=10000000,load_model=False, model_path='./models/tic_tac_toe.cpkt',data_augmentation=False)
    env.close()
