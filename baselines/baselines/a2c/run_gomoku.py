#!/usr/bin/env python
from baselines.a2c.a2c import learn
from baselines.a2c.policies import CnnPolicy_slim_TTT
from AlphaGomoku.games.tic_tac_toe_x import TicTacToeXGameSpec

if __name__ == '__main__':

    policy_fn = CnnPolicy_slim_TTT
    size_board= 3
    winning_length = 3
    env = TicTacToeXGameSpec(size_board, winning_length)
    learn(policy_fn, env, nsteps=size_board*size_board
          ,nstack=1,  seed= 0,total_timesteps=10000000,load_model=True, model_path='./models/tic_tac_toe.cpkt')
    env.close()
