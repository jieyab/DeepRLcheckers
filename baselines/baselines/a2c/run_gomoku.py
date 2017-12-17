#!/usr/bin/env python
from baselines.a2c.a2c import learn
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
from AlphaGomoku.games.tic_tac_toe_x import TicTacToeXGameSpec

if __name__ == '__main__':

    policy_fn = CnnPolicy
    env = TicTacToeXGameSpec(3, 3)
    learn(policy_fn, env, nsteps=10
          ,nstack=1,  seed= 0,total_timesteps=10000000,load_model=False, model_path='./models/tic_tac_toe.cpkt')
    env.close()
