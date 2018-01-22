#!/usr/bin/env python
from AlphaGomoku.games.tic_tac_toe_x import TicTacToeXGameSpec
from baselines.a2c.a2c import learn
from baselines.a2c.policies import CnnPolicy_TTT, CnnPolicySlim, CnnPolicySlim2

if __name__ == '__main__':
    policy_fn = CnnPolicy_TTT
    env = TicTacToeXGameSpec(3, 3)
    learn(policy_fn, env, nsteps=10
          , nstack=1, seed=0, total_timesteps=10000000, load_model=False, model_path='./model.cpkt')
    env.close()
