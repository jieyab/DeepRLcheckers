#!/usr/bin/env python
from AlphaGomoku.games.tic_tac_toe_x import TicTacToeXGameSpec
from baselines.a2c.human_vs_AI import learn
from baselines.a2c.policies import CnnPolicy, CnnPolicy_VS,CnnPolicy_TTT,CnnPolicy_VS_TTT

if __name__ == '__main__':
    policy_fn = CnnPolicy_TTT
    policy_fn_VS = CnnPolicy_VS_TTT
    env = TicTacToeXGameSpec(3, 3)
    learn(policy_fn,policy_fn_VS, env, nsteps=10
          , nstack=1, seed=0, total_timesteps=10000000, load_model=True, model_path='./model.cpkt')
    env.close()

