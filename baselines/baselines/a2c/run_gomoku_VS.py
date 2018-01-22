#!/usr/bin/env python
from AlphaGomoku.games.tic_tac_toe_x import TicTacToeXGameSpec
from baselines.a2c.AI_vs_AI import learn
from baselines.a2c.policies import CnnPolicy, CnnPolicy_VS,CnnPolicy_TTT,CnnPolicy_VS_TTT, CnnPolicySlim, CnnPolicySlim2

if __name__ == '__main__':
    policy_fn = CnnPolicySlim
    policy_fn_VS = CnnPolicySlim2
    env = TicTacToeXGameSpec(5, 4)
    learn(policy_fn,policy_fn_VS, env, nsteps=25
          , nstack=1, seed=9, total_timesteps=10000000, load_model=False, model_path='./model.cpkt')
    env.close()

