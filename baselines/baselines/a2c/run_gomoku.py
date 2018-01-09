#!/usr/bin/env python
import os
import sys

from baselines.a2c.a2c import learn
from baselines.a2c.policies import CnnPolicy
from games.tic_tac_toe_x import TicTacToeXGameSpec

is_mute = 0

if __name__ == '__main__':
    if is_mute:
        sys.stdout = open(os.devnull, 'w')

    policy_fn = CnnPolicy
    env = TicTacToeXGameSpec(3, 3)
    learn(policy_fn, env, nsteps=10,
          nstack=1, seed=0, total_timesteps=10000, load_model=False, model_path='./models/tic_tac_toe.cpkt')
    env.close()
    # sys.stdout = sys.__stdout__
