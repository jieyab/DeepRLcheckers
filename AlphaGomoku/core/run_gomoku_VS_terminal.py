import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from AlphaGomoku.core.AI_vs_AI_2 import learn
from AlphaGomoku.core.policies import CnnPolicy_slim_scope5x5, CnnPolicy_slim_scope5x5_1x1, CnnPolicy_slim_scope5x5_3, \
    CnnPolicy_slim_scope5x5_3_1x1
from AlphaGomoku.games.tic_tac_toe_x_2 import TicTacToeXGameSpec

if __name__ == '__main__':
    policy_fn = CnnPolicy_slim_scope5x5
    policy_comand = int(sys.argv[1])
    if policy_comand == 1:
        policy_fn = CnnPolicy_slim_scope5x5
    elif policy_comand == 2:
        policy_fn = CnnPolicy_slim_scope5x5_1x1
    elif policy_comand == 3:
        policy_fn = CnnPolicy_slim_scope5x5_3
    elif policy_comand == 4:
        policy_fn = CnnPolicy_slim_scope5x5_3_1x1
    size_board = int(sys.argv[2])
    winning_length = int(sys.argv[3])
    NUMBER_OF_MODELS = int(sys.argv[4])
    BATCH_SIZE = int(sys.argv[5])
    seed = int(sys.argv[6])
    print('________________________policy', policy_fn, 'size_board', size_board, 'winning_length', winning_length, 'NUMBER_OF_MODELS',
          NUMBER_OF_MODELS, 'batch size', BATCH_SIZE, 'seed', seed)
    env = TicTacToeXGameSpec(size_board, winning_length)
    learn(policy_fn, env, nsteps=size_board * size_board
          , nstack=1, seed=seed, total_timesteps=1000000000, load_model=False, model_path='./models/tic_tac_toe.cpkt',
          data_augmentation=True, BATCH_SIZE=BATCH_SIZE, NUMBER_OF_MODELS=NUMBER_OF_MODELS)
    env.close()
