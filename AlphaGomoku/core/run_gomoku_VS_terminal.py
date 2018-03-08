import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from AlphaGomoku.core.AI_vs_AI_2 import learn
from AlphaGomoku.core.policies import policy_4x4_2x2, policy_4x4_2x2_1x1, policy_4x4_1x1_2x2_1x1, policy_4x4_1x1_2x2_1x1_features
from AlphaGomoku.games.tic_tac_toe_x_2 import TicTacToeXGameSpec

if __name__ == '__main__':
    policy_fn = policy_4x4_2x2
    policy_comand = int(sys.argv[1])
    if policy_comand == 1:
        policy_fn = policy_4x4_2x2
    elif policy_comand == 2:
        policy_fn = policy_4x4_2x2_1x1
    elif policy_comand == 3:
        policy_fn = policy_4x4_1x1_2x2_1x1
    elif policy_comand == 4:
        policy_fn = policy_4x4_1x1_2x2_1x1_features
    size_board = int(sys.argv[2])
    data = int(sys.argv[3])

    if data == 0:
        data_augmentation = False
    else:
        data_augmentation = True


    winning_length = 4

    BATCH_SIZE = 128

    learning_rate = float(sys.argv[4])
    CF = 0.01
    NUMBER_OF_MODELS = int(sys.argv[5])
    BATCH_SIZE = int(sys.argv[6])
    seed = int(sys.argv[7])

    print('________________________policy', policy_fn, 'size_board', size_board, 'winning_length', winning_length, 'NUMBER_OF_MODELS',
          NUMBER_OF_MODELS, 'batch size', BATCH_SIZE, 'seed', seed)
    env = TicTacToeXGameSpec(size_board, winning_length)
    learn(policy_fn, env, nsteps=size_board * size_board
          , nstack=1, seed=seed, total_timesteps=1000000000, load_model=False, model_path='./models/tic_tac_toe.cpkt',
          data_augmentation=data_augmentation, BATCH_SIZE=BATCH_SIZE, NUMBER_OF_MODELS=NUMBER_OF_MODELS, CF=CF,learning_rate=learning_rate)
    env.close()
