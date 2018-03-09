Project for Machine Learning subject in the master of AI in RUG, 2017/2018
Exist two different options for training, self-training or training against a policy.

For self training it is possible to select the options in core/run_gomoku_VS.py or run from terminal core/run_gomoku_VS_terminal.py.
Exist 6 different options for running core/run_gomoku_VS_terminal.py:
    - Option 1: Policy option, we can choose the dimension of the convolution in the next list
        - 1: 1st layer 4x4x128, 2nd layer 2x2x256
        - 2: 1st layer 4x4x128, 2nd layer 2x2x256, 3rd 1x1x128
        - 3: 1st layer 3x3x128, 2nd layer 3x3x256
        - 4: 1st layer 3x3x128, 2nd layer 3x3x256, 3rd 1x1x128
    - Option 2: Size side of the board, for a table of 5x5 write 5.
    - Option 3: Winning length, for a winning length of 4 write 4.
    - Option 4: Number of AI to train, it has to be bigger than 1.
    - Option 5: Batch size, the number of games to play before training. Select 1 to play at the end of every game.
    - Option 6: Seed for the generation of random numbers so that you can repeat experiments.

For self training it is possible to select the options in core/run_gomoku_2.py or run from terminal core/run_gomoku_2_terminal.py.
Exist 6 different options for running core/run_gomoku_2_terminal.py:
    - Option 1: Policy option, we can choose the dimension of the convolution in the next list
        - 1: 1st layer 4x4x128, 2nd layer 2x2x256
        - 2: 1st layer 4x4x128, 2nd layer 2x2x256, 3rd 1x1x128
        - 3: 1st layer 3x3x128, 2nd layer 3x3x256
        - 4: 1st layer 3x3x128, 2nd layer 3x3x256, 3rd 1x1x128
    - Option 2: Size side of the board, for a table of 5x5 write 5.
    - Option 3: Winning length, for a winning length of 4 write 4.
    - Option 4: Constant to control the temperature parameter of the exploration.
    - Option 5: Batch size, the number of games to play before training. Select 1 to play at the end of every game.
    - Option 6: Seed for the generation of random numbers so that you can repeat experiments.
    - Option 7: Opponent policy
        -1 : Policy that doesn't block the AI.
        -2 : Policy that tries to block the AI.

Models are saved every 1000 games for policy training and 3000 games for self training. Also tensorboard file is generated in the same folder of the model.