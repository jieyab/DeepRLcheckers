"""
Full code for running a game of tic-tac-toe on a board of any size with a specified number in a row for the win. This is
similar to tic_tac_toe.py but all relevent moves are paramiterized by board_size arg that sets how big the board is and
winning_length which determines how many in a row are needed to win. Defaults are 5 and 4. This allows you to play games
in a more complex environment than standard tic-tac-toe.

Two players take turns making moves on squares of the board, the first to get winning_length in a row, including
diagonals, wins. If there are no valid moves left to make the game ends a draw.

The main method to use here is play_game which simulates a game to the end using the function args it takes to determine
where each player plays.
The board is represented by a board_size x board_size tuple of ints. A 0 means no player has played in a space, 1 means
player one has played there, -1 means the seconds player has played there. The apply_move method can be used to return a
copy of a given state with a given move applied. This can be useful for doing min-max or monte carlo sampling.
"""
import itertools
import random


from common.base_game_spec import BaseGameSpec
from gym import spaces
import numpy as np


def _new_board2(board_size):
    """Return a emprty tic-tac-toe board we can use for simulating a game.

    Args:
        board_size (int): The size of one side of the board, a board_size * board_size board is created

    Returns:
        board_size x board_size tuple of ints
    """
    return tuple(tuple(0 for _ in range(board_size)) for _ in range(board_size))

def _new_board(board_size):
    """Return a emprty tic-tac-toe board we can use for simulating a game.

    Args:
        board_size (int): The size of one side of the board, a board_size * board_size board is created

    Returns:
        board_size x board_size numpy array of ints
    """
    return np.zeros((1,board_size,board_size,1))


def apply_move(board_state, move, side):
    """Returns a copy of the given board_state with the desired move applied.

    Args:
        board_state (2d tuple of int): The given board_state we want to apply the move to.
        move (int, int): The position we want to make the move in.
        side (int): The side we are making this move for, 1 for the first player, -1 for the second player.

    Returns:
        (2d tuple of int): A copy of the board_state with the given move applied for the given side.
    """
    print(move)
    move_x, move_y = move
    print(board_state.shape)
    board_state[0,move_x,move_y,0] = side

    return board_state


def available_moves(board_state):
    """Get all legal moves for the current board_state. For Tic-tac-toe that is all positions that do not currently have
    pieces played.

    Args:
        board_state: The board_state we want to check for valid moves.

    Returns:
        Generator of (int, int): All the valid moves that can be played in this position.
    """
    for x, y in itertools.product(range(len(board_state)), range(len(board_state[0]))):
        if board_state[x][y] is 0:
            yield (x, y)


def _has_winning_line(line, winning_length):
    count = 0
    last_side = 0
    for x in line:
        if x == last_side:
            count += 1
            if count == winning_length:
                return last_side
        else:
            count = 1
            last_side = x
    return 0


def has_winner(board_state, winning_length):
    """Determine if a player has won on the given board_state.

    Args:
        board_state (2d tuple of int): The current board_state we want to evaluate.
        winning_length (int): The number of moves in a row needed for a win.

    Returns:
        int: 1 if player one has won, -1 if player 2 has won, otherwise 0.
    """
    board_width = len(board_state)
    board_height = len(board_state[0])

    # check rows
    for x in range(board_width):
        winner = _has_winning_line(board_state[x], winning_length)
        if winner != 0:
            # return True
            return np.ones((1),dtype=bool)
    # check columns
    for y in range(board_height):
        winner = _has_winning_line((i[y] for i in board_state), winning_length)
        if winner != 0:
            # return True
            return np.ones((1), dtype=bool)

    # check diagonals
    diagonals_start = -(board_width - winning_length)
    diagonals_end = (board_width - winning_length)
    for d in range(diagonals_start, diagonals_end + 1):
        winner = _has_winning_line(
            (board_state[i][i + d] for i in range(max(-d, 0), min(board_width, board_height - d))),
            winning_length)
        if winner != 0:
            # return True
            return np.ones((1), dtype=bool)
    for d in range(diagonals_start, diagonals_end + 1):
        winner = _has_winning_line(
            (board_state[i][board_height - i - d - 1] for i in range(max(-d, 0), min(board_width, board_height - d))),
            winning_length)
        if winner != 0:
            # return True
            return np.ones((1), dtype=bool)

    # return False  # no one has won, return 0 for a draw
    return np.zeros((1), dtype=bool)

def _evaluate_line(line, winning_length):
    count = 0
    last_side = 0
    score = 0
    neutrals = 0

    for x in line:
        if x == last_side:
            count += 1
            if count == winning_length and neutrals == 0:
                return 100000 * x  # a side has already won here
        elif x == 0:  # we could score here
            neutrals += 1
        elif x == -last_side:
            if neutrals + count >= winning_length:
                score += (count - 1) * last_side
            count = 1
            last_side = x
            neutrals = 0
        else:
            last_side = x
            count = 1

    if neutrals + count >= winning_length:
        score += (count - 1) * last_side

    return score


def evaluate(board_state, winning_length):
    """An evaluation function for this game, gives an estimate of how good the board position is for the plus player.
    There is no specific range for the values returned, they just need to be relative to each other.

    Args:
        winning_length (int): The length needed to win a game
        board_state (tuple): State of the board

    Returns:
        number
    """
    board_width = len(board_state)
    board_height = len(board_state[0])

    score = 0

    # check rows
    for x in range(board_width):
        score += _evaluate_line(board_state[x], winning_length)
    # check columns
    for y in range(board_height):
        score += _evaluate_line((i[y] for i in board_state), winning_length)

    # check diagonals
    diagonals_start = -(board_width - winning_length)
    diagonals_end = (board_width - winning_length)
    for d in range(diagonals_start, diagonals_end + 1):
        score += _evaluate_line(
            (board_state[i][i + d] for i in range(max(-d, 0), min(board_width, board_height - d))),
            winning_length)
    for d in range(diagonals_start, diagonals_end + 1):
        score += _evaluate_line(
            (board_state[i][board_height - i - d - 1] for i in range(max(-d, 0), min(board_width, board_height - d))),
            winning_length)

    return score


def play_game(plus_player_func, minus_player_func, board_size=5, winning_length=4, log=False):
    """Run a single game of tic-tac-toe until the end, using the provided function args to determine the moves for each
    player.

    Args:
        plus_player_func ((board_state(board_size by board_size tuple of int), side(int)) -> move((int, int))):
            Function that takes the current board_state and side this player is playing, and returns the move the player
            wants to play.
        minus_player_func ((board_state(board_size by board_size tuple of int), side(int)) -> move((int, int))):
            Function that takes the current board_state and side this player is playing, and returns the move the player
            wants to play.
        board_size (int): The size of a single side of the board. Game is played on a board_size*board_size sized board
        winning_length (int): The number of pieces in a row needed to win a game.
        log (bool): If True progress is logged to console, defaults to False

    Returns:
        int: 1 if the plus_player_func won, -1 if the minus_player_func won and 0 for a draw
    """
    board_state = _new_board(board_size)
    player_turn = 1

    while True:
        _available_moves = list(available_moves(board_state))
        if len(_available_moves) == 0:
            # draw
            if log:
                print("no moves left, game ended a draw")
            return 0.
        if player_turn > 0:
            move = plus_player_func(board_state, 1)
        else:
            move = minus_player_func(board_state, -1)

        if move not in _available_moves:
            # if a player makes an invalid move the other player wins
            if log:
                print("illegal move ", move)
            return -player_turn

        board_state = apply_move(board_state, move, player_turn)
        print(board_state)

        winner = has_winner(board_state, winning_length)
        if winner != 0:
            if log:
                print("we have a winner, side: %s" % player_turn)
            return winner
        player_turn = -player_turn


def random_player(board_state, _):
    """A player func that can be used in the play_game method. Given a board state it chooses a move randomly from the
    valid moves in the current state.

    Args:
        board_state (2d tuple of int): The current state of the board
        _: the side this player is playing, not used in this function because we are simply choosing the moves randomly

    Returns:
        (int, int): the move we want to play on the current board
    """
    moves = list(available_moves(board_state))
    return random.choice(moves)



class TicTacToeXGameSpec(BaseGameSpec):
    def __init__(self, board_size, winning_length):
        """

        Args:
            board_size (int): The length of one side of the board, so the bard will have board_size*board_size total
                squares
            winning_length (int): The length in a row needed to win the game. Should be less than or equal to board_size
        """
        if not isinstance(board_size, int):
            raise TypeError("board_size must be an int")
        if not isinstance(winning_length, int):
            raise TypeError("winning_length must be an int")
        if winning_length > board_size:
            raise ValueError("winning_length must be less than or equal to board_size")

        self._winning_length = winning_length
        self._board_size = board_size
        self.available_moves = available_moves
        self.apply_move = apply_move


        ##Our code
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self._board_size, self._board_size, 1))
        self.action_space = (self._board_size*self._board_size)
        self.board_state = _new_board(board_size)
        self.num_envs = 1 #Change for more enviroments
        self.remotes =[1]

    def new_board(self):
        self.board_state = _new_board(self._board_size)
        return self.board_state

    def has_winner(self):
        return has_winner(self.board_state, self._winning_length)

    def board_dimensions(self):
        return self._board_size, self._board_size

    def evaluate(self, board_state):
        return evaluate(board_state, self._winning_length)

    def step(self,actions_nn):
        actions = [int(actions_nn%self._board_size),int(actions_nn/self._board_size)]
        self.board_state = apply_move(self.board_state, actions, 1)
        winner = has_winner(self.board_state, self._winning_length)
        return self.board_state, np.zeros(1), [winner], 0

    def reset(self):
        print('Enviroment has been reset')
        self.board_state = _new_board(self._board_size)
        return self.board_state





if __name__ == '__main__':
    # example of playing a game
     play_game(random_player, random_player, log=True, board_size=10, winning_length=4)
      # winning_length = 3

      # if(has_winner(((1,1,0),(-1,-1,-1),(0,0,0)), winning_length)):
      #     print("we have a winner")
      # else :
      #     print("no winner yet")
