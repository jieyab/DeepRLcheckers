"""
Full code for running a game of tic-tac-toe on a 3 by 3 board.
Two players take turns making moves on squares of the board, the first to get 3 in a row, including diagonals, wins. If
there are no valid moves left to make the game ends a draw.

The main method to use here is play_game which simulates a game to the end using the function args it takes to determine
where each player plays.
The board is represented by a 3 x 3 tuple of ints. A 0 means no player has played in a space, 1 means player one has
played there, -1 means the seconds player has played there. The apply_move method can be used to return a copy of a
given state with a given move applied. This can be useful for doing min-max or monte carlo sampling.
"""
import itertools
import random

from common.base_game_spec import BaseGameSpec
from techniques.min_max import evaluate

board_size = 3
pieces_number = 3


def new_board():
    """Return a emprty tic-tac-toe board we can use for simulating a game.

    Returns:
        3x3 tuple of ints
    """
    board_row = []
    board = []
    # board_size = random.randint(3, 8)
    for j in range(0, board_size):
        board_row.append(0)
    for i in range(0, board_size):
        board.append(tuple(board_row))

    return tuple(board)


def turn(board_state):
    """
    Return the turn of a certain board state

    :param board_state:
    :return:
        turn
    """
    num_plus = 0
    num_minus = 0
    for x, y in itertools.product(range(board_size), range(board_size)):
        if board_state[x][y] == 1:
            num_plus += 1
        elif board_state[x][y] == -1:
            num_minus += 1
    if num_plus == num_minus:
        return 1
    else:
        return -1


def apply_move(board_state, move, side):
    """Returns a copy of the given board_state with the desired move applied.

    Args:
        board_state (3x3 tuple of int): The given board_state we want to apply the move to.
        move (int, int): The position we want to make the move in.
        side (int): The side we are making this move for, 1 for the first player, -1 for the second player.

    Returns:
        (3x3 tuple of int): A copy of the board_state with the given move applied for the given side.
    """
    if side == 0:
        side = turn(board_state)

    move_x, move_y = move

    def get_tuples():
        for x in range(board_size):
            if move_x == x:
                temp = list(board_state[x])
                temp[move_y] = side
                yield tuple(temp)
            else:
                yield board_state[x]

    return tuple(get_tuples())


def available_moves(board_state):
    """Get all legal moves for the current board_state. For Tic-tac-toe that is all positions that do not currently have
    pieces played.

    Args:
        board_state: The board_state we want to check for valid moves.

    Returns:
        Generator of (int, int): All the valid moves that can be played in this position.
    """
    for x, y in itertools.product(range(board_size), range(board_size)):
        if board_state[x][y] == 0:
            yield (x, y)


def _has_x_in_a_line(line):
    count, side = 0, 0

    for i in line:
        if not side == i:
            side = i
            count = 1
        elif side != 0:
            count += 1

        if count == pieces_number:
            return side

    return 0


def has_winner(board_state):
    """Determine if a player has won on the given board_state.
    Args:
        board_state (3x3 tuple of int): The current board_state we want to evaluate.

    Returns:
        int: 1 if player one has won, -1 if player 2 has won, otherwise 0.
    """
    # check rows
    for x in range(board_size):
        # print board_state[x]
        result = _has_x_in_a_line(board_state[x])
        if result != 0:
            return result

    # check columns
    for y in range(board_size):
        # print [i[y] for i in board_state]
        result = _has_x_in_a_line([i[y] for i in board_state])
        if result != 0:
            return result

    # check diagonals
    for x in range(board_size - pieces_number + 1):
        # print [board_state[x + i][i] for i in range(board_size - x)]
        result = _has_x_in_a_line([board_state[x + i][i] for i in range(board_size - x)])
        if result != 0:
            return result

    for y in range(1, board_size - pieces_number + 1):
        # print [board_state[i][y + i] for i in range(board_size - y)]
        result = _has_x_in_a_line([board_state[i][y + i] for i in range(board_size - y)])
        if result != 0:
            return result

    for x in list(reversed(range(pieces_number - 1, board_size))):
        # print [board_state[x - i][i] for i in range(x + 1)]
        result = _has_x_in_a_line([board_state[x - i][i] for i in range(x + 1)])
        if result != 0:
            return result

    for y in range(1, board_size - pieces_number + 1):
        # print [board_state[i][board_size - 1 - i + y] for i in list(reversed(range(y, board_size)))]
        result = _has_x_in_a_line([board_state[i][board_size - 1 - i + y]
                                   for i in list(reversed(range(pieces_number + y, board_size)))])
        if result != 0:
            return result

    return 0  # no one has won, return 0 for a draw


def play_game(plus_player_func, minus_player_func, log=False):
    """Run a single game of tic-tac-toe until the end, using the provided function args to determine the moves for each
    player.

    Args:
        plus_player_func ((board_state(3 by 3 tuple of int), side(int)) -> move((int, int))): Function that takes the
            current board_state and side this player is playing, and returns the move the player wants to play.
        minus_player_func ((board_state(3 by 3 tuple of int), side(int)) -> move((int, int))): Function that takes the
            current board_state and side this player is playing, and returns the move the player wants to play.
        log (bool): If True progress is logged to console, defaults to False

    Returns:
        int: 1 if the plus_player_func won, -1 if the minus_player_func won and 0 for a draw
    """
    board_state = new_board()
    player_turn = 1
    length_game = 0

    while True:
        length_game = length_game + 1
        _available_moves = list(available_moves(board_state))
        # print _available_moves

        if len(_available_moves) == 0:
            # draw
            if log:
                print("no moves left, game ended a draw")
            return 0.

        # just randomly select available moves
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
        if log:
            print(board_state)

        winner = has_winner(board_state)
        if winner != 0:
            if log:
                print("we have a winner, side: %s" % player_turn)
                print('player turn', length_game)
            return winner
        player_turn = -player_turn


def random_player(board_state, _):
    """A player func that can be used in the play_game method. Given a board state it chooses a move randomly from the
    valid moves in the current state.

    Args:
        board_state (3x3 tuple of int): The current state of the board
        _: the side this player is playing, not used in this function because we are simply choosing the moves randomly

    Returns:
        (int, int): the move we want to play on the current board
    """
    moves = list(available_moves(board_state))
    return random.choice(moves)


def lazy_ai(board_state, _):
    pass


class TicTacToeGameSpec(BaseGameSpec):
    def __init__(self):
        self.available_moves = available_moves
        self.has_winner = has_winner
        self.new_board = new_board
        self.apply_move = apply_move
        self.evaluate = evaluate

    def board_dimensions(self):
        return board_size, board_size


if __name__ == '__main__':
    # example of playing a game
    play_game(random_player, random_player, log=True)
