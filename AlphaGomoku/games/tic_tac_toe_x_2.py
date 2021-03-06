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
import math

from AlphaGomoku.common import logger
from AlphaGomoku.games.base_game_spec import BaseGameSpec

from gym import spaces
import numpy as np
import datetime


def _new_board(board_size):
    """Return a emprty tic-tac-toe board we can use for simulating a game.

    Args:
        board_size (int): The size of one side of the board, a board_size * board_size board is created

    Returns:
        board_size x board_size numpy array of ints
    """
    return np.zeros((1, board_size, board_size, 1), dtype=np.int)


def apply_move(board_state, move, side):
    """Returns a copy of the given board_state with the desired move applied.

    Args:
        board_state (2d tuple of int): The given board_state we want to apply the move to.
        move (int, int): The position we want to make the move in.
        side (int): The side we are making this move for, 1 for the first player, -1 for the second player.

    Returns:
        (2d tuple of int): A copy of the board_state with the given move applied for the given side.
    """
    move_x, move_y = move
    board_state[0, move_x, move_y, 0] = side

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


def available_moves_for_original(board_state):
    """Get all legal moves for the current board_state. For Tic-tac-toe that is all positions that do not currently have
    pieces played.

    Args:
        board_state: The board_state we want to check for valid moves.

    Returns:
        Generator of (int, int): All the valid moves that can be played in this position.
    """
    for x, y in itertools.product(range(len(board_state[0])), range(len(board_state[0]))):
        if board_state[0, x, y, 0] == 0:
            yield (x, y)


def _has_winning_line(line, winning_length):
    count = 0
    last_side = 0
    temp = np.array(line)
    # print("length of temp", len(temp))
    for x in range(len(line)):
        if temp.item(x) != last_side:
            count = 1
            last_side = temp.item(x)

        elif (last_side != 0):
            count += 1
            last_side = temp.item(x)

        if count == winning_length:
            return last_side
    return 0


def has_winner(board_state, winning_length):
    # print("from has_winner",board_state)
    """Determine if a player has won on the given board_state.

    Args:
        board_state (2d tuple of int): The current board_state we want to evaluate.
        winning_length (int): The number of moves in a row needed for a win.

    Returns:
        int: 1 if player one has won, -1 if player 2 has won, otherwise 0.
    """
    board_state = board_state[0, :, :, 0]
    board_width = len(board_state)
    board_height = len(board_state[0])

    # check rows
    for x in range(board_width):
        winner = _has_winning_line(board_state[x, :], winning_length)
        if winner != 0:
            return np.ones((1), dtype=bool)
    # check columns
    for y in range(board_height):
        winner = _has_winning_line(board_state[:, y], winning_length)
        if winner != 0:
            return np.ones((1), dtype=bool)

    # Check diagonals
    for d in range(0, (board_height - winning_length + 1)):
        winner = _has_winning_line(np.diagonal(board_state, d), winning_length)
        if winner != 0:
            return np.ones((1), dtype=bool)
    for d in range(0, (board_height - winning_length + 1)):
        winner = _has_winning_line(np.diagonal(board_state, -d), winning_length)
        if winner != 0:
            return np.ones((1), dtype=bool)
    for d in range(0, (board_height - winning_length + 1)):
        winner = _has_winning_line(np.diagonal(np.fliplr(board_state), d), winning_length)
        if winner != 0:
            return np.ones((1), dtype=bool)
    for d in range(0, (board_height - winning_length + 1)):
        winner = _has_winning_line(np.diagonal(np.fliplr(board_state), -d), winning_length)
        if winner != 0:
            return np.ones((1), dtype=bool)
    return np.zeros((1), dtype=bool)


def _evaluate_line(line, winning_length):
    score = 0
    neutrals = 0
    count = 0
    last_side = 0
    temp = np.array(line)
    for x in range(len(line)):
        if temp.item(x) == last_side:
            count += 1
            if count == winning_length and neutrals == 0:
                return 100000 * temp.item(x)  # a side has already won here
        elif temp.item(x) == 0:  # we could score here
            neutrals += 1
        elif temp.item(x) == -last_side:
            if neutrals + count >= winning_length:
                score += (count - 1) * last_side
            count = 1
            last_side = temp.item(x)
            neutrals = 0
        else:
            last_side = temp.item(x)
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
        score += _evaluate_line(board_state[x, :], winning_length)
    # check columns
    for y in range(board_height):
        score += _evaluate_line(board_state[:, y], winning_length)

    for d in range(0, (board_height - winning_length + 1)):
        score = _evaluate_line(np.diagonal(board_state, d), winning_length)
    for d in range(0, (board_height - winning_length + 1)):
        score = _evaluate_line(np.diagonal(board_state, -d), winning_length)
    for d in range(0, (board_height - winning_length + 1)):
        score = _evaluate_line(np.diagonal(np.fliplr(board_state), d), winning_length)
    for d in range(0, (board_height - winning_length + 1)):
        score = _evaluate_line(np.diagonal(np.fliplr(board_state), -d), winning_length)
    return score


# def play_game(plus_player_func, minus_player_func, board_size=5, winning_length=4, log=False):
#     """Run a single game of tic-tac-toe until the end, using the provided function args to determine the moves for each
#     player.
#
#     Args:
#         plus_player_func ((board_state(board_size by board_size tuple of int), side(int)) -> move((int, int))):
#             Function that takes the current board_state and side this player is playing, and returns the move the player
#             wants to play.
#         minus_player_func ((board_state(board_size by board_size tuple of int), side(int)) -> move((int, int))):
#             Function that takes the current board_state and side this player is playing, and returns the move the player
#             wants to play.
#         board_size (int): The size of a single side of the board. Game is played on a board_size*board_size sized board
#         winning_length (int): The number of pieces in a row needed to win a game.
#         log (bool): If True progress is logged to console, defaults to False
#
#     Returns:
#         int: 1 if the plus_player_func won, -1 if the minus_player_func won and 0 for a draw
#     """
#     board_state = _new_board(board_size)
#     player_turn = 1
#
#     while True:
#         _available_moves = list(available_moves(board_state))
#         if len(_available_moves) == 0:
#             # draw
#             if log:
#                 print("no moves left, game ended a draw")
#             return 0.
#         if player_turn > 0:
#             move = plus_player_func(board_state, 1)
#         else:
#             move = minus_player_func(board_state, -1)
#
#         if move not in _available_moves:
#             # if a player makes an invalid move the other player wins
#             if log:
#                 print("illegal move ", move)
#             return -player_turn
#
#         board_state = apply_move(board_state, move, player_turn)
#
#         winner = has_winner(board_state, winning_length)
#         if winner != 0:
#             if log:
#                 print("we have a winner, side: %s" % player_turn)
#             return winner
#         player_turn = -player_turn


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


def _possible_move(line, winning_length, side):
    # print(line)
    count = 0
    tmp_start = 0
    start = 0
    end = 0
    max_length = 0
    temp = np.array(line)
    # print("length of temp", len(temp))
    for x in range(len(line)):
        if temp.item(x) == side:
            if count == 0:
                tmp_start = x
            count += 1
        else:
            count = 0

        if count > max_length:
            before_count = 0
            after_count = 0
            for i in reversed(range(tmp_start)):
                if temp.item(i) == 0:
                    before_count += 1
                else:
                    break

            for i in range(x + 1, len(line)):
                if temp.item(i) == 0:
                    after_count += 1
                else:
                    break

            if before_count + after_count + count >= winning_length and count > max_length:
                # print('start', tmp_start, 'end', x)
                max_length = count
                start = tmp_start
                end = x

    if max_length == 0:
        return 0, 0
    else:
        possible_moves = []
        if start != 0:
            if temp.item(start - 1) == 0:
                possible_moves.append(start - 1)

        if end != len(line) - 1:
            if temp.item(end + 1) == 0:
                possible_moves.append(end + 1)

        return max_length, random.choice(possible_moves)


def next_move_by_policy(board_state, winning_length, side, expert=False):
    from datetime import datetime
    random.seed(datetime.now())
    if len(list(available_moves(board_state))) == len(board_state[0]) * len(board_state[0]):
        move = [random.randint(math.floor(float((len(board_state[0]) - 1) / 2.0)) - 1,
                               math.ceil(float((len(board_state[0]) - 1) / 2.0)) + 1),
                random.randint(math.floor(float((len(board_state[0]) - 1) / 2.0)) - 1,
                               math.ceil(float((len(board_state[0]) - 1) / 2.0)) + 1)]
        return move
    elif len(list(available_moves(board_state))) == len(board_state[0]) * len(board_state[0]) - 1:
        while True:
            move = [random.randint(math.floor(float((len(board_state[0]) - 1) / 2.0)) - 1,
                                   math.ceil(float((len(board_state[0]) - 1) / 2.0)) + 1),
                    random.randint(math.floor(float((len(board_state[0]) - 1) / 2.0)) - 1,
                                   math.ceil(float((len(board_state[0]) - 1) / 2.0)) + 1)]
            if tuple(move) in list(available_moves(board_state)):
                return move

    original_state = np.copy(board_state)
    board_state = board_state[0, :, :, 0]
    board_width = len(board_state)
    board_height = len(board_state[0])
    new_length = 0
    new_length_b = 0
    new_position = []
    new_position_b = []

    # check rows
    for x in range(board_width):
        max_length, position = _possible_move(board_state[x, :], winning_length, side)
        max_length_b, position_b = _possible_move(board_state[x, :], winning_length, -side)

        if max_length > new_length:
            new_length = max_length
            new_position.clear()
            new_position.append([x, position])
        elif max_length == new_length:
            new_position.append([x, position])

        if max_length_b > new_length_b:
            new_length_b = max_length_b
            new_position_b.clear()
            new_position_b.append([x, position_b])
        elif max_length_b == new_length_b:
            new_position_b.append([x, position_b])

    # check columns
    for y in range(board_height):
        max_length, position = _possible_move(board_state[:, y], winning_length, side)
        max_length_b, position_b = _possible_move(board_state[:, y], winning_length, -side)

        if max_length > new_length:
            new_length = max_length
            new_position.clear()
            new_position.append([position, y])
        elif max_length == new_length:
            new_position.append([position, y])

        if max_length_b > new_length_b:
            new_length_b = max_length_b
            new_position_b.clear()
            new_position_b.append([position_b, y])
        elif max_length_b == new_length_b:
            new_position_b.append([position_b, y])

    # Check diagonals
    for d in range(0, (board_height - winning_length + 1)):
        max_length, position = _possible_move(np.diagonal(board_state, d), winning_length, side)
        max_length_b, position_b = _possible_move(np.diagonal(board_state, d), winning_length, -side)

        if max_length > new_length:
            new_length = max_length
            new_position.clear()
            new_position.append([position, position + d])
        elif max_length == new_length:
            new_position.append([position, position + d])

        if max_length_b > new_length_b:
            new_length_b = max_length_b
            new_position_b.clear()
            new_position_b.append([position_b, position_b + d])
        elif max_length_b == new_length_b:
            new_position_b.append([position_b, position_b + d])

    for d in range(1, (board_height - winning_length + 1)):
        max_length, position = _possible_move(np.diagonal(board_state, -d), winning_length, side)
        max_length_b, position_b = _possible_move(np.diagonal(board_state, -d), winning_length, -side)

        if max_length > new_length:
            new_length = max_length
            new_position.clear()
            new_position.append([position + d, position])
        elif max_length == new_length:
            new_position.append([position + d, position])

        if max_length_b > new_length_b:
            new_length_b = max_length_b
            new_position_b.clear()
            new_position_b.append([position_b + d, position_b])
        elif max_length_b == new_length_b:
            new_position_b.append([position_b + d, position_b])

    for d in range(0, (board_height - winning_length + 1)):
        max_length, position = _possible_move(np.diagonal(np.fliplr(board_state), d), winning_length, side)
        max_length_b, position_b = _possible_move(np.diagonal(np.fliplr(board_state), d), winning_length, -side)

        if max_length > new_length:
            new_length = max_length
            new_position.clear()
            new_position.append([position, len(board_state[0]) - 1 - position - d])
        elif max_length == new_length:
            new_position.append([position, len(board_state[0]) - 1 - position - d])

        if max_length_b > new_length_b:
            new_length_b = max_length_b
            new_position_b.clear()
            # print(np.diagonal(np.fliplr(board_state), d))
            # print('max_length_b', max_length_b, 'position_b', position_b)
            new_position_b.append([position_b, len(board_state[0]) - 1 - position_b - d])
        elif max_length_b == new_length_b:
            new_position_b.append([position_b, len(board_state[0]) - 1 - position_b - d])

    for d in range(1, (board_height - winning_length + 1)):
        max_length, position = _possible_move(np.diagonal(np.fliplr(board_state), -d), winning_length, side)
        max_length_b, position_b = _possible_move(np.diagonal(np.fliplr(board_state), -d), winning_length, -side)

        if max_length > new_length:
            new_length = max_length
            new_position.clear()
            new_position.append([position + d, len(board_state[0]) - 1 - position])
        elif max_length == new_length:
            new_position.append([position + d, len(board_state[0]) - 1 - position])

        if max_length_b > new_length_b:
            new_length_b = max_length_b
            new_position_b.clear()
            new_position_b.append([position_b + d, len(board_state[0]) - 1 - position_b])
        elif max_length_b == new_length_b:
            new_position_b.append([position_b + d, len(board_state[0]) - 1 - position_b])

    if new_length == winning_length - 1:
        return random.choice(new_position)

    if expert:
        if new_length_b >= winning_length - 2:
            return random.choice(new_position_b)

    if new_length != 0:
        return random.choice(new_position)
    else:
        if new_length_b != 0:
            return random.choice(new_position_b)
        else:
            # print(board_state)
            # print(original_state)
            # print(list(available_moves_for_original(original_state)))
            return random.choice(list(available_moves_for_original(original_state)))


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

        # rewards
        self.reward_winning = 0.5
        self.reward_lossing = -0.5
        self.reward_illegal_move = -1
        self.reward_draw = -0.5

        ##Our code
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self._board_size, self._board_size, 1))
        self.action_space = (self._board_size * self._board_size)
        self.board_state = _new_board(board_size)
        self.num_envs = 1  # Change for more enviroments
        self.remotes = [1]
        self.games_wonAI = 0
        self.games_won_A = 0
        self.games_won_B = 0
        self.games_wonRandom = 0
        self.flag = False
        self.illegal_games = 0
        self.print = True
        self.games_finish_in_draw = 0

    def new_board(self):
        self.board_state = _new_board(self._board_size)
        return self.board_state

    def has_winner(self):
        return has_winner(self.board_state, self._winning_length)

    def board_dimensions(self):
        return self._board_size, self._board_size

    def evaluate(self, board_state):
        return evaluate(board_state, self._winning_length)

    def reset(self):
        # print('Enviroment has been reset')
        self.board_state = _new_board(self._board_size)
        return self.board_state

    def random_move(self):
        return random.randint(0, self._board_size * self._board_size)

    def close(self):
        pass

    def illegal_move(self, move):
        move_x, move_y = move
        if (self.board_state[0, move_x, move_y, 0] != 0):
            # print(self.board_state[0, :, :, 0], move)
            return True

        return False

    def available_moves_1(self):
        list_avaible_moves = []
        for x, y in itertools.product(range(self._board_size), range(self._board_size)):
            if (self.board_state[0, y, x, 0] == 0):
                list_avaible_moves.append(x * self._board_size + y)
        return list_avaible_moves

    def opponent_move(self):
        action_random_move = random.randint(0, len(self.available_moves_1()) - 1)
        action_random = self.available_moves_1()
        action_random = action_random[action_random_move]
        action_random = [int(action_random % self._board_size), int(action_random / self._board_size)]
        self.board_state = apply_move(self.board_state, action_random, -1)

    def print_stadistics(self, thread):
        self.print = False
        total = 1 + self.games_wonAI + self.games_wonRandom + self.games_finish_in_draw + self.illegal_games
        print('- ' * 20)
        print('Thread:', thread)
        print('AI wins in', (100 * self.games_wonAI) / total)
        print('Random player wins ', (100 * self.games_wonRandom) / total)
        print('Draws', (100 * self.games_finish_in_draw) / total)
        print('AI made', (100 * self.illegal_games / total), 'illegal moves')
        print('- ' * 20)

        self.games_wonAI = 0
        self.games_wonRandom = 0
        self.games_finish_in_draw = 0
        self.illegal_games = 0

    def get_stadistics(self):
        total = 1 + self.games_wonAI + self.games_wonRandom + self.games_finish_in_draw + self.illegal_games

        won_AI = 100 * self.games_wonAI / total
        won_random = 100 * self.games_wonRandom / total
        draws =100* self.games_finish_in_draw / total
        illegal_games = 100*self.illegal_games / total

        return won_AI, won_random, draws, illegal_games

    def step(self, actions_nn):
        # print(actions_nn)
        actions_nn = actions_nn[0]
        reward = np.zeros(1)
        actions_nn = [int(actions_nn % self._board_size),
                      int(actions_nn / self._board_size)]  # Convert move from number to X,Y

        if (self.illegal_move(actions_nn)):  # Check if the move was illegal
            self.illegal_games += 1
            print(self.board_state[0, :, :, 0])
            self.board_state = self.new_board()
            self.print = True
            reward[0] = self.reward_illegal_move
            winner = np.ones((1), dtype=bool)

            return (self.board_state, reward, winner, 0, True)

        self.board_state = apply_move(self.board_state, actions_nn, 1)  # Apply move to the board
        winner = has_winner(self.board_state, self._winning_length)  # Check for winner

        if winner[0] == True:
            reward[0] = self.reward_winning
            self.games_wonAI += 1
            # print('AI won')

        else:  # If there is no winner check for draw and make random move
            if (len(self.available_moves_1()) == 0):

                # print('Draw')
                self.games_finish_in_draw += 1
                reward[0] = self.reward_draw

            else:
                self.opponent_move()
                winner = has_winner(self.board_state, self._winning_length)
                if winner[0] == True:
                    reward[0] = self.reward_lossing
                    self.games_wonRandom += 1

                if (len(self.available_moves_1()) == 0):
                    # print('Draw')
                    self.games_finish_in_draw += 1
                    reward[0] = self.reward_draw

        # if ((self.games_won_A + self.games_won_B + self.games_finish_in_draw + self.illegal_games) % 999 == 0):
        #     print(self.board_state[0, :, :, 0])
        #     print(datetime.datetime.now().time())

        if reward[0] != 0:
            # print(self.board_state[0, :, :, 0])
            self.board_state = _new_board(self._board_size)
            self.print = True

            # if (((self.games_wonAI + self.games_wonRandom + self.games_finish_in_draw + self.illegal_games) % 999 == 0) and self.print):
            # self.print_stadistics()

        return self.board_state, reward, winner, 0, False

    def print_stadistics_vs(self):
        total = 1 + self.games_won_A + self.games_won_B + self.games_finish_in_draw + self.illegal_games

        print('A wins in', (100 * self.games_won_A) / total)
        print('B wins in', (100 * self.games_won_B) / total)
        print('Draws', (100 * self.games_finish_in_draw) / total)
        print('AI made', (100 * self.illegal_games / total), 'illegal moves')
        print('total', total)
        print(' - - - - - - - - - - - -')

    def get_stadistics_vs(self):
        total = 1 + self.games_won_A + self.games_won_B + self.games_finish_in_draw + self.illegal_games

        won_A = 100 * self.games_won_A / total
        won_B = 100 * self.games_won_B / total
        draws = 100 * self.games_finish_in_draw / total
        illegal_games = 100 * self.illegal_games / total

        self.games_won_A = 0
        self.games_won_B = 0
        self.games_finish_in_draw = 0
        self.illegal_games = 0

        return won_A, won_B, draws, illegal_games

    def current_state(self, state, action, side):
        """return the board state from the perspective of the current player
        shape: 4*width*height"""

        square_state = np.zeros((4, len(state[0]), len(state[0])))

        for x, y in itertools.product(range(len(state[0])), range(len(state[0]))):
            if state[x, y, 0] == side:
                square_state[0][x, y] = 1.0
            elif state[x, y, 0] == -side:
                square_state[1][x, y] = 1.0

        if action >= 0:
            square_state[2][int(action % len(state[0])), int(action / len(state[0]))] = 1
        if side == 1:
            square_state[3][:, :] = 1.0

        return square_state

    def step_vs(self, actions_nn, side, clear_last_obs=True):
        if side == 'A':
            token = 1

        if side == 'B':
            token = -1

        actions_nn = actions_nn[0]
        reward = np.zeros(1)
        actions_nn = [int(actions_nn % self._board_size),
                      int(actions_nn / self._board_size)]  # Convert move from number to X,Y
        if (self.illegal_move(actions_nn)):
            print('ILLEGALILLEGALILLEGALVILLEGALV')
            illegal = True

        self.board_state = apply_move(self.board_state, actions_nn, token)  # Apply move to the board
        winner = has_winner(self.board_state, self._winning_length)  # Check for winner

        if winner[0]:
            reward[0] = self.reward_winning
            if side == 'A':
                self.games_won_A += 1
            if side == 'B':
                self.games_won_B += 1
                # print(self.board_state[0, :, :, 0], actions_nn,'win')

        else:  # If there is no winner check for draw and make random move
            if len(self.available_moves_1()) == 0:
                # print(self.board_state[0, :, :, 0], actions_nn,'draw')
                self.games_finish_in_draw += 1
                reward[0] = self.reward_draw

        # if ((self.games_won_A + self.games_won_B + self.games_finish_in_draw + self.illegal_games) % 999 == 0):
        #    print(self.board_state[0, :, :, 0])
        #    print(datetime.datetime.now().time())

        if reward[0] != 0:
            # print(self.board_state[0, :, :, 0])
            if clear_last_obs:
                self.board_state = _new_board(self._board_size)

        # if (((self.games_won_A + self.games_won_B + self.games_finish_in_draw + self.illegal_games) % 1000 == 0)
        #     and self.print):
        #     self.print_stadistics_vs()

        return self.board_state, reward, winner, 0, False

    def clear_board(self):
        self.board_state = _new_board(self._board_size)

    def get_illegal_moves(self):
        illegal_moves = list(range(self._board_size * self._board_size))
        legal_moves = self.available_moves_1()
        for i in legal_moves:
            illegal_moves.remove(i)
        return illegal_moves

    def dimensions(self):
        return self._board_size

    def step_smart(self, actions_nn, expert):
        # print(actions_nn)
        actions_nn = actions_nn[0]
        reward = np.zeros(1)
        actions_nn = [int(actions_nn % self._board_size),
                      int(actions_nn / self._board_size)]  # Convert move from number to X,Y

        if (self.illegal_move(actions_nn)):  # Check if the move was illegal
            self.illegal_games += 1
            print(self.board_state[0, :, :, 0])
            self.board_state = self.new_board()
            self.print = True
            reward[0] = self.reward_illegal_move
            winner = np.ones((1), dtype=bool)

            return (self.board_state, reward, winner, 0, True)

        self.board_state = apply_move(self.board_state, actions_nn, 1)  # Apply move to the board
        winner = has_winner(self.board_state, self._winning_length)  # Check for winner

        if winner[0] == True:
            reward[0] = self.reward_winning
            self.games_wonAI += 1
            #print(self.board_state[0, :, :, 0])
            # print('AI won')

        else:  # If there is no winner check for draw and make random move
            if len(self.available_moves_1()) == 0:

                # print('Draw')
                self.games_finish_in_draw += 1
                reward[0] = self.reward_draw

            else:
                # self.opponent_move()
                # print(self.board_state[0, :, :, 0])
                move = next_move_by_policy(self.board_state, self._winning_length, -1, expert)
                self.board_state = apply_move(self.board_state, move, -1)  # Apply move to the board

                winner = has_winner(self.board_state, self._winning_length)
                if winner[0] == True:
                    reward[0] = self.reward_lossing
                    self.games_wonRandom += 1

                if (len(self.available_moves_1()) == 0):
                    # print('Draw')
                    self.games_finish_in_draw += 1
                    reward[0] = self.reward_draw

        # if ((self.games_won_A + self.games_won_B + self.games_finish_in_draw + self.illegal_games) % 999 == 0):
        #     print(self.board_state[0, :, :, 0])
        #     print(datetime.datetime.now().time())

        if reward[0] != 0:
            #print(self.board_state[0, :, :, 0])
            self.board_state = _new_board(self._board_size)
            self.print = True
            if (random.uniform(0, 1) >= 0.5):
                self.board_state = apply_move(self.board_state, actions_nn, 1)

                # if (((self.games_wonAI + self.games_wonRandom + self.games_finish_in_draw + self.illegal_games) % 999 == 0) and self.print):
            # self.print_stadistics()

        return self.board_state, reward, winner, 0, False


if __name__ == '__main__':
    env = TicTacToeXGameSpec(3, 3)

    print(env.step(8))
