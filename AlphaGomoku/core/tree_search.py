import math
import operator
import random

import networkx as nx
import numpy as np

import AlphaGomoku.core.utils as ut
import AlphaGomoku.games.tic_tac_toe_x as tt
from AlphaGomoku.common import logger


class MonteCarlo:
    def __init__(self, env, model, model2):
        self.digraph = nx.DiGraph()
        self.node_counter = 0

        self.num_simulations = 0
        self.board_size = env.get_board_size()
        self.winning_length = env.get_winning_length()

        self.digraph.add_node(self.node_counter,
                              num_visit=0,
                              Q=0,
                              u=0,
                              P=1,
                              side=-1,
                              action=0,
                              state=tt.new_board(self.board_size))

        self.node_counter += 1
        self.last_node = 0

        self.model = model
        self.model2 = model2
        self._c_puct = 5
        self._n_play_out = 200

        self.list_plus_board_states = []
        self.list_minus_board_states = []
        self.list_plus_actions = []
        self.list_minus_actions = []

        self.games_wonAI = 0
        self.games_wonAI2 = 0
        self.games_finish_in_draw = 0

        self.current_player = 1

    def reset_game(self):
        self.digraph = nx.DiGraph()
        self.node_counter = 0
        self.num_simulations = 0
        self.digraph.add_node(self.node_counter,
                              num_visit=0,
                              Q=0,
                              u=0,
                              P=1,
                              side=-1,
                              action=0,
                              state=tt.new_board(self.board_size))

        self.node_counter += 1
        self.last_node = 0

    def random_player(self, board_state, _):
        moves = list(tt.available_moves(board_state))
        return random.choice(moves)

    def selection(self, root):
        """
        Select node

        :param root:
        :return:
        """
        children = self.digraph.successors(root)
        values = {}
        has_children = False

        for child_node in children:
            has_children = True
            values[child_node] = self.get_value(child_node)

        if not has_children:
            return True, root

        # Choose the child node that maximizes the expected value
        best_child_node = max(values.items(), key=operator.itemgetter(1))[0]
        return False, best_child_node

    def expansion(self, parent, dict_prob):
        """
        Expand node

        :param parent:
        :return:
        """

        state = self.digraph.node[parent]['state']
        side = self.digraph.node[parent]['side']

        for key, value in dict_prob.items():
            new_state = tt.apply_move(np.copy(state), key, -side)
            self.digraph.add_node(self.node_counter,
                                  num_visit=0,
                                  Q=0,
                                  u=0,
                                  P=value,
                                  side=-side,
                                  action=self.board_size * key[1] + key[0],
                                  state=np.copy(new_state))
            self.digraph.add_edge(parent, self.node_counter)
            logger.debug('Add node ', str(parent), ' -> ', str(self.node_counter))
            logger.debug('Node ', str(self.node_counter), ' -> ', str(new_state.tolist()))
            self.node_counter += 1

    def get_value(self, node):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        """

        num_visit = int(self.digraph.node[node]['num_visit'])
        parent_num_visit = 0
        Q = float(self.digraph.node[node]['Q'])
        P = float(self.digraph.node[node]['P'])
        for key in self.digraph.predecessors(node):
            parent_num_visit = float(self.digraph.node[key]['num_visit'])
            break
        u = self._c_puct * P * math.sqrt(parent_num_visit) / (1 + num_visit)
        self.digraph.node[node]['u'] = u
        return Q + u

    def update(self, node, value):
        """Update node values from leaf evaluation.
        Arguments:
        leaf_value -- the value of subtree evaluation from the current player's perspective.
        """

        num_visit = int(self.digraph.node[node]['num_visit']) + 1
        Q = float(self.digraph.node[node]['Q']) + 1.0 * (value - float(self.digraph.node[node]['Q'])) / num_visit
        self.digraph.node[node]['num_visit'] = num_visit
        self.digraph.node[node]['Q'] = Q

    def update_recursive(self, node, value):
        """Like a call to update(), but applied recursively for all ancestors.
        """

        if node != 0:
            new_node = -1
            for key in self.digraph.predecessors(node):
                new_node = key
                break
            self.update_recursive(new_node, -value)
        self.update(node, value)

    def play_out(self, node):
        """
        Play out algorithm

        :return:
        """

        while True:
            is_leaf, selected_node = self.selection(node)
            node = selected_node
            if is_leaf:
                break

        logger.debug('Leaf_node: ', str(node))
        # state.do_move(action)
        done = tt.has_winner(self.digraph.node[node]['state'], self.winning_length)

        if self.digraph.node[node]['side'] == 1:
            actions, value, _, prob = self.model2.step(np.copy(self.digraph.node[node]['state']), [], done)
        else:
            actions, value, _, prob = self.model.step(np.copy(self.digraph.node[node]['state']), [], done)

        dict_prob = tt.get_available_moves_with_prob(self.digraph.node[node]['state'], prob)
        logger.debug('dict_prob ', str(dict_prob))

        if done[0]:
            value = [self.digraph.node[node]['side']]
        elif len(list(tt.available_moves(self.digraph.node[node]['state']))) == 0:
            value = [0.0]
        else:
            self.expansion(node, dict_prob)

        # Update value and visit count of nodes in this traversal.
        self.update_recursive(node, value[0])

    def get_move_probs(self, state, temp=1e-3):
        """Runs all playouts sequentially and returns the available actions and their corresponding probabilities
        Arguments:
        state -- the current state, including both game state and the current player.
        temp -- temperature parameter in (0, 1] that controls the level of exploration
        Returns:
        the available actions and the corresponding probabilities
        """

        logger.debug('get_move_probs, state: ', str(state.tolist()))
        for node in self.digraph.nodes():
            if np.array_equal(self.digraph.node[node]['state'], state):
                current_node = node
                break
        else:
            raise Exception('Cannot find the board state!')

        logger.debug('get_move_probs, root node: ', str(current_node))
        for n in range(self._n_play_out):
            self.play_out(current_node)

        children = self.digraph.successors(current_node)
        nodes = []
        visits = []
        for child_node in children:
            nodes.append(child_node)
            visits.append(self.digraph.node[child_node]['num_visit'])

        node_probs = ut.softmax(1.0 / temp * np.log(visits))
        return nodes, node_probs

    def get_action(self, state, is_self_play=True):
        available = list(tt.available_moves(state))

        if len(available) > 0:
            nodes, probs = self.get_move_probs(state)
            logger.debug('Prob: ', str(probs))

            if is_self_play:
                node = np.random.choice(nodes, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
            else:
                # with the default temp=1e-3, this is almost equivalent to choosing the move with the highest prob
                node = np.random.choice(nodes, p=probs)

            return node
        else:
            logger.error("WARNING: the board is full")
            return None

    def get_human_action(self):
        location = input("Your move: ")
        if isinstance(location, str):
            location = [int(n, 10) for n in location.split(",")]
        return tuple(location)

    def self_play(self):
        self.reset_game()
        self.num_simulations += 1
        node = 0

        while True:
            state = np.copy(self.digraph.node[node]['state'])
            # logger.info('Node : ', str(node))
            # logger.info('State : ', str(state.tolist()))
            self.show_state(state)

            if len(list(tt.available_moves(state))) == 0:
                self.games_finish_in_draw += 1
                return node, 0

            win = tt.has_winner(state, self.winning_length)
            if win[0]:
                reward = self.digraph.node[node]['side']
                if reward * self.current_player == 1:
                    self.games_wonAI += 1
                else:
                    self.games_wonAI2 += 1
                return node, reward

            node = self.get_action(np.copy(self.digraph.node[node]['state']))

    def start_play(self, is_shown=1):
        """
        start a game between two players
        """

        board_state = tt.new_board(self.board_size)
        player_turn = 1

        while True:
            _available_moves = list(tt.available_moves(board_state))
            if is_shown:
                state = str(board_state).replace(']]', ']').replace(']]', '').replace('[[[[', '').replace('\n', '') \
                    .replace('[[', '\n').replace('[', '').replace(' ', '').replace(']', ' | ').replace(' | \n', '\n')
                print('- ' * 20)
                print(state)
                print('- ' * 20)
            if len(_available_moves) == 0:
                # draw
                if is_shown:
                    print("no moves left, game ended a draw")
                return 0.
            if player_turn > 0:
                node = self.get_action(board_state, False)
                self.last_node = node
                board_state = np.copy(self.digraph.node[self.last_node]['state'])
            else:
                move = self.get_human_action()
                # self.play_out(self.last_node)

                if move not in _available_moves:
                    # if a player makes an invalid move the other player wins
                    if is_shown:
                        print("illegal move ", move)
                    return -player_turn
                board_state = tt.apply_move(board_state, move, player_turn)

            winner = tt.has_winner(board_state, self.winning_length)
            if winner != 0:
                if is_shown:
                    state = str(board_state).replace(']]', ']').replace(']]', '').replace('[[[[', '').replace('\n', '') \
                        .replace('[[', '\n').replace('[', '').replace(' ', '').replace(']', ' | ').replace(' | \n',
                                                                                                           '\n')
                    print('- ' * 20)
                    print(state)
                    print('- ' * 20)
                    print("we have a winner, side: %s" % player_turn)
                return winner
            player_turn = -player_turn

    def get_state_recursive(self, node, is_root=True):
        """
        Get board state recursively
        :param is_root:
        :param node:
        :return:
        """

        if node != 0:
            parent_node = -1
            for key in self.digraph.predecessors(node):
                parent_node = key
                break
            self.get_state_recursive(parent_node, False)
        else:
            return

        if self.digraph.node[node]['side'] == 1:
            self.list_plus_board_states.append(np.copy(self.digraph.node[node]['state']))
            self.list_plus_actions.append(np.copy(self.digraph.node[node]['action']))
        else:
            self.list_minus_board_states.append(np.copy(self.digraph.node[node]['state']))
            self.list_minus_actions.append(np.copy(self.digraph.node[node]['action']))

        # if is_root:
        #     self.list_plus_board_states.append(np.copy(self.digraph.node[node]['state']))
        #     self.list_minus_board_states.append(np.copy(self.digraph.node[node]['state']))
        #     self.list_plus_actions.append(np.copy(self.digraph.node[node]['action']))
        #     self.list_minus_actions.append(np.copy(self.digraph.node[node]['action']))
        # else:
        #     if self.digraph.node[node]['side'] == 1:
        #         self.list_plus_board_states.append(np.copy(self.digraph.node[node]['state']))
        #         self.list_plus_actions.append(np.copy(self.digraph.node[node]['action']))
        #     else:
        #         self.list_minus_board_states.append(np.copy(self.digraph.node[node]['state']))
        #         self.list_minus_actions.append(np.copy(self.digraph.node[node]['action']))

    def get_state(self, node):
        self.list_plus_board_states.clear()
        self.list_minus_board_states.clear()
        self.list_plus_actions.clear()
        self.list_minus_actions.clear()
        self.get_state_recursive(node)
        return self.list_plus_board_states, self.list_minus_board_states, \
               self.list_plus_actions, self.list_minus_actions

    def exchange_models(self):
        tmp = self.model
        self.model = self.model2
        self.model2 = tmp
        self.current_player = -self.current_player

    def show_state(self, state):
        print('- ' * 15 + 'state' + ' -' * 15)
        for x in range(self.board_size):
            print("{0:5}".format(x), end='')
        print('\r\n')
        for i in range(self.board_size):
            print("{0:2d}".format(i), end='')
            for j in range(self.board_size):
                if state[0, i, j, 0] == -1:
                    print('X'.center(5), end='')
                elif state[0, i, j, 0] == 1:
                    print('O'.center(5), end='')
                else:
                    print('_'.center(5), end='')
            print('\r\n')

    def print_statistic(self):
        logger.warn('- ' * 40)
        logger.warn('AI wins in ', str((100 * self.games_wonAI) / (
                self.games_wonAI + self.games_wonAI2 + self.games_finish_in_draw)))
        logger.warn('Random player wins ', str((100 * self.games_wonAI2) / (
                self.games_wonAI + self.games_wonAI2 + self.games_finish_in_draw)))
        logger.warn('Draws ', str((100 * self.games_finish_in_draw) / (
                self.games_wonAI + self.games_wonAI2 + self.games_finish_in_draw)))
        logger.warn('- ' * 40)
        self.games_wonAI = 0
        self.games_wonAI2 = 0
        self.games_finish_in_draw = 0

    def visualization(self, limited_size=300):
        """
        Draw dot graph of Monte Carlo Tree
        :return:
        """
        for node in range(limited_size, len(self.digraph.nodes())):
            self.digraph.remove_node(node)

        pd_tree = nx.nx_pydot.to_pydot(self.digraph)
        for node in pd_tree.get_nodes():
            attr = node.get_attributes()
            try:
                state = attr['state'].replace(']]', ']').replace(']]', '').replace('[[[[', '').replace('\n', '') \
                    .replace('[[', '\n').replace('[', '').replace(' ', '').replace(']', ' | ').replace(' | \n', '\n')
                # state = attr['state'].replace('),', '\n').replace('(', '').replace(')', '').replace(' ', '') \
                #     .replace(',', ' | ')
                n = attr['num_visit']
                Q = attr['Q']
                u = attr['u']
                P = attr['P']
                # side = attr['side']
                # action = attr['action']
                node.set_label(state + '\n' + 'n: ' + n + '\n' + 'Q: ' + Q + '\n' + 'u: ' + u + '\n' + 'P: ' + P)
            except KeyError:
                pass
        pd_tree.write_png('../models/tree.png')


if __name__ == '__main__':
    print('start...')
