import math
import operator
import random

import networkx as nx
import numpy as np

import AlphaGomoku.games.tic_tac_toe_x as tt
from AlphaGomoku.common import logger

EPSILON = 10e-6  # Prevents division by 0 in calculation of UCT


class MonteCarlo:
    def __init__(self, env, model):
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
                              state=tt.new_board(self.board_size))

        self.node_counter += 1
        self.last_move = None

        self.model = model
        self._c_puct = 5
        self._n_play_out = 200
        self.list_plus_board_states = []
        self.list_minus_board_states = []

    def reset_game(self):
        self.last_move = None

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
        Q = float(self.digraph.node[node]['Q'])
        P = float(self.digraph.node[node]['P'])
        u = self._c_puct * P * math.sqrt(num_visit) / (1 + num_visit)
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
            rs = np.copy(self.digraph.node[node]['state'])
            np.place(rs, rs == 1, 2)
            np.place(rs, rs == -1, 1)
            np.place(rs, rs == 2, -1)
            actions, value, _, prob = self.model.step(np.copy(rs), [], done)
        else:
            actions, value, _, prob = self.model.step(np.copy(self.digraph.node[node]['state']), [], done)

        dict_prob = tt.get_available_moves_with_prob(self.digraph.node[node]['state'], prob)
        logger.debug('dict_prob ', str(dict_prob))

        if not done[0]:
            self.expansion(node, dict_prob)
        else:
            if len(list(tt.available_moves(self.digraph.node[node]['state']))) == 0:
                value = 0.0
            else:
                value = 1

        # Update value and visit count of nodes in this traversal.
        self.update_recursive(node, value)

    def softmax(self, x):
        probs = np.exp(x - np.max(x))
        probs /= np.sum(probs)
        return probs

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

        node_probs = self.softmax(1.0 / temp * np.log(visits))
        return nodes, node_probs

    def get_action(self, state, is_self_play=True):
        available = list(tt.available_moves(state))

        if len(available) > 0:
            nodes, probs = self.get_move_probs(state)

            if is_self_play:
                node = np.random.choice(nodes, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
            else:
                # with the default temp=1e-3, this is almost equivalent to choosing the move with the highest prob
                node = np.random.choice(nodes, p=probs)

            return node
        else:
            logger.error("WARNING: the board is full")
            return None

    def self_play(self):
        self.num_simulations += 1
        node = 0

        while True:
            state = np.copy(self.digraph.node[node]['state'])
            logger.info('Node : ', str(node))
            logger.info('State : ', str(state.tolist()))

            if len(list(tt.available_moves(state))) == 0:
                return node, 0
            win = tt.has_winner(state, self.winning_length)
            if win[0]:
                return node, 1

            node = self.get_action(np.copy(self.digraph.node[node]['state']))

    def get_state_recursive(self, node, is_root=True):
        """
        Get board state recursively
        :param node:
        :return:
        """

        if node != 0:
            parent_node = -1
            for key in self.digraph.predecessors(node):
                parent_node = key
                break
            self.get_state_recursive(parent_node, False)

        if is_root:
            self.list_plus_board_states.append(np.copy(self.digraph.node[node]['state']))
            self.list_minus_board_states.append(np.copy(self.digraph.node[node]['state']))
        else:
            if self.digraph.node[node]['side'] == 1:
                self.list_plus_board_states.append(np.copy(self.digraph.node[node]['state']))
            else:
                self.list_minus_board_states.append(np.copy(self.digraph.node[node]['state']))

    def get_state(self, node):
        self.list_plus_board_states.clear()
        self.list_minus_board_states.clear()
        self.get_state_recursive(node)
        return self.list_plus_board_states, self.list_minus_board_states

    def play_against_random(self, play_round=20):
        win_count = 0
        record = []
        for i in range(play_round):
            result = tt.play_game(self.ai_player, self.random_player, log=False)
            record.append(result)
            if result == 1:
                win_count += 1
        print('Win rate: %f' % (win_count / float(play_round)))

    def visualization(self):
        """
        Draw dot graph of Monte Carlo Tree
        :return:
        """
        pd_tree = nx.nx_pydot.to_pydot(self.digraph)
        for node in pd_tree.get_nodes():
            attr = node.get_attributes()
            try:
                state = attr['state'].replace(']]', ']').replace(']]', '').replace('[[[[', '').replace('\n', '') \
                    .replace('[[', '\n').replace('[', '').replace(' ', '').replace(']', ' | ').replace(' | \n', '\n')
                # state = attr['state'].replace('),', '\n').replace('(', '').replace(')', '').replace(' ', '') \
                #     .replace(',', ' | ')
                w = attr['nw']
                n = attr['nn']
                num = attr['num']
                uct = attr['uct'][:4]
                node.set_label(state + '\n' + w + '/' + n + '\n' + uct + '\n' + num)
            except KeyError:
                pass
        pd_tree.write_png('tree.png')


if __name__ == '__main__':
    print('start...')
    mc = MonteCarlo()
    mc.train(5)
    # mc.visualization()
    # mc.play_against_random(5)
