import operator
import random

import networkx as nx
import numpy as np

import games.tic_tac_toe_x as tt

EPSILON = 10e-6  # Prevents division by 0 in calculation of UCT


class MonteCarlo:
    def __init__(self):
        self.digraph = nx.DiGraph()
        self.node_counter = 0
        self.computational_budget = 1

        self.num_simulations = 0
        self.board_size = 3
        self.winning_length = 3

        # Constant parameter to weight exploration vs. exploitation for UCT
        self.uct_c = np.sqrt(2)

        self.digraph.add_node(self.node_counter,
                              num=self.node_counter,
                              nw=0,
                              nn=0,
                              uct=0,
                              expanded=False,
                              state=tt.new_board(self.board_size))
        # empty_board_node_id = self.node_counter
        self.node_counter += 1
        self.last_move = None

    def reset_game(self):
        self.last_move = None

    def random_player(self, board_state, _):
        moves = list(tt.available_moves(board_state))
        return random.choice(moves)

    def pure_ai_player(self, board_state, _):
        starting_node = None

        if self.last_move is not None:
            # Check if the starting state is already in the graph as a child of the last move that we made
            for child in self.digraph.successors(self.last_move):
                # Check if the child has the same state attribute as the starting state
                if np.array_equal(self.digraph.node[child]['state'], board_state):
                    # If it does, then check if there is a link between the last move and this child state
                    if self.digraph.has_edge(self.last_move, child):
                        starting_node = child
                        break
            else:
                for node in self.digraph.nodes():
                    if np.array_equal(self.digraph.node[node]['state'], board_state):
                        starting_node = node
        else:
            for node in self.digraph.nodes():
                if np.array_equal(self.digraph.node[node]['state'], board_state):
                    starting_node = node

        selected_node = self.selection(starting_node)
        if selected_node is None:
            self.last_move = None
            move = random.choice(list(tt.available_moves(board_state)))
            return move

        new_child_node = self.expansion(selected_node)
        reward = self.simulation(new_child_node)
        self.backpropagation(new_child_node, reward)
        move, resulting_node = self.best(starting_node)
        self.last_move = resulting_node

        return move

    def get_next_move(self, board_state, _):
        # starting_node = None
        # if self.last_move is not None:
        #     # Check if the starting state is already in the graph as a child of the last move that we made
        #     for child in self.digraph.successors(self.last_move):
        #         # Check if the child has the same state attribute as the starting state
        #         if np.array_equal(self.digraph.node[child]['state'], board_state):
        #             # If it does, then check if there is a link between the last move and this child state
        #             if self.digraph.has_edge(self.last_move, child):
        #                 starting_node = child
        #                 break
        #     else:
        #         for node in self.digraph.nodes():
        #             print(board_state.tolist())
        #             print(self.digraph.node[node])
        #             if np.array_equal(self.digraph.node[node]['state'], board_state):
        #                 starting_node = node
        # else:
        for node in self.digraph.nodes():
            if np.array_equal(self.digraph.node[node]['state'], board_state):
                starting_node = node
                break
        else:
            starting_node = 0

        print('-' * 20 + ' selection ' + '-' * 20)
        print("Running MCTS from this starting state with node id {}:\n{}".format(starting_node,
                                                                                  board_state.tolist()))
        to_be_expanded, selected_node = self.az_selection(starting_node)
        print(str(starting_node) + ' -> select -> ' + str(selected_node) + ': ' + str(to_be_expanded))
        print('selected:\n{}'.format(self.digraph.node[selected_node]['state'].tolist()))

        # self.last_move = selected_node
        return to_be_expanded, selected_node

    def ai_player(self, board_state, _):
        starting_node = None

        if self.last_move is not None:
            # Check if the starting state is already in the graph as a child of the last move that we made
            for child in self.digraph.successors(self.last_move):
                # Check if the child has the same state attribute as the starting state
                if np.array_equal(self.digraph.node[child]['state'], board_state):
                    # If it does, then check if there is a link between the last move and this child state
                    if self.digraph.has_edge(self.last_move, child):
                        starting_node = child
                        break
            else:
                for node in self.digraph.nodes():
                    if np.array_equal(self.digraph.node[node]['state'], board_state):
                        starting_node = node
        else:
            for node in self.digraph.nodes():
                if np.array_equal(self.digraph.node[node]['state'], board_state):
                    starting_node = node

        for i in range(self.computational_budget):
            self.num_simulations += 1

            print("Running MCTS from this starting state with node id {}:\n{}".format(starting_node,
                                                                                      board_state.tolist()))
            # Until computational budget runs out, run simulated trials through the tree:

            # Selection: Recursively pick the best node that maximizes UCT until reaching an unvisited node
            print('-' * 20 + ' selection ' + '-' * 20)
            selected_node = self.selection(starting_node)
            print(str(starting_node) + ' -> select -> ' + str(selected_node))
            print('selected:\n{}'.format(self.digraph.node[selected_node]['state'].tolist()))

            # Check if the selected node is a terminal state, and if so, this iteration is finished
            if tt.has_winner(self.digraph.node[selected_node]['state'], self.winning_length):
                break

            # Expansion: Add a child node where simulation will start
            print('-' * 20 + ' expansion ' + '-' * 20)
            new_child_node = self.expansion(selected_node)
            print('Node chosen for expansion:\n{}'.format(new_child_node))

            # Simulation: Conduct a light playout
            print('-' * 20 + ' simulation ' + '-' * 20)
            reward = self.simulation(new_child_node)
            print('Reward obtained: {}\n'.format(reward))

            # Backpropagation: Update the nodes on the path with the simulation results
            print('-' * 20 + ' backpropagation ' + '-' * 20)
            self.backpropagation(new_child_node, reward)

        move, resulting_node = self.best(starting_node)
        print('MCTS complete. Suggesting move: {}\n'.format(move))

        self.last_move = resulting_node

        # If we won, reset the last move to None for future games
        if tt.has_winner(self.digraph.node[resulting_node]['state'], 3):
            print(self.digraph.node[resulting_node]['state'])
            self.last_move = None

        return move

    def best(self, root):
        """
        Returns the action that results in the child with the highest UCT value
        (An alternative strategy could also be used, where the action leading to
        the child with the most number of visits is chosen
        """
        children = self.digraph.successors(root)

        uct_values = {}
        for child_node in children:
            uct_values[child_node] = self.uct(state=child_node)

        # Choose the child node that maximizes the expected value given by UCT
        # If more than one has the same UCT value then break ties randomly
        best_children = [key for key, val in uct_values.items() if val == max(uct_values.values())]
        idx = np.random.randint(len(best_children))
        best_child = best_children[idx]

        # Determine which action leads to this child
        action = self.digraph.get_edge_data(root, best_child)['action']
        return action, best_child

    def az_selection(self, root):
        children = self.digraph.successors(root)

        uct_values = {}
        has_children = False
        for child_node in children:
            has_children = True
            uct_values[child_node] = self.uct(state=child_node)

        if not has_children:
            return True, root

        # Choose the child node that maximizes the expected value given by UCT
        best_child_node = max(uct_values.items(), key=operator.itemgetter(1))[0]
        return False, best_child_node

    def selection(self, root):
        """
        Starting at root, recursively select the best node that maximizes UCT
        until a node is reached that has no explored children
        Keeps track of the path traversed by adding each node to path as
        it is visited
        :return: the node to expand
        """
        # In the case that the root node is not in the graph, add it
        if root not in self.digraph.nodes():
            self.digraph.add_node(self.node_counter,
                                  num=self.node_counter,
                                  nw=0,
                                  nn=0,
                                  uct=0,
                                  expanded=False,
                                  state=root)
            self.node_counter += 1
            return root
        elif not self.digraph.node[root]['expanded']:
            print('root in digraph but not expanded')
            return root  # This is the node to expand
        else:
            print('root expanded, move on to a child')
            # Handle the general case
            children = self.digraph.successors(root)
            uct_values = {}
            for child_node in children:
                uct_values[child_node] = self.uct(state=child_node)

            # Choose the child node that maximizes the expected value given by UCT
            best_child_node = max(uct_values.items(), key=operator.itemgetter(1))[0]

            return self.selection(best_child_node)

    def az_expansion(self, parent, child):
        print('az_exp')
        print(parent.tolist())
        print(child.tolist())

        if np.array_equal(self.digraph.node[0]['state'], child):
            return

        parent_node = -1
        for node in self.digraph.nodes():
            # print(node)
            # print(self.digraph.node[node]['state'].tolist())
            if np.array_equal(self.digraph.node[node]['state'], parent):
                parent_node = node
                break

        if parent_node < 0:
            print('Cannot find parent node!')
            return

        child_node = -1
        for node in self.digraph.nodes():
            # print(node)
            # print(self.digraph.node[node]['state'].tolist())
            if np.array_equal(self.digraph.node[node]['state'], child):
                child_node = node
                break

        if child_node < 0:
            self.digraph.add_node(self.node_counter,
                                  num=self.node_counter,
                                  nw=0,
                                  nn=0,
                                  uct=0,
                                  expanded=False,
                                  state=np.copy(child))
            self.digraph.add_edge(parent_node, self.node_counter)
            print('Add node %d -> %d' % (parent_node, self.node_counter))
            print(self.node_counter)
            print('node', self.digraph.node[self.node_counter]['state'].tolist())
            self.node_counter += 1
        else:
            print('Find child node!')
            self.digraph.add_edge(parent_node, child_node)
            print('Add node %d -> %d' % (parent_node, child_node))

        # for edge in self.digraph.edges:
        #     print(edge)
        return child

    def expansion(self, node):
        # As long as this node has at least one unvisited child, choose a legal move
        legal_moves = list(tt.available_moves(self.digraph.node[node]['state']))
        print('Legal moves: {}'.format(legal_moves))

        # Select the next unvisited child with uniform probability
        unvisited_children = []
        corresponding_actions = []

        for move in legal_moves:
            print('adding to expansion analysis with: {}'.format(move))
            child_state = tt.apply_move(self.digraph.node[node]['state'], move, 0)

            in_children = False
            for child_node in self.digraph.successors(node):
                if np.array_equal(self.digraph.node[child_node]['state'], child_state):
                    in_children = True

            if not in_children:
                unvisited_children.append(child_state)
                corresponding_actions.append(move)

        print('unvisited children: {}'.format(len(unvisited_children)))
        if len(unvisited_children) > 0:
            idx = np.random.randint(len(unvisited_children))
            child, move = unvisited_children[idx], corresponding_actions[idx]

            print('Add node %d -> %d' % (node, self.node_counter))
            print(child)
            self.digraph.add_node(self.node_counter,
                                  num=self.node_counter,
                                  nw=0,
                                  nn=0,
                                  uct=0,
                                  expanded=False,
                                  state=np.copy(child))
            self.digraph.add_edge(node, self.node_counter, action=move)
            child_node_id = self.node_counter
            self.node_counter += 1
        else:
            return node

        # If all legal moves are now children, mark this node as expanded.
        num_children = 0
        for _ in self.digraph.successors(node):
            num_children += 1

        if num_children == len(legal_moves):
            self.digraph.node[node]['expanded'] = True
            print('node is expanded')

        return child_node_id

    def simulation(self, node):
        """
        Conducts a light playout from the specified node
        :return: The reward obtained once a terminal state is reached
        """
        # random_policy = RandomPolicy()
        current_state = self.digraph.node[node]['state']
        while not tt.has_winner(current_state, self.winning_length):
            available = list(tt.available_moves(current_state))
            if len(available) == 0:
                return 0
            move = random.choice(available)
            current_state = tt.apply_move(current_state, move, 0)

        return tt.has_winner(current_state, self.winning_length)

    def az_backup(self, list_ai_node, reward):
        for state in list_ai_node:
            for node in self.digraph.nodes():
                if np.array_equal(self.digraph.node[node]['state'], state):
                    self.digraph.node[node]['nn'] += 1
                    self.digraph.node[node]['nw'] += reward
                    break

    def backpropagation(self, last_visited, reward):
        """
        Walk the path upwards to the root, incrementing the
        'nn' and 'nw' attributes of the nodes along the way
        """
        current = last_visited
        while True:
            self.digraph.node[current]['nn'] += 1
            self.digraph.node[current]['nw'] += reward

            print('Updating to n={} and w={}:\n{}'.format(self.digraph.node[current]['nn'],
                                                          self.digraph.node[current]['nw'],
                                                          self.digraph.node[current]['state']))

            # Terminate when we reach the empty board
            if np.array_equal(self.digraph.node[current]['state'], tt.new_board(self.board_size)):
                break
            # Todo:
            # Does this handle the necessary termination conditions for both 'X' and 'O'?
            # As far as we can tell, it does

            # Will throw an IndexError when we arrive at a node with no predecessors
            # Todo: see if this additional check is no longer necessary
            print('predecessors')
            for key in self.digraph.predecessors(current):
                print(key)
                print()
            try:
                print('predecessors')
                current = next(self.digraph.predecessors(current))
            except StopIteration:
                break

    def uct(self, state):
        """
        Returns the expected value of a state, calculated as a weighted sum of
        its exploitation value and exploration value
        """
        n = self.digraph.node[state]['nn']  # Number of plays from this node
        w = self.digraph.node[state]['nw']  # Number of wins from this node
        t = self.num_simulations
        c = self.uct_c
        epsilon = EPSILON

        exploitation_value = w / (n + epsilon)
        exploration_value = c * np.sqrt(np.log(t) / (n + epsilon))
        print('exploration_value: {}'.format(exploration_value))

        value = exploitation_value + exploration_value

        print(exploitation_value)
        print(exploration_value)
        print('UCT value {:.3f} for state:\n'.format(value))
        print(state)

        self.digraph.node[state]['uct'] = value

        return value

    def train(self, times):
        for i in range(times):
            tt.play_game(self.ai_player, self.random_player, self.board_size, self.winning_length, log=False)

    def play_game(self, board):
        print('Start play', board.tolist())
        self.num_simulations += 1
        list_board = [board]
        list_ai_board = []

        while True:
            win = tt.has_winner(board, self.winning_length)
            print(board.tolist())
            print('win', win)
            if win[0]:
                return board, list_board, list_ai_board
            if len(list(tt.available_moves(board))) == 0:
                return board, list_board, list_ai_board

            to_be_expanded, selected_node = self.get_next_move(board, None)

            if not to_be_expanded:
                list_board.append(self.digraph.node[selected_node]['state'])
                list_ai_board.append(self.digraph.node[selected_node]['state'])

                board = np.copy(self.digraph.node[selected_node]['state'])
                win = tt.has_winner(board, self.winning_length)
                if win[0]:
                    return board, list_board, list_ai_board
                if len(list(tt.available_moves(board))) == 0:
                    return board, list_board, list_ai_board
                print(len(list(tt.available_moves(board))))

                moves = list(tt.available_moves(board))
                move = random.choice(moves)
                board = tt.apply_move(board, move, -1)
            else:
                return board, list_board, list_ai_board

    def play_against_random(self, play_round=20):
        win_count = 0
        record = []
        for i in range(play_round):
            result = tt.play_game(self.ai_player, self.random_player, log=False)
            record.append(result)
            if result == 1:
                win_count += 1
        print(record)
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
