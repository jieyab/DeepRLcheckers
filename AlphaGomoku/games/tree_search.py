import operator
import random

import networkx as nx
import numpy as np

import tic_tac_toe as tt

EPSILON = 10e-6  # Prevents division by 0 in calculation of UCT


class MonteCarlo:
    def __init__(self):
        self.digraph = nx.DiGraph()
        self.node_counter = 0

        self.num_simulations = 0
        # Constant parameter to weight exploration vs. exploitation for UCT
        self.uct_c = np.sqrt(2)

        self.digraph.add_node(self.node_counter,
                              nw=0,
                              nn=0,
                              uct=0,
                              expanded=False,
                              state=tt.new_board())
        # empty_board_node_id = self.node_counter
        self.node_counter += 1
        self.last_move = None

        self.computational_budget = 5

    def reset_game(self):
        self.last_move = None

    def random_player(self, board_state, _):
        moves = list(tt.available_moves(board_state))
        return random.choice(moves)

    def ai_player(self, board_state, side):
        # starting_state = copy.deepcopy(starting_state)

        starting_node = None

        if self.last_move is not None:
            # Check if the starting state is already in the graph as a child of the last move that we made
            exists = False
            for child in self.digraph.successors(self.last_move):
                # Check if the child has the same state attribute as the starting state
                if self.digraph.node[child]['state'] == board_state:
                    # If it does, then check if there is a link between the last move and this child state
                    if self.digraph.has_edge(self.last_move, child):
                        exists = True
                        starting_node = child
            if not exists:
                # If it wasn't found, then add the starting state and an edge to it from the last move
                self.digraph.add_node(self.node_counter,
                                      nw=0,
                                      nn=0,
                                      uct=0,
                                      expanded=False,
                                      state=board_state)
                self.digraph.add_edge(self.last_move, self.node_counter)
                starting_node = self.node_counter
                self.node_counter += 1
        else:
            for node in self.digraph.nodes():
                if self.digraph.node[node]['state'] == board_state:
                    starting_node = node

        for i in range(self.computational_budget):
            self.num_simulations += 1

            print("Running MCTS from this starting state with node id {}:\n{}".format(starting_node,
                                                                                      board_state))
            # Until computational budget runs out, run simulated trials through the tree:

            # Selection: Recursively pick the best node that maximizes UCT until reaching an unvisited node
            print('-' * 20 + ' selection ' + '-' * 20)
            selected_node = self.selection(starting_node)
            print str(starting_node) + ' -> select -> ' + str(selected_node)
            print('selected:\n{}'.format(self.digraph.node[selected_node]['state']))

            # Check if the selected node is a terminal state, and if so, this iteration is finished
            if tt.has_winner(self.digraph.node[selected_node]['state']):
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
        if tt.has_winner(self.digraph.node[resulting_node]['state']):
            self.last_move = None

        return move

    def best(self, root):
        """
        Returns the action that results in the child with the highest UCT value
        (An alternative strategy could also be used, where the action leading to
        the child with the most number of visits is chosen
        """
        # Todo: explore various strategies for choosing the best action
        children = self.digraph.successors(root)

        uct_values = {}
        for child_node in children:
            uct_values[child_node] = self.uct(child_node)

        # Choose the child node that maximizes the expected value given by UCT
        # If more than one has the same UCT value then break ties randomly
        best_children = [key for key, val in uct_values.iteritems() if val == max(uct_values.values())]
        idx = np.random.randint(len(best_children))
        best_child = best_children[idx]

        # Determine which action leads to this child
        action = self.digraph.get_edge_data(root, best_child)['action']
        return action, best_child

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

    def expansion(self, node):
        # As long as this node has at least one unvisited child, choose a legal move
        children = self.digraph.successors(node)
        legal_moves = list(tt.available_moves(self.digraph.node[node]['state']))
        print('Legal moves: {}'.format(legal_moves))

        # Select the next unvisited child with uniform probability
        unvisited_children = []
        corresponding_actions = []

        for move in legal_moves:
            print('adding to expansion analysis with: {}'.format(move))
            child_state = tt.apply_move(self.digraph.node[node]['state'], move, 0)

            in_children = False
            for child_node in children:
                if self.digraph.node[child_node]['state'] == child_state:
                    in_children = True

            if not in_children:
                unvisited_children.append(child_state)
                corresponding_actions.append(move)

        # Todo: why is it possible for there to be no unvisited children?
        print('unvisited children: {}'.format(len(unvisited_children)))
        if len(unvisited_children) > 0:
            idx = np.random.randint(len(unvisited_children))
            child, move = unvisited_children[idx], corresponding_actions[idx]

            print 'Add node %d -> %d' % (node, self.node_counter)
            print child
            self.digraph.add_node(self.node_counter,
                                  nw=0,
                                  nn=0,
                                  uct=0,
                                  expanded=False,
                                  state=child)
            self.digraph.add_edge(node, self.node_counter, action=move)
            child_node_id = self.node_counter
            self.node_counter += 1
        else:
            # Todo:
            # Is this the correct behavior? The issue is, it was getting to the expansion
            # expansion method with nodes that were already expanded for an unknown reason,
            # so here we return the node that was passed. Maybe there is a case where a
            # node had been expanded but not yet marked as expanded until it got here.
            return node

        # If all legal moves are now children, mark this node as expanded.
        length_of_children = 0
        while True:
            try:
                children.next()
                length_of_children += 1
            except StopIteration:
                break

        if length_of_children + 1 == len(legal_moves):
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
        while not tt.has_winner(current_state):
            available = list(tt.available_moves(current_state))
            if len(available) == 0:
                return 0
            move = random.choice(available)
            current_state = tt.apply_move(current_state, move, 0)

        return tt.has_winner(current_state)

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
            if self.digraph.node[current]['state'] == tt.new_board():
                break
            # Todo:
            # Does this handle the necessary termination conditions for both 'X' and 'O'?
            # As far as we can tell, it does

            # Will throw an IndexError when we arrive at a node with no predecessors
            # Todo: see if this additional check is no longer necessary
            try:
                current = self.digraph.predecessors(current).next()
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

        print('UCT value {:.3f} for state:\n{}'.format(value, state))

        self.digraph.node[state]['uct'] = value

        return value

    def train(self, times):
        for i in xrange(times):
            tt.play_game(self.ai_player, self.ai_player, log=False)

    def play_against_random(self, play_round=20):
        win_count = 0
        record = []
        for i in xrange(play_round):
            result = tt.play_game(self.ai_player, self.random_player, log=False)
            record.append(result)
            if result == 1:
                win_count += 1
        print record
        print 'Win rate: %f' % (win_count / float(play_round))

    def visualization(self):
        # nx.draw(self.digraph, with_labels=True)
        # plt.show()
        pd_tree = nx.nx_pydot.to_pydot(self.digraph)
        for node in pd_tree.get_nodes():
            attr = node.get_attributes()
            try:
                state = attr['state'].replace('),', '\n').replace('(', '').replace(')', '').replace(' ', '')\
                    .replace(',', ' | ')
                w = attr['nw']
                n = attr['nn']
                uct = attr['uct'][:4]
                node.set_label(state + '\n' + w + '/' + n + '\n' + uct)
            except KeyError:
                pass
        pd_tree.write_png('tree.png')


if __name__ == '__main__':
    print 'start...'
    mc = MonteCarlo()
    mc.train(1)
    mc.visualization()
    # mc.play_against_random()
