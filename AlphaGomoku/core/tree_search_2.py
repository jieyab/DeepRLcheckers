from AlphaGomoku.core.tree import TreeNode


class MonteCarlo2:

    def __init__(self, board_size, winning_length):
        # Initialization
        self._board_size = board_size
        self._winning_length = winning_length
        self._root = TreeNode()

    def play_out(self, state):
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
                value = [0.0]
            else:
                value = [1]

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
                return node, self.digraph.node[node]['side']

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
                        .replace('[[', '\n').replace('[', '').replace(' ', '').replace(']', ' | ').replace(' | \n', '\n')
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
