import lasagne
import numpy as np
import theano
import theano.tensor as T

from AlphaGomoku.core import utils


class PolicyValueNet:
    """policy-value network """

    def __init__(self, board_width, board_height, net_params=None):
        self.board_width = board_width
        self.board_height = board_height
        self.learning_rate = T.scalar('learning_rate')
        self.l2_const = 1e-4  # coef of l2 penalty
        self.create_policy_value_net()
        self._loss_train_op()
        if net_params:
            lasagne.layers.set_all_param_values([self.policy_net, self.value_net], net_params)

    def create_policy_value_net(self):
        """create the policy value network """
        self.state_input = T.tensor4('state')
        self.winner = T.vector('winner')
        self.mcts_probs = T.matrix('mcts_probs')
        network = lasagne.layers.InputLayer(shape=(None, 4, self.board_width, self.board_height),
                                            input_var=self.state_input)
        # conv layers
        network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3, 3), pad='same')
        network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(3, 3), pad='same')
        network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(3, 3), pad='same')
        # action policy layers
        policy_net = lasagne.layers.Conv2DLayer(network, num_filters=4, filter_size=(1, 1))
        self.policy_net = lasagne.layers.DenseLayer(policy_net, num_units=self.board_width * self.board_height,
                                                    nonlinearity=lasagne.nonlinearities.softmax)
        # state value layers
        value_net = lasagne.layers.Conv2DLayer(network, num_filters=2, filter_size=(1, 1))
        value_net = lasagne.layers.DenseLayer(value_net, num_units=64)
        self.value_net = lasagne.layers.DenseLayer(value_net, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)
        # get action probs and state score value
        self.action_probs, self.value = lasagne.layers.get_output([self.policy_net, self.value_net])
        self.policy_value = theano.function([self.state_input], [self.action_probs, self.value],
                                            allow_input_downcast=True)

    def policy_value_fn(self, current_state, legal_positions):
        act_probs, value = self.policy_value(current_state.reshape(-1, 4, self.board_width, self.board_height))
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value[0][0]

    def _loss_train_op(self):
        """
        Three loss terms：
        loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        """
        params = lasagne.layers.get_all_params([self.policy_net, self.value_net], trainable=True)
        value_loss = lasagne.objectives.squared_error(self.winner, self.value.flatten())
        policy_loss = lasagne.objectives.categorical_crossentropy(self.action_probs, self.mcts_probs)
        l2_penalty = lasagne.regularization.apply_penalty(params, lasagne.regularization.l2)
        self.loss = lasagne.objectives.aggregate(value_loss + policy_loss, mode='mean') + self.l2_const * l2_penalty
        # policy entropy，for monitoring only
        self.entropy = -T.mean(T.sum(self.action_probs * T.log(self.action_probs + 1e-10), axis=1))
        # get the train op
        updates = lasagne.updates.adam(self.loss, params, learning_rate=self.learning_rate)
        self.train_step = theano.function([self.state_input, self.mcts_probs, self.winner, self.learning_rate],
                                          [self.loss, self.entropy], updates=updates, allow_input_downcast=True)

    def get_policy_param(self):
        net_params = lasagne.layers.get_all_param_values([self.policy_net, self.value_net])
        return net_params


class PolicyValueNetNumpy:
    """policy-value network in numpy """

    def __init__(self, board_width, board_height, net_params):
        self.board_width = board_width
        self.board_height = board_height
        self.params = net_params

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = board.availables
        current_state = board.current_state()

        X = current_state.reshape(-1, 4, self.board_width, self.board_height)
        # first 3 conv layers with ReLu nonlinearity
        for i in [0, 2, 4]:
            X = utils.relu(utils.conv_forward(X, self.params[i], self.params[i + 1]))
        # policy head
        X_p = utils.relu(utils.conv_forward(X, self.params[6], self.params[7], padding=0))
        X_p = utils.fc_forward(X_p.flatten(), self.params[8], self.params[9])
        act_probs = utils.softmax(X_p)
        # value head
        X_v = utils.relu(utils.conv_forward(X, self.params[10], self.params[11], padding=0))
        X_v = utils.relu(utils.fc_forward(X_v.flatten(), self.params[12], self.params[13]))
        value = np.tanh(utils.fc_forward(X_v, self.params[14], self.params[15]))[0]

        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value
