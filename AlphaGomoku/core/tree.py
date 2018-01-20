import math
import operator

import AlphaGomoku.core.constants as cons
from AlphaGomoku.common import logger


class TreeNode(object):
    """
    Tree node to build the Monte Carlo Search Tree
    """

    def __init__(self, parent=None, side=-1, p=1, level=0, action=-1):
        """
        Initialization
        :param parent:
        :param side:
        :param p:
        :param level:
        :param action:
        """
        self._parent = parent
        self._num_visit = 0
        self._Q = 0
        self._u = 0
        self._P = p
        self._side = side
        self._action = action
        self._level = level
        self._children = {}

    def selection(self):
        """
        Select best node from children nodes
        :return:
        """
        if self._children == {}:
            return True, None

        values = {}
        for _, child in self._children.items():
            values[child] = child.get_value()
        # Choose the child node that maximizes the expected value
        best_child = max(values.items(), key=operator.itemgetter(1))[0]
        return False, best_child

    def expansion(self, dict_prob):
        """
        Expand node based on probabilities
        :param dict_prob:
        :return:
        """
        for action, prob in dict_prob.items():
            self._children[action] = TreeNode(parent=self, side=-self._side, p=prob, level=self._level + 1,
                                              action=action)
            logger.debug('Add node level = ', str(self._level), ', action = ', str(self._action))

    def get_visit_num(self):
        """
        Get number of visit the current node
        :return:
        """
        return self._num_visit

    def get_children(self):
        """
        Get all children
        :return:
        """
        return self._children

    def get_value(self):
        """
        Calculate the value of each node
        :return:
        """
        num_visit = self._parent.get_visit_num()
        self._u = cons.C_PUCT * self._P * math.sqrt(num_visit) / (1 + self._num_visit)
        return self._Q + self._u

    def update(self, value):
        """
        Update after visiting
        :param value:
        :return:
        """
        self._num_visit += 1
        self._Q += 1.0 * (value - self._Q) / self._num_visit

    def update_recursive(self, value):
        """
        Update value recursively
        :param node:
        :param value:
        :return:
        """
        if self._parent:
            self._parent.update_recursive(-value)
        self.update(value)

    def is_root(self):
        """
        Check whether current node is the root node
        :return:
        """
        return self._parent is None
