import numpy as np
import torch


class EpsilonGreedyPolicy_eager:
    """ Epsilon Greedy Policy for eager execution """

    def __init__(self, num_action, epsilon_fn):
        self._epsilon_fn = epsilon_fn
        self._num_action = num_action

    def select_action(self, q_value_fn, state, epsilon=None):
        _epsilon = self.current_epsilon() if epsilon is None else epsilon
        if np.random.random() < _epsilon:
            action = np.random.randint(self._num_action)
        else:
            q_values = q_value_fn(state).numpy()
            action = np.argmax(q_values)
        return action

    def current_epsilon(self):
        return self._epsilon_fn().numpy()


class EpsilonGreedyPolicy_torch:
    """ Epsilon Greedy Policy for eager execution """

    def __init__(self, num_action, epsilon_fn):
        self._epsilon_fn = epsilon_fn
        self._num_action = num_action

    def select_action(self, q_value_fn, state, ts, epsilon=None):
        _epsilon = self.current_epsilon(ts) if epsilon is None else epsilon
        if np.random.random() < _epsilon:
            action = np.random.randint(self._num_action)
        else:
            q_values = q_value_fn(state).cpu().numpy().squeeze()
            action = np.argmax(q_values)
        return action

    def current_epsilon(self, ts):
        return self._epsilon_fn.get_value(ts)