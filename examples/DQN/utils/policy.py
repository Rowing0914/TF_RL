import numpy as np

class EpsilonGreedyPolicy_eager:
    """ Epsilon Greedy Policy for eager execution """

    def __init__(self, dim_action, epsilon_fn):
        self._epsilon_fn = epsilon_fn
        self._dim_action = dim_action

    def select_action(self, q_value_fn, state, epsilon=None):
        _epsilon = self.current_epsilon() if epsilon is None else epsilon
        if np.random.uniform() < _epsilon:
            action = np.random.randint(self._dim_action)
        else:
            q_values = q_value_fn(state).numpy()
            action = np.argmax(q_values)
        return action

    def current_epsilon(self):
        return self._epsilon_fn().numpy()