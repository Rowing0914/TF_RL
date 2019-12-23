import numpy as np


class QAgent(object):
    """ Taken from tf_rl/examples/Q_Learning """
    def __init__(self, num_state, num_action, gamma=0.95):
        self._num_action = num_action
        self._gamma = gamma
        self.Q = np.zeros((num_state, num_action))

    def select_action(self, state, epsilon=1.0):
        if np.random.random() <= epsilon:
            return np.random.choice(a=np.arange(self._num_action), p=np.ones(self._num_action) / self._num_action)
        else:
            return np.argmax(self.Q[state])

    def select_action_eval(self, state):
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, alpha):
        # === I don't know why tho, self.gamma = 0.99 does not converge in Q-learning ===
        # self.Q[state][action] += alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
        self.Q[state][action] += alpha * (reward + 1. * np.max(self.Q[next_state]) - self.Q[state][action])