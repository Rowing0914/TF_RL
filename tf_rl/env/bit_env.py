import numpy as np


class Bit_Env:
    def __init__(self, size):
        self.size = size

    def reset(self):
        state = np.random.randint(2, size=self.size)
        goal = np.random.randint(2, size=self.size)

        while np.sum(state == goal) == self.size:
            goal = np.random.randint(2, size=self.size)
        self.goal = goal
        self.state = state
        return state

    def step(self, action):
        self.state[action] = 1 - self.state[action]
        return self.compute_reward(self.state, self.goal)

    def compute_reward(self, state, goal):
        if not self.check_success(state, goal):
            return state, -1, False, ""
        else:
            return state, 0, True, ""

    def check_success(self, state, goal):
        return np.sum(state == goal) == self.size
