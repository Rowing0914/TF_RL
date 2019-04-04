import random
import numpy as np
from collections import deque

class ReplayBuffer:
    """
    Experience Replay Memory
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def store(self, state, action, reward, next_state, done):
        """
        Store the values in experience memory

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        sample a batch of experiences from the memory

        :param batch_size:
        :return:
        """
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
