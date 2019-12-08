# Appendix C in https://arxiv.org/pdf/1708.02596.pdf

import numpy as np


def cheetah_cost_fn(action, next_state):
    """ Cost function for HalfCheetah """
    if len(next_state.shape) <= 2:
        xvelafter = next_state[:, 9]
        reward = xvelafter - 0.05 * np.sqrt(np.sum(np.square(action), axis=-1))
    else:
        xvelafter = next_state[:, :, 9]
        reward = xvelafter - 0.05 * np.sqrt(np.sum(np.square(action), axis=-1))
        reward = np.sum(reward, axis=-1)
    return reward
