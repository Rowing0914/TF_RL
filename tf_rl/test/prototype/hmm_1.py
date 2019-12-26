""" Hidden Markov Models(Tabular example)

Notation:
    - t: time step
    - pi: prob of init hidden states
    - A: hidden state transition matrix
    - B: emission matrix(state-observation likelihood)
    - h: a sequence of hidden states
    - o: a sequence of observations
"""

import numpy as np


def Forward(A, B, pi, h, o):
    """ Forward algorithm: computes alpha """
    horizon = o.shape[0]  # length of horizon
    num_states = A.shape[0]  # num of hidden states

    result = np.zeros(shape=(num_states, horizon))

    for t in range(horizon):
        if t == 0:
            # alpha_0 = p(h_0) x p(o_0 | h_0)  -> 1 x num_states
            result[:, t] = pi * B[:, o[t]]
        else:
            # alpha_t = p(o_t | h_t) x sum alpha_t-1 * p(h_t | h_t-1) -> 1 x num_states
            for n in range(num_states):
                result[n, t] = B[n, o[t]] * np.dot(result[:, t-1], A[:, n])
    return result


def Viterbi(A, B, pi, h, o):
    """ Viterbi algorithm """
    horizon = o.shape[0]  # length of horizon
    num_states = A.shape[0]  # num of hidden states

    result = np.zeros(shape=(num_states, horizon))

    for t in range(horizon):
        if t == 0:
            # v_0 = p(h_0) x p(o_0 | h_0)  -> 1 x num_states
            result[:, t] = pi * B[:, o[t]]
        else:
            # v_t = max(p(o_t | h_t) x v_t-1 x p(h_t | h_t-1)) -> 1 x num_states
            for n in range(num_states):
                result[n, t] = B[n, o[t]] * np.max(result[:, t - 1] * A[:, n])
    return result


if __name__ == '__main__':
    # RHUL case
    # A = np.array([[0.8, 0.2],
    #               [0.3, 0.7]])
    # B = np.array([[0.3, 0.4, 0.1, 0.2],
    #               [0.2, 0.2, 0.3, 0.3]])
    # pi = np.array([0.4, 0.6])
    # h = np.array([2, 1, 2]) - 1
    # o = np.array([4, 1, 2]) - 1

    # Stanford case
    A = np.array([[0.6, 0.4],
                  [0.5, 0.5]])
    B = np.array([[0.2, 0.4, 0.4],
                  [0.5, 0.4, 0.1]])
    pi = np.array([0.8, 0.2])
    h = np.array([2, 1, 2]) - 1
    o = np.array([3, 1, 3]) - 1

    # Test for Forward algorithm
    alpha = Forward(A, B, pi, h, o)
    answer = np.array([[0.32, 0.0404, 0.023496],
                       [0.02, 0.069, 0.005066]])
    assert (np.isclose(a=alpha, b=answer)).all(), "Wrong implementation of the forward algorithm"

    # Test for Viterbi algorithm
    viterbi = Viterbi(A, B, pi, h, o)
    print(viterbi)