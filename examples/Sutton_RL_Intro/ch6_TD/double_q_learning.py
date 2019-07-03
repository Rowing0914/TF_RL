# 6.7 Maximization Bias and Double Learning

from collections import defaultdict
import numpy as np
import sys
import itertools

if "../" not in sys.path:
    sys.path.append("../")

from libs.envs.windy_gridworld import WindyGridworldEnv
from libs.plot import plot_result, compare_plots
from ch6_TD.sarsa import Sarsa
from ch6_TD.q_learning import Q_learning


def make_epsilon_greedy_policy(Q1, Q2, epsilon, nA):
    def policy(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q1[observation] + Q2[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy


def Double_Q_learning(env, Q1, Q2, alpha=0.5, discount_factor=1.0, epsilon=0.1, num_episodes=1000):
    stats = np.zeros((num_episodes, 2))
    policy = make_epsilon_greedy_policy(Q1, Q2, epsilon, env.nA)

    for i in range(num_episodes):
        # this satisfies the exploraing start condition
        state = env.reset()
        action = np.random.choice(np.arange(env.nA), p=policy(state))
        # generate an episode
        for t in itertools.count():
            next_state, reward, done, _ = env.step(action)

            if np.random.random() >= 0.5:
                next_action = np.argmax(Q1[next_state])
                Q1[state][action] += alpha * (
                            reward + discount_factor * Q1[next_state][next_action] - Q1[state][action])
            else:
                next_action = np.argmax(Q2[next_state])
                Q2[state][action] += alpha * (
                            reward + discount_factor * Q2[next_state][next_action] - Q2[state][action])

            stats[i, 0] = t
            stats[i, 1] += reward

            if done:
                break

            state = next_state
            action = next_action
    return [Q1, Q2], stats


if __name__ == '__main__':
    NUM_EPISODES = 100
    DISCOUNT_FACTOR = 1.0
    ALPHA = 0.5

    env = WindyGridworldEnv()
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    Q1 = defaultdict(lambda: np.zeros(env.action_space.n))
    Q2 = defaultdict(lambda: np.zeros(env.action_space.n))
    _, stats_double_Q = Double_Q_learning(env, Q1, Q2, alpha=ALPHA, discount_factor=DISCOUNT_FACTOR,
                                          num_episodes=NUM_EPISODES)
    _, stats_Q = Q_learning(env, Q, alpha=ALPHA, discount_factor=DISCOUNT_FACTOR, num_episodes=NUM_EPISODES)
    _, stats_sarsa = Sarsa(env, Q, alpha=ALPHA, discount_factor=DISCOUNT_FACTOR, num_episodes=NUM_EPISODES)
    stats = {'Double Q-Learning': stats_double_Q, 'Q-Learning': stats_Q, 'Sarsa': stats_sarsa}
    compare_plots(**stats)
