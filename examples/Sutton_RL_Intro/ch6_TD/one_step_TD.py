# 6.1 TD Prediction
# Tabular TD(0)
# reference: https://github.com/dennybritz/reinforcement-learning/blob/master/MC/Off-Policy%20MC%20Control%20with%20Weighted%20Importance%20Sampling%20Solution.ipynb

from collections import defaultdict
import numpy as np
import sys

if "../" not in sys.path:
    sys.path.append("../")

from libs.envs.windy_gridworld import WindyGridworldEnv


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy


def TD_Prediction(env, state_value, alpha=0.01, discount_factor=1.0, num_episodes=1000):
    policy = make_epsilon_greedy_policy(state_value, discount_factor, env.nA)

    for i in range(num_episodes):
        # this satisfies the exploraing start condition
        state = env.reset()
        # generate an episode
        for t in range(100):
            action = np.random.choice(np.arange(env.nA), p=policy(state))
            next_state, reward, done, _ = env.step(action)
            state_value[state] += alpha * (reward + discount_factor * state_value[next_state] - state_value[state])
            if done:
                break
            state = next_state

    return state_value


if __name__ == '__main__':
    env = WindyGridworldEnv()
    state_value = defaultdict(lambda: np.zeros(env.action_space.n))
    state_value = TD_Prediction(env, state_value, alpha=0.01, discount_factor=1.0, num_episodes=1000)
    print(state_value)
