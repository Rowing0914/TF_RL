# 7.1 N-STEP TD PREDICTION

from collections import defaultdict, namedtuple
import numpy as np
import sys

if "../" not in sys.path:
    sys.path.append("../")

from libs.envs.windy_gridworld import WindyGridworldEnv

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy


def n_step_TD_Prediction(env, state_value, n_step=3, alpha=0.01, discount_factor=1.0, num_episodes=1000):
    policy = make_epsilon_greedy_policy(state_value, discount_factor, env.nA)

    for i in range(num_episodes):
        cnt = 0
        # this satisfies the exploraing start condition
        state = env.reset()
        current_state = state
        # initialise the memory for n-step
        memory = list()

        # generate an episode
        for t in range(100):
            for _ in range(n_step):
                action = np.random.choice(np.arange(env.nA), p=policy(state))
                next_state, reward, done, _ = env.step(action)

                memory.append(reward)

                # if we arrive at the terminal state before n_step count
                if done:
                    G = np.sum([discount_factor ** (i + 1) * reward for i, reward in enumerate(memory)])
                    state_value[state] += alpha * (G + state_value[state] - state_value[current_state])
                    break

            G = np.sum([discount_factor ** (i + 1) * reward for i, reward in enumerate(memory)])
            state_value[state] += alpha * (G + state_value[next_state] - state_value[current_state])

            # reset them
            current_state = state
            memory = list()

            if done:
                break

            cnt += 1
            state = next_state

    return state_value


if __name__ == '__main__':
    env = WindyGridworldEnv()
    state_value = defaultdict(lambda: np.zeros(env.action_space.n))
    state_value = n_step_TD_Prediction(env, state_value, n_step=3, alpha=0.01, discount_factor=1.0, num_episodes=10)
    print(state_value)
