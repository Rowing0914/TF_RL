# 5.4 Monte Carlo Control without Exploring Starts
# On-policy first-visit MC control (for ε-soft policies)
# reference: https://github.com/dennybritz/reinforcement-learning/blob/master/MC/MC%20Control%20with%20Epsilon-Greedy%20Policies%20Solution.ipynb

from collections import defaultdict
import numpy as np
import sys

if "../" not in sys.path:
    sys.path.append("../")

from utils.envs.blackjack import BlackjackEnv


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy


def On_Policy_First_Visit_MC_WO_ES(env, action_value, discount_factor=1.0, num_episodes=1000):
    Returns = defaultdict(float)
    Returns_count = defaultdict(float)
    policy = make_epsilon_greedy_policy(action_value, discount_factor, env.nA)

    for i in range(num_episodes):
        # observe the environment and store the observation
        experience = []
        # this satisfies the exploraing start condition
        observation = env.reset()
        # generate an episode
        for t in range(100):
            action = np.random.choice(np.arange(env.nA), p=policy(observation))
            next_observation, reward, done, _ = env.step(action)
            experience.append((observation, action, reward))
            observation = next_observation
            if done:
                break

        # remove duplicated state-action pairs in the episode
        state_action_in_experience = set([(x[0], x[1]) for x in experience])
        # update the state-value function using the obtained episode
        for row in state_action_in_experience:
            state, action = row[0], row[1]
            # Find the first occurance of the state-action pair in the episode
            first_occurence_idx = next(i for i, x in enumerate(experience) if ((x[0] == state) and (x[1] == action)))
            # Sum up all discounted rewards over time since the first occurance in the episode
            G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(experience[first_occurence_idx:])])
            # Calculate average return for this state over all sampled experiences
            Returns[row] += G
            Returns_count[row] += 1.0
            action_value[state][action] = Returns[row] / Returns_count[row]

    return action_value, policy


if __name__ == '__main__':
    env = BlackjackEnv()
    action_value = defaultdict(lambda: np.zeros(env.action_space.n))
    discount_factor = 1.0
    num_episodes = 100
    action_value, policy = On_Policy_First_Visit_MC_WO_ES(env, action_value, discount_factor=1.0,
                                                          num_episodes=num_episodes)
    print(action_value)
