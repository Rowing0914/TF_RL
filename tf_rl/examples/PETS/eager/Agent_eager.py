from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Agent:
    """An general class for RL agents.
    """

    def __init__(self, env, horizon, policy):
        """Initializes an agent.
        """
        self.env = env
        self.horizon = horizon
        self.policy = policy
        self.has_been_trained = False

    def sample(self):
        """Samples a rollout from the agent.

        Returns: (dict) A dictionary containing data from the rollout.
            The keys of the dictionary are 'obs', 'ac', and 'reward_sum'.
        """
        times, rewards = [], []
        O, A, reward_sum, done = [self.env.reset()], [], 0, False

        self.policy.reset()
        for t in range(self.horizon):
            if self.has_been_trained:
                print(t)

            A.append(self.policy.act(O[t], t, self.has_been_trained))

            obs, reward, done, info = self.env.step(A[t])

            O.append(obs)
            reward_sum += reward
            rewards.append(reward)
            if done:
                break

        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
        }
