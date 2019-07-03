# Following the algo in section 4.4 Value Iteration

from policy_evaluation import Policy_Evaluation
import sys
import numpy as np

if "../" not in sys.path:
    sys.path.append("../")

from utils.envs.grid_world import GridworldEnv


def Value_Iteration(env, policy, state_value, gamma, theta):
    state_value = Policy_Evaluation(env, policy, state_value, gamma, theta).flatten()
    for s in range(env.nS):
        policy[s] = np.eye(env.nA)[np.argmax(policy[s])]
    return (policy)


if __name__ == '__main__':
    env = GridworldEnv()
    state_value = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA
    gamma = 1
    theta = 0.00001

    print("===== Training Started =====")
    policy = Value_Iteration(env, policy, state_value, gamma, theta)
    print("===== Training Finished =====")
    print(policy)
    print(state_value)
