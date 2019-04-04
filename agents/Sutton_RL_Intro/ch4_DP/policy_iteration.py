# Following the algo in section 4.3 Policy Iteration
from policy_evaluation import Policy_Evaluation
import sys
import numpy as np

if "../" not in sys.path:
    sys.path.append("../")

from utils.envs.grid_world import GridworldEnv

def Policy_Improvement(env, policy, state_value, gamma, theta):
	state_value = Policy_Evaluation(env, policy, state_value, gamma, theta).flatten()
	policy_stable = True
	while policy_stable:
		for s in range(env.nS):
			old_action = np.random.choice(env.nA, p=policy[s])
			new_action = np.zeros(env.nA)
			for a in range(env.nA):
				p, next_s, r, _ = env.P[s][a][0]
				new_action[a] = p*(r + gamma*state_value[next_s])
			new_action = np.argmax(new_action)
			policy[s] = np.eye(env.nA)[new_action]
		if old_action != new_action:
			policy_stable = False
	return(policy)

if __name__ == '__main__':
	env = GridworldEnv()
	state_value = np.zeros(env.nS)
	policy = np.ones([env.nS, env.nA])/env.nA
	gamma = 1
	theta = 0.00001

	print("===== Training Started =====")
	policy = Policy_Improvement(env, policy, state_value, gamma, theta)
	print("===== Training Finished =====")
	print(policy)
	print(state_value)
