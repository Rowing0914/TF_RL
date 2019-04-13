# Following the algo in section 4.1 Policy Evaluation (Prediction)

import sys
import numpy as np

if "../" not in sys.path:
    sys.path.append("../")

from utils.envs.grid_world import GridworldEnv

def Policy_Evaluation(env, policy, state_value, gamma, theta):
	while True:
		delta = 0
		for s in range(env.nS):
			v = 0
			for a in range(env.nA):
				p, next_s, r, _ = env.P[s][a][0]
				v += p*policy[s][a]*(r + gamma*state_value[next_s])
			delta = max(delta, np.abs(v - state_value[s]))
			state_value[s] = v
		if delta < theta:
			break
	return(np.array(state_value).reshape(env.shape))

if __name__ == '__main__':
	env = GridworldEnv()
	state_value = np.zeros(env.nS)
	policy = np.ones([env.nS, env.nA])/env.nA
	gamma = 1.0
	theta = 0.00001

	print("===== Training Started =====")
	state_value = Policy_Evaluation(env, policy, state_value, gamma, theta)
	print("===== Training Finished =====")
	print(state_value)
