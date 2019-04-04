# 5.6 Incremental Implementation
# Off-policy MC prediction
# reference: https://github.com/dennybritz/reinforcement-learning/blob/master/MC/Off-Policy%20MC%20Control%20with%20Weighted%20Importance%20Sampling%20Solution.ipynb

from collections import defaultdict
import numpy as np
import sys
if "../" not in sys.path:
    sys.path.append("../")

from utils.envs.blackjack import BlackjackEnv


def make_epsilon_greedy_policy(Q, epsilon, nA):
	def target_policy(observation):
		A = np.ones(nA, dtype=float) * epsilon / nA
		best_action = np.argmax(Q[observation])
		A[best_action] += (1.0 - epsilon)
		return A

	def behaviour_policy(observation):
		return np.ones(nA)/nA

	return target_policy, behaviour_policy

def Off_Policy_MC(env, action_value, discount_factor=1.0, num_episodes=1000):
	C = defaultdict(lambda: np.zeros(env.action_space.n))
	target_policy, behaviour_policy = make_epsilon_greedy_policy(action_value, discount_factor, env.nA)

	for i in range(num_episodes):
		# observe the environment and store the observation
		experience = []
		# this satisfies the exploraing start condition
		observation = env.reset()
		# generate an episode
		for t in range(100):
			action = np.random.choice(np.arange(env.nA), p=behaviour_policy(observation))
			next_observation, reward, done, _ = env.step(action)
			experience.append((observation, action, reward))
			observation = next_observation
			if done:
				break
		
		G = 0.0
		W = 1.0
		
		# update the state-value function using the obtained episode
		for row in experience[::-1]:
			state, action, reward = row[0], row[1], row[2]
			G = discount_factor*G + reward
			C[state][action] += W
			action_value[state][action] += (W/C[state][action])*(G - action_value[state][action])
			# assume that the optimal policy assigns the maximum probability to the action
			W = W*(target_policy(state)[action]/behaviour_policy(state)[action])
			# stopping condition of Dennybritz
			# if action != np.argmax(target_policy(state)[action]):
			if W == 0.0:
				# print("hi")
				break
	return action_value, target_policy


if __name__ == '__main__':
	env = BlackjackEnv()
	action_value = defaultdict(lambda: np.zeros(env.action_space.n))
	discount_factor = 1.0
	num_episodes = 10000
	action_value, policy = Off_Policy_MC(env, action_value, discount_factor=1.0, num_episodes=num_episodes)
	print(action_value)