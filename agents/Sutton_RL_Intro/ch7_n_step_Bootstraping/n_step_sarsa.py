# 7.2 n-step Sarsa

from collections import defaultdict
import numpy as np
import sys
import itertools

if "../" not in sys.path:
	sys.path.append("../")

from utils.envs.windy_gridworld import WindyGridworldEnv
from utils.plot import plot_result


def make_epsilon_greedy_policy(Q, epsilon, nA):
	def policy(observation):
		A = np.ones(nA, dtype=float) * epsilon / nA
		best_action = np.argmax(Q[observation])
		A[best_action] += (1.0 - epsilon)
		return A
	return policy


def n_step_Sarsa(env, Q, n_step=3, alpha=0.5, discount_factor=1.0, epsilon=0.1, num_episodes=1000):
	stats = np.zeros((num_episodes, 2))
	policy = make_epsilon_greedy_policy(Q, epsilon, env.nA)

	for i in range(num_episodes):
		cnt = 0
		# this satisfies the exploraing start condition
		state = env.reset()
		current_state = state

		# initialise the memory for n-step
		memory = list()

		action = np.random.choice(np.arange(env.nA), p=policy(state))

		# generate an episode
		for t in itertools.count():
			next_state, reward, done, _ = env.step(action)
			next_action = np.random.choice(np.arange(env.nA), p=policy(next_state))
			memory.append(reward)

			if cnt == n_step:
				G = np.sum([ discount_factor**i * r for i, r in enumerate(memory)])
				Q[state][action] += alpha * (G - Q[state][action])

			stats[i, 0] = t
			stats[i, 1] += reward

			if done:
				break

			state = next_state
			action = next_action
	return Q, stats


if __name__ == '__main__':
	env = WindyGridworldEnv()
	Q = defaultdict(lambda: np.zeros(env.action_space.n))
	Q, stats = n_step_Sarsa(env, Q, n_step=3, alpha=0.5, discount_factor=1.0, num_episodes=100)
	plot_result(stats)
