""" Q-Learning implementation for Cartpole """

import gym
import numpy as np
import collections
from common.utils import AnnealingSchedule
from common.params import Parameters
from common.wrappers import DiscretisedEnv

class Q_Agent:
	def __init__(self, env, params, policy_type="Eps"):
		self.env = env
		self.Q = np.zeros(self.env.buckets + (env.action_space.n,))
		self.gamma = params.gamma
		self.tau = params.tau
		self.clip = params.clip
		self.policy_type = policy_type

	def choose_action(self, state, epsilon):
		if self.policy_type == "Eps":
			return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])
		elif self.policy_type == "Boltz":
			q_values = self.Q[state]
			nb_actions = q_values.shape[0]

			exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
			probs = exp_values / np.sum(exp_values)
			return np.random.choice(range(nb_actions), p=probs)

	def update(self, state, action, reward, next_state, alpha):
		self.Q[state][action] += alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])


def test(agent):
	current_state = env.reset()
	done = False

	while not done:
		action = agent.choose_action(current_state, 0)
		obs, reward, done, _ = env.step(action)
		env.render()
		current_state = obs
	return


if __name__ == '__main__':
	# DiscretisedEnv
	env = DiscretisedEnv(gym.make('CartPole-v0'))

	# hyperparameters
	n_episodes = 2000
	goal_duration = 195

	durations = collections.deque(maxlen=100)
	params = Parameters(mode="CartPole")
	Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
	Alpha = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
	agent = Q_Agent(env, params)

	for episode in range(n_episodes):

		current_state = env.reset()

		done = False
		duration = 0

		# one episode of q learning
		while not done:
			# env.render()
			action = agent.choose_action(current_state, Epsilon.get_value(episode))
			obs, reward, done, _ = env.step(action)
			new_state = obs
			agent.update(current_state, action, reward, new_state, Alpha.get_value(episode))
			current_state = new_state
			duration += 1

		# mean duration of last 100 episodes
		durations.append(duration)
		mean_duration = np.mean(durations)

		# check if our policy is good
		if mean_duration >= goal_duration and episode >= 100:
			print('Ran {} episodes. Solved after {} trials'.format(episode, episode - 100))
			# test()
			env.close()
			break

		elif episode % 100 == 0:
			print('[Episode {}] - Mean time over last 100 episodes was {} frames.'.format(episode, mean_duration))