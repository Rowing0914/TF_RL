import gym
import numpy as np
import collections
import tensorflow as tf
from tf_rl.common.utils import AnnealingSchedule, eager_setup
from tf_rl.common.wrappers import DiscretisedEnv
from tf_rl.common.visualise import plot_Q_values

eager_setup()

class Q_Agent:
	def __init__(self, env):
		self.env = env
		self.Q = np.zeros(self.env.buckets + (env.action_space.n,))
		self.gamma = 0.995

	def choose_action(self, state, epsilon):
		return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

	def update(self, state, action, reward, next_state, alpha):
		self.Q[state][action] += alpha * (reward + 1. * np.max(self.Q[next_state]) - self.Q[state][action])

	def test(self):
		"""
		Test the agent with a visual aid!
		"""

		# xmax = 2
		# xmin = -1
		# ymax = np.amax(self.Q) + 30
		# ymin = 0

		scores = list()
		for ep in range(10):
			current_state = self.env.reset()
			done = False
			score = 0
			while not done:
				# self.env.render()
				action = self.choose_action(current_state, 0)
				# plot_Q_values(self.Q[current_state], xmin, xmax, ymin, ymax)
				# print(self.Q[current_state])
				obs, reward, done, _ = self.env.step(action)
				current_state = obs
				score += reward
			scores.append(score)
			print("Ep: {}, Score: {}".format(ep, score))
		scores = np.array(scores)
		print("Eval => Std: {}, Mean: {}".format(np.std(scores), np.mean(scores)))


if __name__ == '__main__':
	# DiscretisedEnv
	env = DiscretisedEnv(gym.make('CartPole-v0'))

	# hyperparameters
	n_episodes = 1000
	goal_duration = 190
	decay_steps = 5000
	all_rewards = list()
	durations = collections.deque(maxlen=10)
	Epsilon = AnnealingSchedule(start=1.0, end=0.01, decay_steps=decay_steps)
	Alpha = AnnealingSchedule(start=1.0, end=0.01, decay_steps=decay_steps)
	agent = Q_Agent(env)

	global_timestep = tf.train.get_or_create_global_step()
	for episode in range(n_episodes):
		current_state = env.reset()

		done = False
		duration = 0

		stds, means = [], []

		# one episode of q learning
		while not done:
			# env.render()
			duration += 1
			global_timestep.assign_add(1)
			action = agent.choose_action(current_state, Epsilon.get_value())
			new_state, reward, done, _ = env.step(action)
			agent.update(current_state, action, reward, new_state, Alpha.get_value())
			current_state = new_state
			stds.append(np.std(agent.Q[current_state]))
			means.append(np.mean(agent.Q[current_state]))

		print("Ep: {}, Std: {}, Mean: {}".format(episode, np.mean(stds), np.mean(means)))

		# mean duration of last 100 episodes
		durations.append(duration)
		all_rewards.append(duration)
		mean_duration = np.mean(durations)

		# check if our policy is good
		if mean_duration >= goal_duration and episode >= 100:
			print('Ran {} episodes. Solved after {} trials'.format(episode, episode - 100))
			agent.test()
			env.close()
			break

		elif episode % 100 == 0:
			print('[Episode {}] - Mean time over last 100 episodes was {} frames.'.format(episode, mean_duration))

	# import pickle
	# data = agent.Q
	# with open('q_vals.pickle', 'wb') as handle:
	# 	pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
