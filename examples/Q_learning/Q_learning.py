# TODO: Bug fix to compatible with Tensorflow.....
import gym
import numpy as np
import collections
import tensorflow as tf
from tf_rl.common.utils import AnnealingSchedule
from tf_rl.common.params import Parameters
from tf_rl.common.wrappers import DiscretisedEnv
from tf_rl.common.visualise import plot_Q_values

class Q_Agent:
	def __init__(self, env):
		self.env = env
		self.Q = np.zeros(self.env.buckets + (env.action_space.n,))
		self.gamma = 0.95

	def choose_action(self, state, epsilon):
		return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

	def update(self, state, action, reward, next_state, alpha):
		# === I don't know why tho, self.gamma = 0.99 does not converge in Q-learning ===
		# self.Q[state][action] += alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
		self.Q[state][action] += alpha * (reward + 1. * np.max(self.Q[next_state]) - self.Q[state][action])

	def test(self):
		"""
		Test the agent with a visual aid!

		:return:
		"""
		current_state = self.env.reset()
		done = False

		xmax = 2
		xmin = -1
		ymax = np.amax(self.Q) + 30
		ymin = 0

		while not done:
			self.env.render()
			action = self.choose_action(current_state, 0)
			plot_Q_values(self.Q[current_state], xmin, xmax, ymin, ymax)
			print(self.Q[current_state])
			obs, reward, done, _ = self.env.step(action)
			current_state = obs
		return


if __name__ == '__main__':
	# DiscretisedEnv
	env = DiscretisedEnv(gym.make('CartPole-v0'))

	# hyperparameters
	n_episodes = 1000
	goal_duration = 198
	all_rewards = list()
	durations = collections.deque(maxlen=100)
	params = Parameters(algo="DQN", mode="CartPole")
	Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
	Alpha = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
	agent = Q_Agent(env)

	global_timestep = tf.train.get_or_create_global_step()
	for episode in range(n_episodes):
		current_state = env.reset()

		done = False
		duration = 0

		# one episode of q learning
		while not done:
			# env.render()
			action = agent.choose_action(current_state, Epsilon.get_value())
			new_state, reward, done, _ = env.step(action)
			agent.update(current_state, action, reward, new_state, Alpha.get_value())
			current_state = new_state
			tf.assign(global_timestep, global_timestep.numpy() + 1, name='update_global_step')

		# mean duration of last 100 episodes
		durations.append(global_timestep)
		all_rewards.append(global_timestep)
		mean_duration = np.mean(durations)

		# check if our policy is good
		if mean_duration >= goal_duration and episode >= 100:
			print('Ran {} episodes. Solved after {} trials'.format(episode, episode - 100))
			# agent.test()
			env.close()
			break

		elif episode % 100 == 0:
			print('[Episode {}] - Mean time over last 100 episodes was {} frames.'.format(episode, mean_duration))

	np.save("../logs/value/rewards_Q_learning.npy", all_rewards)