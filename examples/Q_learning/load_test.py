import pickle
import gym
import numpy as np
import collections
import tensorflow as tf
from tf_rl.common.utils import AnnealingSchedule, eager_setup
from tf_rl.common.wrappers import DiscretisedEnv

eager_setup()

class Q_Agent:
	def __init__(self, env):
		self.env = env
		self.Q_learn = np.zeros(self.env.buckets + (env.action_space.n,))
		self.gamma = 0.95

	def choose_action(self, state, epsilon):
		return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q_learn[state])

	def update(self, state, action, reward, next_state, alpha):
		# original update
		# self.Q_learn[state][action] += alpha * (reward + 1. * np.max(self.Q_learn[next_state]) - self.Q_learn[state][action])

		# hard update: rely on the expert
		# self.Q_learn[state][action] += alpha * (reward + 1. * np.max(self.Q_expert[next_state]) - self.Q_learn[state][action])

		# soft update: rely on the expert to some extent
		self.Q_learn[state][action] += alpha * (reward + 1. * np.max((1-0.2)*self.Q_learn[next_state] + 0.2*self.Q_expert[next_state]) - self.Q_learn[state][action])

	def load_Q(self):
		with open('q_vals.pickle', 'rb') as handle:
			self.Q_expert = pickle.load(handle)
		print(self.Q_expert)

if __name__ == '__main__':
	# DiscretisedEnv
	env = DiscretisedEnv(gym.make('CartPole-v0'))

	# hyperparameters
	n_episodes = 1000
	goal_duration = 150
	decay_steps = 5000
	all_rewards = list()
	durations = collections.deque(maxlen=100)
	Epsilon = AnnealingSchedule(start=1.0, end=0.01, decay_steps=decay_steps)
	Alpha = AnnealingSchedule(start=1.0, end=0.01, decay_steps=decay_steps)
	agent = Q_Agent(env)
	agent.load_Q()

	global_timestep = tf.train.get_or_create_global_step()
	for episode in range(n_episodes):
		current_state = env.reset()

		done = False
		duration = 0

		# one episode of q learning
		while not done:
			# env.render()
			duration += 1
			global_timestep.assign_add(1)
			action = agent.choose_action(current_state, Epsilon.get_value())
			new_state, reward, done, _ = env.step(action)
			agent.update(current_state, action, reward, new_state, Alpha.get_value())
			current_state = new_state

		# mean duration of last 100 episodes
		durations.append(duration)
		all_rewards.append(duration)
		mean_duration = np.mean(durations)

		# check if our policy is good
		if mean_duration >= goal_duration and episode >= 100:
			print('Ran {} episodes. Solved after {} trials'.format(episode, episode - 100))
			# agent.test()
			env.close()
			break

		elif episode % 100 == 0:
			print('[Episode {}] - Mean time over last 100 episodes was {} frames.'.format(episode, mean_duration))
