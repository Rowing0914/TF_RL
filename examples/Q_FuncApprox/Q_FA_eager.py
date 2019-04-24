import gym
import numpy as np
import collections
import tensorflow as tf
from tf_rl.common.utils import AnnealingSchedule
from tf_rl.common.params import Parameters
from tf_rl.common.wrappers import DiscretisedEnv
from tf_rl.common.visualise import plot_Q_values

tf.enable_eager_execution()

class Model(tf.keras.Model):
	def __init__(self, num_action):
		super(Model, self).__init__()
		self.dense1 = tf.keras.layers.Dense(16, activation='relu')
		self.dense2 = tf.keras.layers.Dense(16, activation='relu')
		self.dense3 = tf.keras.layers.Dense(16, activation='relu')
		self.pred = tf.keras.layers.Dense(num_action, activation='linear')

	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.dense2(x)
		x = self.dense3(x)
		pred = self.pred(x)
		return pred

class Q_FA_Agent:
	def __init__(self, env, params, policy_type="Eps"):
		self.env = env
		self.num_action = 1
		self.model = Model(num_action=self.num_action)
		self.params = params
		self.policy_type = policy_type
		self.optimizer = tf.train.AdamOptimizer()

	def choose_action(self, state, epsilon):
		if (np.random.random() <= epsilon):
			return self.env.action_space.sample()
		else:
			return self.model(tf.convert_to_tensor(state[None,:], dtype=tf.float32)).numpy()[0]

	def update(self, state, action, reward, next_state):
		with tf.GradientTape() as tape:
			# make sure to fit all process to compute gradients within this Tape context!!

			# calculate target: R + gamma * max_a Q(s',a')
			next_Q = self.model(tf.convert_to_tensor(next_state[None,:], dtype=tf.float32))
			Y = reward + self.params.gamma * np.max(next_Q, axis=-1).flatten() * np.logical_not(done)

			# calculate Q(s,a)
			q_values = self.model(tf.convert_to_tensor(state[None,:], dtype=tf.float32))

			# use MSE
			batch_loss = tf.squared_difference(Y, q_values)
			loss = tf.reduce_mean(batch_loss)

		# get gradients
		grads = tape.gradient(loss, self.model.trainable_weights)

		# apply processed gradients to the network
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

		return loss, batch_loss


if __name__ == '__main__':
	env = gym.make('MountainCarContinuous-v0')

	# hyperparameters
	n_episodes = 1000
	goal_duration = 198
	all_rewards = list()
	durations = collections.deque(maxlen=100)
	params = Parameters(algo="DQN", mode="CartPole")
	Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
	Alpha = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
	agent = Q_FA_Agent(env, params)

	for episode in range(n_episodes):
		current_state = env.reset()

		done = False
		duration = 0

		# one episode of q learning
		while not done:
			# env.render()
			action = agent.choose_action(current_state, Epsilon.get_value(episode))
			new_state, reward, done, _ = env.step(action)
			agent.update(current_state, action, reward, new_state)
			current_state = new_state
			duration += 1

			if duration >= 200:
				done = True

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

	np.save("../logs/value/rewards_Q_learning.npy", all_rewards)