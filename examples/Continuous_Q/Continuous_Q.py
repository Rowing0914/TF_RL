import gym
import numpy as np
import itertools
import tensorflow as tf
from tf_rl.common.utils import AnnealingSchedule
from tf_rl.common.params import Parameters
from tf_rl.common.utils import logging
from tf_rl.common.filters import Particle_Filter
from tf_rl.common.wrappers import MyWrapper_revertable

tf.enable_eager_execution()
tf.random.set_random_seed(123)

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

class Continuous_Q_Agent:
	def __init__(self, env, params, policy_type="Eps"):
		self.env = env
		self.num_action = 1
		self.model = Model(num_action=self.num_action)
		self.params = params
		self.policy_type = policy_type
		self.optimizer = tf.train.AdamOptimizer()

	def estimate_Q(self, state, epsilon):
		if (np.random.random() <= epsilon):
			return self.env.action_space.sample()
		else:
			return self.model(tf.convert_to_tensor(state[None,:], dtype=tf.float32)).numpy()[0]

	def update(self, state, action, reward, next_state, done):
		with tf.GradientTape() as tape:
			# make sure to fit all process to compute gradients within this Tape context!!

			# calculate target: R + gamma * max_a' Q(s', a')
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
	env = MyWrapper_revertable(gym.make('MountainCarContinuous-v0'))

	# hyperparameters
	all_rewards = list()
	params = Parameters(algo="DQN", mode="CartPole")
	Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=100)
	agent = Continuous_Q_Agent(env, params)
	pf = Particle_Filter(N=10,type="uniform")
	global_step = 0

	for episode in range(params.num_episodes):
		state = env.reset()
		episode_loss = 0
		episode_reward = 0

		for t in itertools.count():
			# estimate
			mean, var = pf.estimate()
			action = np.random.normal(mean, var, 1)

			if episode > 100:
				env.render()

			# predict and update particles
			pf.predict(env, action)
			q_values = agent.estimate_Q(state, Epsilon.get_value(0))
			pf.update(q_values=q_values)
			pf.simple_resample()

			next_state, reward, done, _ = env.step(action)
			loss, batch_loss = agent.update(state, action, reward, next_state, done)

			episode_loss += loss
			episode_reward += reward
			state = next_state
			global_step += 1

			if t >= 300 or done:
				logging(global_step, params.num_frames, episode, 0, episode_reward, episode_loss, Epsilon.get_value(episode), [0])
				break
