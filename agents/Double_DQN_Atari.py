import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from common.core import Agent
from common.memory import ReplayBuffer
from common.utils import AnnealingEpsilon, sync_main_target, huber_loss
from common.wrappers_Atari import make_atari, wrap_deepmind

class Double_DQN(Agent):
	"""
	DQN Agent
	"""

	def __init__(self, scope, env):
		self.scope = scope
		with tf.variable_scope(scope):
			self.state = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.float32, name="X")
			self.Y = tf.placeholder(shape=[None], dtype=tf.float32, name="Y")
			self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")

			conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation=tf.nn.relu)(self.state)
			conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation=tf.nn.relu)(conv1)
			conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation=tf.nn.relu)(conv2)
			flat = tf.keras.layers.Flatten()(conv3)
			fc1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(flat)
			self.pred = tf.keras.layers.Dense(env.action_space.n, activation=tf.nn.relu)(fc1)
			# indices of the executed actions
			idx_flattened = tf.range(0, tf.shape(self.pred)[0]) * tf.shape(self.pred)[1] + self.action
			# passing [-1] to tf.reshape means flatten the array
			# using tf.gather, associate Q-values with the executed actions
			self.action_probs = tf.gather(tf.reshape(self.pred, [-1]), idx_flattened)
			self.losses = tf.squared_difference(self.Y, self.action_probs)
			self.loss = tf.reduce_mean(huber_loss(self.losses))
			# self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
			self.optimizer = tf.train.AdamOptimizer()
			self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

	def act(self, sess, state, epsilon):
		"""
		Given a state, it performs an epsilon-greedy policy

		:param sess:
		:param state:
		:param epsilon:
		:return:
		"""
		if np.random.rand() > epsilon:
			q_value = sess.run(self.pred, feed_dict={self.state: state})[0]
			action = np.argmax(q_value)
		else:
			action = np.random.randint(env.action_space.n)
		return action

	def predict(self, sess, state):
		"""
		predict q-values given a state

		:param sess:
		:param state:
		:return:
		"""
		return sess.run(self.pred, feed_dict={self.state: state})

	def update(self, sess, state, action, Y):
		feed_dict = {self.state: state, self.action: action, self.Y: Y}
		_, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
		return loss


if __name__ == '__main__':
	env = make_atari("PongNoFrameskip-v4")
	env = wrap_deepmind(env)

	num_frames = 1000000
	memory_size = 10000
	learning_start = 10000
	sync_freq = 1000
	decay_steps = 100000
	batch_size = 32
	gamma = 0.99

	# log purpose
	losses = []
	all_rewards = []
	episode_reward = 0

	# initialise a graph in a session
	tf.reset_default_graph()

	# training epoch
	global_step = tf.Variable(0, name="global_step", trainable=False)

	# initialise models and replay memory
	main_model = Double_DQN("main", env)
	target_model = Double_DQN("target", env)
	replay_buffer = ReplayBuffer(memory_size)
	Epsilon = AnnealingEpsilon(start=1.0, end=0.01, decay_steps=decay_steps)

	with tf.Session() as sess:
		# initialise all variables used in the model
		sess.run(tf.global_variables_initializer())
		state = env.reset()
		for frame_idx in range(1, num_frames + 1):
			action = target_model.act(sess, state.reshape((1, 84, 84, 1)), Epsilon.get_epsilon(frame_idx))

			next_state, reward, done, _ = env.step(action)
			replay_buffer.store(state, action, reward, next_state, done)

			state = next_state
			episode_reward += reward

			if done:
				state = env.reset()
				all_rewards.append(episode_reward)
				print("\rGAME OVER AT STEP: {0}, SCORE: {1}".format(frame_idx, episode_reward), end="")
				episode_reward = 0

				if frame_idx > learning_start:
					if len(replay_buffer) > batch_size:
						states, actions, rewards, next_states, done = replay_buffer.sample(batch_size)
						next_Q = target_model.predict(sess, next_states)
						Y = rewards + gamma * np.argmax(next_Q, axis=1) * done
						loss = main_model.update(sess, states, actions, Y)
						losses.append(loss)
				else:
					pass

			if frame_idx > learning_start:
				if frame_idx % sync_freq == 0:
					print("\nModel Sync")
					sync_main_target(sess, main_model, target_model)

	# temporal visualisation
	plt.subplot(2, 1, 1)
	plt.plot(all_rewards)
	plt.title("Score over time")
	plt.xlabel("Timestep")
	plt.ylabel("Score")

	plt.subplot(2, 1, 2)
	plt.plot(losses)
	plt.title("Loss over time")
	plt.xlabel("Timestep")
	plt.ylabel("Loss")
	plt.show()