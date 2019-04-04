import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from common.memory import ReplayBuffer
from common.utils import AnnealingEpsilon, sync_main_target, huber_loss, ClipIfNotNone


class Duelling_DQN:
	"""
	DQN Agent
	"""

	def __init__(self, scope, env):
		self.scope = scope
		self.num_action = env.action_space.n
		with tf.variable_scope(scope):
			self.state = tf.placeholder(shape=[None, env.observation_space.shape[0]], dtype=tf.float32, name="X")
			self.Y = tf.placeholder(shape=[None], dtype=tf.float32, name="Y")
			self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")

			fc1 = tf.keras.layers.Dense(16, activation=tf.nn.relu)(self.state)
			fc2 = tf.keras.layers.Dense(16, activation=tf.nn.relu)(fc1)
			self.pred = tf.keras.layers.Dense(env.action_space.n, activation=tf.nn.relu)(fc2)
			self.state_value = tf.keras.layers.Dense(1, activation=tf.nn.relu)(fc2)

			# indices of the executed actions
			idx_flattened = tf.range(0, tf.shape(self.pred)[0]) * tf.shape(self.pred)[1] + self.action

			# passing [-1] to tf.reshape means flatten the array
			# using tf.gather, associate Q-values with the executed actions
			self.action_probs = tf.gather(tf.reshape(self.pred, [-1]), idx_flattened)

			# MSE loss function
			self.losses = tf.squared_difference(self.Y, self.action_probs)
			self.loss = tf.reduce_mean(huber_loss(self.losses))

			# you can choose whatever you want for the optimiser
			# self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
			self.optimizer = tf.train.AdamOptimizer()

			# to apply Gradient Clipping, we have to directly operate on the optimiser
			# check this: https://www.tensorflow.org/api_docs/python/tf/train/Optimizer#processing_gradients_before_applying_them
			self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
			self.clipped_grads_and_vars = [(ClipIfNotNone(grad, -1., 1.), var) for grad, var in self.grads_and_vars]
			self.train_op = self.optimizer.apply_gradients(self.clipped_grads_and_vars)

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
			# print(q_value)
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
	env = gym.make("CartPole-v0")

	num_frames = 10000
	memory_size = 1000
	learning_start = 1000
	sync_freq = 200
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
	main_model = DQN("main", env)
	target_model = DQN("target", env)
	replay_buffer = ReplayBuffer(memory_size)
	Epsilon = AnnealingEpsilon(start=1.0, end=0.1, decay_steps=500)

	with tf.Session() as sess:
		# initialise all variables used in the model
		sess.run(tf.global_variables_initializer())
		state = env.reset()
		for frame_idx in range(1, num_frames + 1):
			action = target_model.act(sess, state.reshape(1, 4), Epsilon.get_epsilon(frame_idx))

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
						states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
						next_Q = target_model.predict(sess, next_states)
						Y = rewards + gamma * np.argmax(next_Q, axis=1) * dones
						# print(Y)
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