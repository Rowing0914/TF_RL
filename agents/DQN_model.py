import numpy as np
import time
import os
import tensorflow as tf
from common.utils import sync_main_target, soft_target_model_update, huber_loss, ClipIfNotNone, logging


class Parameters:
	def __init__(self, mode=None):
		assert mode != None
		print("Loading Params for {} Environment".format(mode))
		if mode == "Atari":
			self.state_reshape = (1, 84, 84, 1)
			self.num_frames = 1000000
			self.memory_size = 10000
			self.learning_start = 10000
			self.sync_freq = 1000
			self.batch_size = 32
			self.gamma = 0.99
			self.update_hard_or_soft = "soft"
			self.soft_update_tau = 1e-2
			self.epsilon_start = 1.0
			self.epsilon_end = 0.01
			self.decay_steps = 1000
			self.prioritized_replay_alpha = 0.6
			self.prioritized_replay_beta_start = 0.4
			self.prioritized_replay_beta_end = 1.0
			self.prioritized_replay_noise = 1e-6
		elif mode == "CartPole":
			self.state_reshape = (1, 4)
			self.num_frames = 10000
			self.memory_size = 20000
			self.learning_start = 100
			self.sync_freq = 100
			self.batch_size = 32
			self.gamma = 0.99
			self.update_hard_or_soft = "soft"
			self.soft_update_tau = 1e-2
			self.epsilon_start = 1.0
			self.epsilon_end = 0.1
			self.decay_steps = 1000
			self.prioritized_replay_alpha = 0.6
			self.prioritized_replay_beta_start = 0.4
			self.prioritized_replay_beta_end = 1.0
			self.prioritized_replay_noise = 1e-6


class _DQN:
	"""
	Boilerplate for DQN Agent
	"""

	def __init__(self):
		"""
		define the deep learning model here!

		"""

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
		summaries, total_t, _, loss = sess.run([self.summaries, tf.train.get_global_step(), self.train_op, self.loss], feed_dict=feed_dict)
		# print(action, Y, sess.run(self.idx_flattened, feed_dict=feed_dict))
		self.summary_writer.add_summary(summaries, total_t)
		return loss


class DQN_Atari(_DQN):
	"""
	DQN Agent for Atari Games
	"""

	def __init__(self, scope, env, loss_fn="MSE"):
		self.scope = scope
		self.num_action = env.action_space.n
		self.summaries_dir = "../logs/summary_{}".format(scope)
		with tf.variable_scope(scope):
			self.state = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.float32, name="X")
			self.Y = tf.placeholder(shape=[None], dtype=tf.float32, name="Y")
			self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")

			conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation=tf.nn.relu)(self.state)
			conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation=tf.nn.relu)(conv1)
			conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation=tf.nn.relu)(conv2)
			flat = tf.keras.layers.Flatten()(conv3)
			fc1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(flat)
			self.pred = tf.keras.layers.Dense(self.num_action, activation=tf.nn.relu)(fc1)

			# indices of the executed actions
			idx_flattened = tf.range(0, tf.shape(self.pred)[0]) * tf.shape(self.pred)[1] + self.action

			# passing [-1] to tf.reshape means flatten the array
			# using tf.gather, associate Q-values with the executed actions
			self.action_probs = tf.gather(tf.reshape(self.pred, [-1]), idx_flattened)

			if loss_fn == "huber_loss":
				# use huber loss
				self.losses = tf.subtract(self.Y, self.action_probs)
				# self.loss = huber_loss(self.losses)
				self.loss = tf.reduce_mean(huber_loss(self.losses))
			elif loss_fn == "MSE":
				# use MSE
				self.losses = tf.squared_difference(self.Y, self.action_probs)
				self.loss = tf.reduce_mean(self.losses)
			else:
				assert False

			# you can choose whatever you want for the optimiser
			# self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
			self.optimizer = tf.train.AdamOptimizer()

			# to apply Gradient Clipping, we have to directly operate on the optimiser
			# check this: https://www.tensorflow.org/api_docs/python/tf/train/Optimizer#processing_gradients_before_applying_them
			self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
			self.clipped_grads_and_vars = [(ClipIfNotNone(grad, -1., 1.), var) for grad, var in self.grads_and_vars]
			self.train_op = self.optimizer.apply_gradients(self.clipped_grads_and_vars, global_step=tf.train.get_global_step())

			if self.summaries_dir:
				summary_dir = os.path.join(self.summaries_dir, "summaries_{}".format(scope))
				if not os.path.exists(summary_dir):
					os.makedirs(summary_dir)
				self.summary_writer = tf.summary.FileWriter(summary_dir)

			self.summaries = tf.summary.merge([
				tf.summary.scalar("loss", self.loss),
				tf.summary.histogram("loss_hist", self.losses),
				tf.summary.histogram("q_values_hist", self.pred),
				tf.summary.scalar("mean_q_value", tf.math.reduce_mean(self.pred)),
				tf.summary.scalar("var_q_value", tf.math.reduce_variance(self.pred)),
				tf.summary.scalar("max_q_value", tf.reduce_max(self.pred))
			])

class DQN_CartPole(_DQN):
	"""
	DQN Agent for CartPole game
	"""

	def __init__(self, scope, env, loss_fn="MSE"):
		self.scope = scope
		self.num_action = env.action_space.n
		self.summaries_dir = "../logs/summary_{}".format(scope)
		with tf.variable_scope(scope):
			self.state = tf.placeholder(shape=[None, 4], dtype=tf.float32, name="X")
			self.Y = tf.placeholder(shape=[None], dtype=tf.float32, name="Y")
			self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")

			fc1 = tf.keras.layers.Dense(16, activation=tf.nn.relu)(self.state)
			fc2 = tf.keras.layers.Dense(16, activation=tf.nn.relu)(fc1)
			fc3 = tf.keras.layers.Dense(16, activation=tf.nn.relu)(fc2)
			self.pred = tf.keras.layers.Dense(self.num_action, activation=tf.nn.relu)(fc3)

			# indices of the executed actions
			self.idx_flattened = tf.range(0, tf.shape(self.pred)[0]) * tf.shape(self.pred)[1] + self.action

			# passing [-1] to tf.reshape means flatten the array
			# using tf.gather, associate Q-values with the executed actions
			self.action_probs = tf.gather(tf.reshape(self.pred, [-1]), self.idx_flattened)

			if loss_fn == "huber_loss":
				# use huber loss
				self.losses = tf.subtract(self.Y, self.action_probs)
				# self.loss = huber_loss(self.losses)
				self.loss = tf.reduce_mean(huber_loss(self.losses))
			elif loss_fn == "MSE":
				# use MSE
				self.losses = tf.squared_difference(self.Y, self.action_probs)
				self.loss = tf.reduce_mean(self.losses)
			else:
				assert False

			# you can choose whatever you want for the optimiser
			# self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
			self.optimizer = tf.train.AdamOptimizer()

			# to apply Gradient Clipping, we have to directly operate on the optimiser
			# check this: https://www.tensorflow.org/api_docs/python/tf/train/Optimizer#processing_gradients_before_applying_them
			self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
			self.clipped_grads_and_vars = [(ClipIfNotNone(grad, -1., 1.), var) for grad, var in self.grads_and_vars]
			self.train_op = self.optimizer.apply_gradients(self.clipped_grads_and_vars)

			if self.summaries_dir:
				summary_dir = os.path.join(self.summaries_dir, "summaries_{}".format(scope))
				if not os.path.exists(summary_dir):
					os.makedirs(summary_dir)
				self.summary_writer = tf.summary.FileWriter(summary_dir)

			self.summaries = tf.summary.merge([
				tf.summary.scalar("loss", self.loss),
				tf.summary.histogram("loss_hist", self.losses),
				tf.summary.histogram("q_values_hist", self.pred),
				tf.summary.scalar("mean_q_value", tf.math.reduce_mean(self.pred)),
				tf.summary.scalar("var_q_value", tf.math.reduce_variance(self.pred)),
				tf.summary.scalar("max_q_value", tf.reduce_max(self.pred))
			])



def train_DQN(main_model, target_model, env, replay_buffer, policy, params):
	"""
	Train DQN agent which defined above

	:param main_model:
	:param target_model:
	:param env:
	:param params:
	:return:
	"""

	# Create a glboal step variable
	global_step = tf.Variable(0, name='global_step', trainable=False)

	# log purpose
	losses, all_rewards, cnt_action = [], [], []
	episode_reward, index_episode = 0, 0

	with tf.Session() as sess:
		# initialise all variables used in the model
		sess.run(tf.global_variables_initializer())
		global_step = sess.run(tf.train.get_global_step())
		state = env.reset()
		start = time.time()
		for frame_idx in range(1, params.num_frames + 1):
			action = policy.select_action(sess, target_model, state.reshape(params.state_reshape))
			cnt_action.append(action)
			next_state, reward, done, _ = env.step(action)
			replay_buffer.add(state, action, reward, next_state, done)

			state = next_state
			episode_reward += reward
			global_step += 1

			if done:
				index_episode += 1
				state = env.reset()
				all_rewards.append(episode_reward)

				if frame_idx > params.learning_start and len(replay_buffer) > params.batch_size:
					states, actions, rewards, next_states, dones = replay_buffer.sample(params.batch_size)
					next_Q = target_model.predict(sess, next_states)
					Y = rewards + params.gamma * np.max(next_Q, axis=1) * np.logical_not(dones)
					loss = main_model.update(sess, states, actions, Y)

					# Logging and refreshing log purpose values
					losses.append(loss)
					logging(frame_idx, params.num_frames, index_episode, time.time()-start, episode_reward, loss, cnt_action)

					episode_summary = tf.Summary()
					episode_summary.value.add(simple_value=episode_reward, node_name="episode_reward",
											  tag="episode_reward")
					episode_summary.value.add(simple_value=index_episode, node_name="episode_length",
											  tag="episode_length")
					main_model.summary_writer.add_summary(episode_summary, global_step)

				episode_reward = 0
				cnt_action = []
				start = time.time()

			if frame_idx > params.learning_start and frame_idx % params.sync_freq == 0:
				# soft update means we partially add the original weights of target model instead of completely
				# sharing the weights among main and target models
				if params.update_hard_or_soft == "hard":
					sync_main_target(sess, main_model, target_model)
				elif params.update_hard_or_soft == "soft":
					soft_target_model_update(sess, main_model, target_model, tau=params.soft_update_tau)


	return all_rewards, losses