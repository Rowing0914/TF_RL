import numpy as np
import tensorflow as tf
from common.utils import sync_main_target, soft_target_model_update, huber_loss, ClipIfNotNone


class _Duelling_Double_DQN_PER:
	"""
	Boilerplate for DQN Agent with PER
	"""

	def __init__(self):
		"""
		define your model here!
		"""
		pass

	def act(self, sess, state, epsilon):
		"""
		Given a state, it performs an epsilon-greedy policy

		:param sess:
		:param state:
		:param epsilon:
		:return:
		"""

		if np.random.uniform() < epsilon:
			action = np.random.randint(self.num_action)
		else:
			q_value = sess.run(self.output, feed_dict={self.state: state})[0]
			# dummy = sess.run(self.pred, feed_dict={self.state: state})[0]
			# print(q_value, dummy)
			action = np.argmax(q_value)
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
		_, loss, batch_loss = sess.run([self.train_op, self.loss, self.losses], feed_dict=feed_dict)
		return loss, batch_loss


class Duelling_Double_DQN_PER_Atari(_Duelling_Double_DQN_PER):
	"""
	DQN Agent with PER for Atari Games
	"""

	def __init__(self, scope, dueling_type, env):
		self.scope = scope
		self.num_action = env.action_space.n
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
			self.state_value = tf.keras.layers.Dense(1, activation=tf.nn.relu)(fc1)

			# ===== Duelling DQN part =====
			if dueling_type == "avg":
				# Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
				self.output = tf.math.add(self.state_value, tf.math.subtract(self.pred, tf.reduce_mean(self.pred)))
			elif dueling_type == "max":
				# Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
				self.output = tf.math.add(self.state_value, tf.math.subtract(self.pred, tf.math.reduce_max(self.pred)))
			elif dueling_type == "naive":
				# Q(s,a;theta) = V(s;theta) + A(s,a;theta)
				self.output = tf.math.add(self.state_value, self.pred)
			else:
				assert False, "dueling_type must be one of {'avg','max','naive'}"

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


class Duelling_Double_DQN_PER_CartPole(_Duelling_Double_DQN_PER):
	"""
	DQN Agent with PER for CartPole game
	"""

	def __init__(self, scope, dueling_type, env):
		self.scope = scope
		self.num_action = env.action_space.n
		with tf.variable_scope(scope):
			self.state = tf.placeholder(shape=[None, 4], dtype=tf.float32, name="X")
			self.Y = tf.placeholder(shape=[None], dtype=tf.float32, name="Y")
			self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")

			fc1 = tf.keras.layers.Dense(16, activation=tf.nn.relu)(self.state)
			fc2 = tf.keras.layers.Dense(16, activation=tf.nn.relu)(fc1)
			self.pred = tf.keras.layers.Dense(self.num_action, activation=tf.nn.relu)(fc2)
			self.state_value = tf.keras.layers.Dense(1, activation=tf.nn.relu)(fc2)

			# ===== Duelling DQN part =====
			if dueling_type == "avg":
				# Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
				self.output = tf.math.add(self.state_value, tf.math.subtract( self.pred, tf.reduce_mean(self.pred) ))
			elif dueling_type == "max":
				# Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
				self.output = tf.math.add(self.state_value, tf.math.subtract( self.pred, tf.math.reduce_max(self.pred) ))
			elif dueling_type == "naive":
				# Q(s,a;theta) = V(s;theta) + A(s,a;theta)
				self.output = tf.math.add(self.state_value, self.pred)
			else:
				assert False, "dueling_type must be one of {'avg','max','naive'}"

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



def train_Duelling_Double_DQN_PER(main_model, target_model, env, replay_buffer, Epsilon, Beta, params):
	"""
	Train DQN agent which defined above

	:param main_model:
	:param target_model:
	:param env:
	:param params:
	:return:
	"""

	# log purpose
	losses = []
	all_rewards = []
	episode_reward = 0

	# training epoch
	global_step = tf.Variable(0, name="global_step", trainable=False)

	with tf.Session() as sess:
		# initialise all variables used in the model
		sess.run(tf.global_variables_initializer())
		state = env.reset()
		for frame_idx in range(1, params.num_frames + 1):
			action = target_model.act(sess, state.reshape(params.state_reshape), Epsilon.get_value(frame_idx))

			next_state, reward, done, _ = env.step(action)
			replay_buffer.add(state, action, reward, next_state, done)

			state = next_state
			episode_reward += reward

			if done:
				state = env.reset()
				all_rewards.append(episode_reward)
				print("\rGAME OVER AT STEP: {0}, SCORE: {1}".format(frame_idx, episode_reward), end="")
				episode_reward = 0

				if frame_idx > params.learning_start and len(replay_buffer) > params.batch_size:
					# ===== Prioritised Experience Replay part =====
					# PER returns: state, action, reward, next_state, done, weights(a weight for a timestep), indices(indices for a batch of timesteps)
					states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(params.batch_size, Beta.get_value(frame_idx))

					# ===== Double DQN part =====
					next_Q_main = main_model.predict(sess, next_states)
					next_Q = target_model.predict(sess, next_states)
					Y = rewards + params.gamma * next_Q[
						np.arange(params.batch_size), np.argmax(next_Q_main, axis=1)] * dones

					# we get a batch of loss for updating PER memory
					loss, batch_loss = main_model.update(sess, states, actions, Y)

					print("GAME OVER AT STEP: {0}, SCORE: {1}, LOSS: {2}".format(frame_idx, episode_reward, loss))

					# add noise to the priorities
					batch_loss = np.abs(batch_loss) + params.prioritized_replay_noise

					# Update a prioritised replay buffer using a batch of losses associated with each timestep
					replay_buffer.update_priorities(indices, batch_loss)

					# log purpose
					losses.append(loss)

			if frame_idx > params.learning_start and frame_idx % params.sync_freq == 0:
				# soft update means we partially add the original weights of target model instead of completely
				# sharing the weights among main and target models
				if params.update_hard_or_soft == "hard":
					sync_main_target(sess, main_model, target_model)
				elif params.update_hard_or_soft == "soft":
					soft_target_model_update(sess, main_model, target_model, tau=params.soft_update_tau)

	return all_rewards, losses