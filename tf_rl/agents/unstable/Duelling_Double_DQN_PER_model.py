import numpy as np
import time
import os
import tensorflow as tf
from tf_rl.common.utils import sync_main_target, soft_target_model_update, huber_loss, logging


class _Duelling_Double_DQN_PER:
	"""
	Boilerplate for DQN Agent with PER
	"""

	def __init__(self):
		"""
		define your model here!
		"""
		pass

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
		summaries, total_t, _, loss, batch_loss = sess.run([self.summaries, tf.train.get_global_step(), self.train_op, self.loss, self.losses], feed_dict=feed_dict)
		self.summary_writer.add_summary(summaries, total_t)
		return loss, batch_loss


class Duelling_Double_DQN_PER_Atari(_Duelling_Double_DQN_PER):
	"""
	DQN Agent with PER for Atari Games
	"""

	def __init__(self, scope, dueling_type, env, loss_fn="MSE", grad_clip_flg=True):
		self.scope = scope
		self.num_action = env.action_space.n
		self.summaries_dir = "../logs/summary_{}".format(scope)
		self.grad_clip_flg = grad_clip_flg

		if self.summaries_dir:
			summary_dir = os.path.join(self.summaries_dir, "summaries_{}".format(scope))
			if not os.path.exists(summary_dir):
				os.makedirs(summary_dir)
			self.summary_writer = tf.summary.FileWriter(summary_dir)

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

			if loss_fn == "huber_loss":
				# use huber loss
				self.losses = tf.subtract(self.Y, self.action_probs)
				self.loss = huber_loss(self.losses)
				# self.loss = tf.reduce_mean(huber_loss(self.losses))
			elif loss_fn == "MSE":
				# use MSE
				self.losses = tf.squared_difference(self.Y, self.action_probs)
				self.loss = tf.reduce_mean(self.losses)
			else:
				assert False

			# you can choose whatever you want for the optimiser
			# self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
			self.optimizer = tf.train.AdamOptimizer()

			if self.grad_clip_flg:
				# to apply Gradient Clipping, we have to directly operate on the optimiser
				# check this: https://www.tensorflow.org/api_docs/python/tf/train/Optimizer#processing_gradients_before_applying_them
				#             https://stackoverflow.com/questions/49987839/how-to-handle-none-in-tf-clip-by-global-norm
				self.gradients, self.variables = zip(*self.optimizer.compute_gradients(self.loss))
				# self.clipped_grads_and_vars = [(ClipIfNotNone(grad, -1., 1.), var) for grad, var in self.grads_and_vars]
				self.gradients, _ = tf.clip_by_global_norm(self.gradients, 2.5)
				self.train_op = self.optimizer.apply_gradients(zip(self.gradients, self.variables))

				for i, grad in enumerate(self.gradients):
					if grad is not None:
						mean = tf.reduce_mean(tf.abs(grad))
						tf.summary.scalar('mean_{}'.format(i + 1), mean)
						tf.summary.histogram('histogram_{}'.format(i + 1), grad)
			else:
				self.train_op = self.optimizer.minimize(self.loss)

			tf.summary.scalar("loss", tf.reduce_mean(self.loss))
			tf.summary.histogram("loss_hist", self.losses)
			tf.summary.histogram("q_values_hist", self.pred)
			tf.summary.scalar("mean_q_value", tf.math.reduce_mean(self.pred))
			tf.summary.scalar("var_q_value", tf.math.reduce_variance(self.pred))
			tf.summary.scalar("max_q_value", tf.reduce_max(self.pred))
			self.summaries = tf.summary.merge_all()



class Duelling_Double_DQN_PER_CartPole(_Duelling_Double_DQN_PER):
	"""
	DQN Agent with PER for CartPole game
	"""

	def __init__(self, scope, dueling_type, env, loss_fn="MSE", grad_clip_flg=True):
		self.scope = scope
		self.num_action = env.action_space.n
		self.summaries_dir = "../logs/summary_{}".format(scope)
		self.grad_clip_flg = grad_clip_flg

		if self.summaries_dir:
			summary_dir = os.path.join(self.summaries_dir, "summaries_{}".format(scope))
			if not os.path.exists(summary_dir):
				os.makedirs(summary_dir)
			self.summary_writer = tf.summary.FileWriter(summary_dir)

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

			if loss_fn == "huber_loss":
				# use huber loss
				self.losses = tf.subtract(self.Y, self.action_probs)
				self.loss = huber_loss(self.losses)
				# self.loss = tf.reduce_mean(huber_loss(self.losses))
			elif loss_fn == "MSE":
				# use MSE
				self.losses = tf.squared_difference(self.Y, self.action_probs)
				self.loss = tf.reduce_mean(self.losses)
			else:
				assert False

			# you can choose whatever you want for the optimiser
			# self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
			self.optimizer = tf.train.AdamOptimizer()

			if self.grad_clip_flg:
				# to apply Gradient Clipping, we have to directly operate on the optimiser
				# check this: https://www.tensorflow.org/api_docs/python/tf/train/Optimizer#processing_gradients_before_applying_them
				#             https://stackoverflow.com/questions/49987839/how-to-handle-none-in-tf-clip-by-global-norm
				self.gradients, self.variables = zip(*self.optimizer.compute_gradients(self.loss))
				# self.clipped_grads_and_vars = [(ClipIfNotNone(grad, -1., 1.), var) for grad, var in self.grads_and_vars]
				self.gradients, _ = tf.clip_by_global_norm(self.gradients, 2.5)
				self.train_op = self.optimizer.apply_gradients(zip(self.gradients, self.variables))

				for i, grad in enumerate(self.gradients):
					if grad is not None:
						mean = tf.reduce_mean(tf.abs(grad))
						tf.summary.scalar('mean_{}'.format(i + 1), mean)
						tf.summary.histogram('histogram_{}'.format(i + 1), grad)
			else:
				self.train_op = self.optimizer.minimize(self.loss)

			tf.summary.scalar("loss", tf.reduce_mean(self.loss))
			tf.summary.histogram("loss_hist", self.losses)
			tf.summary.histogram("q_values_hist", self.pred)
			tf.summary.scalar("mean_q_value", tf.math.reduce_mean(self.pred))
			tf.summary.scalar("var_q_value", tf.math.reduce_variance(self.pred))
			tf.summary.scalar("max_q_value", tf.reduce_max(self.pred))
			self.summaries = tf.summary.merge_all()



def train_Duelling_Double_DQN_PER(main_model, target_model, env, replay_buffer, policy, Beta, params):
	"""
	Train DQN agent which defined above

	:param main_model:
	:param target_model:
	:param env:
	:param params:
	:return:
	"""

	# log purpose
	losses, all_rewards, cnt_action = [], [], []
	episode_reward, index_episode = 0, 0

	# Create a glboal step variable
	global_step = tf.Variable(0, name='global_step', trainable=False)

	with tf.Session() as sess:
		# initialise all variables used in the model
		sess.run(tf.global_variables_initializer())
		state = env.reset()
		start = time.time()
		global_step = sess.run(tf.train.get_global_step())
		for frame_idx in range(1, params.num_frames + 1):
			action = policy.select_action(sess, main_model, state.reshape(params.state_reshape))
			cnt_action.append(action)

			next_state, reward, done, _ = env.step(action)
			replay_buffer.add(state, action, reward, next_state, done)

			state = next_state
			episode_reward += reward
			global_step += 1

			if done:
				index_episode += 1
				policy.index_episode = index_episode
				state = env.reset()
				all_rewards.append(episode_reward)

				if frame_idx > params.learning_start and len(replay_buffer) > params.batch_size:
					# ===== Prioritised Experience Replay part =====
					# PER returns: state, action, reward, next_state, done, weights(a weight for a timestep), indices(indices for a batch of timesteps)
					states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(params.batch_size, Beta.get_value(frame_idx))

					# ===== Double DQN part =====
					next_Q_main = main_model.predict(sess, next_states)
					next_Q = target_model.predict(sess, next_states)
					Y = rewards + params.gamma * next_Q[
						np.arange(params.batch_size), np.argmax(next_Q_main, axis=1)] * np.logical_not(dones)

					# we get a batch of loss for updating PER memory
					loss, batch_loss = main_model.update(sess, states, actions, Y)


					# add noise to the priorities
					batch_loss = np.abs(batch_loss) + params.prioritized_replay_noise

					# Update a prioritised replay buffer using a batch of losses associated with each timestep
					replay_buffer.update_priorities(indices, batch_loss)

					# Logging and refreshing log purpose values
					losses.append(loss)
					logging(frame_idx, params.num_frames, index_episode, time.time()-start, episode_reward, np.mean(loss), policy.current_epsilon(), cnt_action)

					episode_summary = tf.Summary()
					episode_summary.value.add(simple_value=episode_reward, node_name="episode_reward",
											  tag="episode_reward")
					episode_summary.value.add(simple_value=index_episode, node_name="episode_length",
											  tag="episode_length")
					main_model.summary_writer.add_summary(episode_summary, global_step)

					episode_reward = 0
					cnt_action = []
					start = time.time()

				if np.random.rand() > 0.5:
					# soft update means we partially add the original weights of target model instead of completely
					# sharing the weights among main and target models
					if params.update_hard_or_soft == "hard":
						sync_main_target(sess, target_model, main_model)
					elif params.update_hard_or_soft == "soft":
						soft_target_model_update(sess, target_model, main_model, tau=params.soft_update_tau)

	return all_rewards, losses