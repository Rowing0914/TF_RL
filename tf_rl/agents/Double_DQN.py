import numpy as np
import tensorflow as tf
from tf_rl.common.utils import huber_loss, ClipIfNotNone, AnnealingSchedule


class Double_DQN:
	"""
	Double_DQN
	"""

	def __init__(self, env_type, main_model, target_model, num_action, params, checkpoint_dir="../logs/models/DQN/"):
		self.env_type = env_type
		self.num_action = num_action
		self.params = params
		self.main_model = main_model(env_type, num_action)
		self.target_model = target_model(env_type, num_action)
		self.learning_rate = AnnealingSchedule(start=1e-2, end=1e-4, decay_steps=params.decay_steps,
											   decay_type="linear")  # learning rate decay!!
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate.get_value())
		# self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)

		# TF: checkpoint vs Saver => https://stackoverflow.com/questions/53569622/difference-between-tf-train-checkpoint-and-tf-train-saver
		self.checkpoint_dir = checkpoint_dir
		self.check_point = tf.train.Checkpoint(optimizer=self.optimizer,
											   model=self.main_model,
											   optimizer_step=tf.train.get_or_create_global_step())
		self.manager = tf.train.CheckpointManager(self.check_point, checkpoint_dir, max_to_keep=3)

	def predict(self, state):
		if self.env_type == "Atari":
			state = np.array(state).astype('float32') / 255.
		return self.main_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32)).numpy()[0]

	def update(self, states, actions, rewards, next_states, dones):
		# let's keep this for debug purpose!!
		# if you feel that the agent does not keep up with the global time-step, pls open this!
		# print("===== UPDATE ===== Train Step:{}".format(tf.train.get_or_create_global_step()))

		# get the current global-timestep
		self.index_timestep = tf.train.get_global_step()

		if self.env_type == "Atari":
			states, next_states = np.array(states).astype('float32') / 255., np.array(next_states).astype(
				'float32') / 255.

		# putting this outside the scope of GradientTape for safety purpose
		# we don't want to update the target model
		# calculate target: R + gamma * max_a Q(s',a')
		next_Q = self.target_model(tf.convert_to_tensor(next_states, dtype=tf.float32))

		# ===== make sure to fit all process to compute gradients within this Tape context!! =====
		with tf.GradientTape() as tape:
			# this is where Double DQN comes in!!
			# calculate target: R + gamma * max_a Q(s', max_a Q(s', a'; main_model); target_model)
			next_Q_main = self.main_model(tf.convert_to_tensor(next_states, dtype=tf.float32))
			idx_flattened = tf.range(0, tf.shape(next_Q)[0]) * tf.shape(next_Q)[1] + np.argmax(next_Q_main, axis=-1)

			# passing [-1] to tf.reshape means flatten the array
			# using tf.gather, associate Q-values with the executed actions
			action_probs = tf.gather(tf.reshape(next_Q, [-1]), idx_flattened)

			Y = rewards + self.params.gamma * action_probs * np.logical_not(dones)

			# calculate Q(s,a)
			q_values = self.main_model(tf.convert_to_tensor(states, dtype=tf.float32))

			# get the q-values which is associated with actually taken actions in a game
			actions_one_hot = tf.one_hot(actions, self.num_action, 1.0, 0.0)
			action_probs = tf.reduce_sum(actions_one_hot * q_values, reduction_indices=-1)

			if self.params.loss_fn == "huber_loss":
				# use huber loss
				batch_loss = huber_loss(tf.squared_difference(Y, action_probs))
				loss = tf.reduce_mean(batch_loss)
			elif self.params.loss_fn == "MSE":
				# use MSE
				batch_loss = tf.squared_difference(Y, action_probs)
				loss = tf.reduce_mean(batch_loss)
			else:
				assert False

		# get gradients
		grads = tape.gradient(loss, self.main_model.trainable_weights)

		# clip gradients
		if self.params.grad_clip_flg == "by_value":
			grads = [ClipIfNotNone(grad, -1., 1.) for grad in grads]
		elif self.params.grad_clip_flg == "norm":
			grads, _ = tf.clip_by_global_norm(grads, 5.0)
		elif self.params.grad_clip_flg == "None":
			pass

		# apply processed gradients to the network
		self.optimizer.apply_gradients(zip(grads, self.main_model.trainable_weights), global_step=self.index_timestep)

		# for log purpose
		for index, grad in enumerate(grads):
			tf.contrib.summary.histogram("layer_grad_{}".format(index), grad, step=self.index_timestep)
		tf.contrib.summary.scalar("loss", loss, step=self.index_timestep)
		tf.contrib.summary.histogram("next_Q", next_Q, step=self.index_timestep)
		tf.contrib.summary.scalar("mean_q_value", tf.math.reduce_mean(next_Q), step=self.index_timestep)
		tf.contrib.summary.scalar("var_q_value", tf.math.reduce_variance(next_Q), step=self.index_timestep)
		tf.contrib.summary.scalar("max_q_value", tf.reduce_max(next_Q), step=self.index_timestep)
		tf.contrib.summary.scalar("learning_rate", self.learning_rate.get_value(), step=self.index_timestep)

		return loss, batch_loss
