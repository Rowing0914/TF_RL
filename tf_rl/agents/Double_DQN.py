import tensorflow as tf
from tf_rl.common.utils import create_checkpoint
from tf_rl.agents.core import Agent_atari, Agent_cartpole


class Double_DQN(Agent_atari):
	"""
	A complete Double_DQN model for training of Atari or cartpole-pixel
	and don't use this for debugging purpose.
	"""

	def __init__(self, model, optimizer, loss_fn, grad_clip_fn, num_action, params):
		self.params = params
		self.num_action = num_action
		self.grad_clip_fn = grad_clip_fn
		self.loss_fn = loss_fn
		self.eval_flg = False
		self.index_timestep = 0
		self.main_model = model(num_action)
		self.target_model = model(num_action)
		self.optimizer = optimizer
		self.manager = create_checkpoint(model=self.main_model,
										 optimizer=self.optimizer,
										 model_dir=params.model_dir)

	# TODO: make this available to construct graph when this class is being initialised
	# tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32)

	@tf.contrib.eager.defun(autograph=False)
	def _select_action(self, state):
		return self.main_model(state)

	@tf.contrib.eager.defun(autograph=False)
	def _inner_update(self, states, actions, rewards, next_states, dones):
		# get the current global-timestep
		self.index_timestep = tf.train.get_global_step()

		# We divide the grayscale pixel values by 255 here rather than storing
		# normalized values because uint8s are 4x cheaper to store than float32s.
		states, next_states = states / 255., next_states / 255.

		# ===== make sure to fit all process to compute gradients within this Tape context!! =====
		with tf.GradientTape() as tape:
			next_Q_main = self.main_model(next_states)
			next_Q = self.target_model(next_states)
			q_values = self.main_model(states)
			idx_flattened = tf.range(0, tf.shape(next_Q)[0]) * tf.shape(next_Q)[1] + tf.cast(
				tf.math.argmax(next_Q_main, axis=-1), tf.int32)

			# passing [-1] to tf.reshape means flatten the array
			# using tf.gather, associate Q-values with the executed actions
			chosen_next_q = tf.gather(tf.reshape(next_Q, [-1]), idx_flattened)

			Y = rewards + self.params.gamma * chosen_next_q * (1. - dones)
			Y = tf.stop_gradient(Y)

			# get the q-values which is associated with actually taken actions in a game
			actions_one_hot = tf.one_hot(actions, self.num_action, 1.0, 0.0)
			chosen_q = tf.math.reduce_sum(tf.math.multiply(actions_one_hot, q_values), reduction_indices=1)
			batch_loss = self.loss_fn(Y, chosen_q, reduction=tf.losses.Reduction.NONE)
			loss = tf.math.reduce_mean(batch_loss)

		# get gradients
		grads = tape.gradient(loss, self.main_model.trainable_weights)

		# clip gradients
		grads = self.grad_clip_fn(grads)

		# apply processed gradients to the network
		self.optimizer.apply_gradients(zip(grads, self.main_model.trainable_weights))
		return loss, batch_loss


class Double_DQN_cartpole(Agent_cartpole):
	"""
	A complete DQN model for training of cartpole not for debugging purpose
	"""

	def __init__(self, model, optimizer, loss_fn, grad_clip_fn, num_action, params):
		self.params = params
		self.num_action = num_action
		self.grad_clip_fn = grad_clip_fn
		self.loss_fn = loss_fn
		self.eval_flg = False
		self.index_timestep = 0
		self.main_model = model(num_action)
		self.target_model = model(num_action)
		self.optimizer = optimizer
		self.manager = create_checkpoint(model=self.main_model,
										 optimizer=self.optimizer,
										 model_dir=params.model_dir)

	@tf.contrib.eager.defun(autograph=False)
	def _select_action(self, state):
		return self.main_model(state)

	@tf.contrib.eager.defun(autograph=False)
	def _inner_update(self, states, actions, rewards, next_states, dones):
		# get the current global-timestep
		self.index_timestep = tf.train.get_global_step()

		# ===== make sure to fit all process to compute gradients within this Tape context!! =====
		with tf.GradientTape() as tape:
			next_Q_main = self.main_model(next_states)
			next_Q = self.target_model(next_states)
			q_values = self.main_model(states)
			idx_flattened = tf.range(0, tf.shape(next_Q)[0]) * tf.shape(next_Q)[1] + tf.cast(
				tf.math.argmax(next_Q_main, axis=-1), tf.int32)

			# passing [-1] to tf.reshape means flatten the array
			# using tf.gather, associate Q-values with the executed actions
			chosen_next_q = tf.gather(tf.reshape(next_Q, [-1]), idx_flattened)

			Y = rewards + self.params.gamma * chosen_next_q * (1. - dones)
			Y = tf.stop_gradient(Y)

			# get the q-values which is associated with actually taken actions in a game
			actions_one_hot = tf.one_hot(actions, self.num_action, 1.0, 0.0)
			chosen_q = tf.math.reduce_sum(tf.math.multiply(actions_one_hot, q_values), reduction_indices=1)
			batch_loss = self.loss_fn(Y, chosen_q, reduction=tf.losses.Reduction.NONE)
			loss = tf.math.reduce_mean(batch_loss)

		# get gradients
		grads = tape.gradient(loss, self.main_model.trainable_weights)

		# clip gradients
		grads = self.grad_clip_fn(grads)

		# apply processed gradients to the network
		self.optimizer.apply_gradients(zip(grads, self.main_model.trainable_weights))
		return loss, batch_loss


class Double_DQN_debug(Agent_atari):
	"""
	A complete Double_DQN model for training of Atari or cartpole-pixel
	and don't use this for debugging purpose.
	"""

	def __init__(self, model, optimizer, loss_fn, grad_clip_fn, num_action, params):
		self.params = params
		self.num_action = num_action
		self.grad_clip_fn = grad_clip_fn
		self.loss_fn = loss_fn
		self.eval_flg = False
		self.index_timestep = 0
		self.main_model = model(num_action)
		self.target_model = model(num_action)
		self.optimizer = optimizer
		self.manager = create_checkpoint(model=self.main_model,
										 optimizer=self.optimizer,
										 model_dir=params.model_dir)

	# TODO: make this available to construct graph when this class is being initialised
	# tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32)

	# @tf.contrib.eager.defun(autograph=False)
	def _select_action(self, state):
		return self.main_model(state)

	# @tf.contrib.eager.defun(autograph=False)
	def _inner_update(self, states, actions, rewards, next_states, dones):
		# get the current global-timestep
		self.index_timestep = tf.train.get_global_step()

		# We divide the grayscale pixel values by 255 here rather than storing
		# normalized values because uint8s are 4x cheaper to store than float32s.
		states, next_states = states / 255., next_states / 255.

		# ===== make sure to fit all process to compute gradients within this Tape context!! =====
		with tf.GradientTape() as tape:
			next_Q_main = self.main_model(next_states)
			next_Q = self.target_model(next_states)
			q_values = self.main_model(states)
			idx_flattened = tf.range(0, tf.shape(next_Q)[0]) * tf.shape(next_Q)[1] + tf.cast(
				tf.math.argmax(next_Q_main, axis=-1), tf.int32)

			# passing [-1] to tf.reshape means flatten the array
			# using tf.gather, associate Q-values with the executed actions
			chosen_next_q = tf.gather(tf.reshape(next_Q, [-1]), idx_flattened)

			Y = rewards + self.params.gamma * chosen_next_q * (1. - dones)
			Y = tf.stop_gradient(Y)

			# get the q-values which is associated with actually taken actions in a game
			actions_one_hot = tf.one_hot(actions, self.num_action, 1.0, 0.0)
			chosen_q = tf.math.reduce_sum(tf.math.multiply(actions_one_hot, q_values), reduction_indices=1)
			batch_loss = self.loss_fn(Y, chosen_q, reduction=tf.losses.Reduction.NONE)
			loss = tf.math.reduce_mean(batch_loss)

		# get gradients
		grads = tape.gradient(loss, self.main_model.trainable_weights)

		# clip gradients
		grads = self.grad_clip_fn(grads)

		# apply processed gradients to the network
		self.optimizer.apply_gradients(zip(grads, self.main_model.trainable_weights))

		# for log purpose
		for index, grad in enumerate(grads):
			tf.contrib.summary.histogram("layer_grad_{}".format(index), grad, step=self.index_timestep)
		tf.contrib.summary.scalar("loss", loss, step=self.index_timestep)
		tf.contrib.summary.histogram("next_Q(TargetModel)", next_Q, step=self.index_timestep)
		tf.contrib.summary.histogram("q_values(MainModel)", next_Q, step=self.index_timestep)
		tf.contrib.summary.histogram("Y(target)", Y, step=self.index_timestep)
		tf.contrib.summary.scalar("mean_Y(target)", tf.math.reduce_mean(Y), step=self.index_timestep)
		tf.contrib.summary.scalar("var_Y(target)", tf.math.reduce_variance(Y), step=self.index_timestep)
		tf.contrib.summary.scalar("max_Y(target)", tf.math.reduce_max(Y), step=self.index_timestep)
		tf.contrib.summary.scalar("mean_q_value(TargetModel)", tf.math.reduce_mean(next_Q), step=self.index_timestep)
		tf.contrib.summary.scalar("var_q_value(TargetModel)", tf.math.reduce_variance(next_Q), step=self.index_timestep)
		tf.contrib.summary.scalar("max_q_value(TargetModel)", tf.math.reduce_max(next_Q), step=self.index_timestep)
		tf.contrib.summary.scalar("mean_q_value(MainModel)", tf.math.reduce_mean(q_values), step=self.index_timestep)
		tf.contrib.summary.scalar("var_q_value(MainModel)", tf.math.reduce_variance(q_values), step=self.index_timestep)
		tf.contrib.summary.scalar("max_q_value(MainModel)", tf.math.reduce_max(q_values), step=self.index_timestep)
		return loss, batch_loss
