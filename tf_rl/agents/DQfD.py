import tensorflow as tf
from tf_rl.common.utils import *
from tf_rl.agents.core import Agent_atari, Agent_cartpole


class DQfD_atari(Agent_atari):
	"""
    DQfD
    """

	def __init__(self, model, optimizer, loss_fn, grad_clip_fn, num_action, params):
		self.params = params
		self.num_action = num_action
		self.pretrain_flag = True
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

		# We divide the grayscale pixel values by 255 here rather than storing
		# normalized values because uint8s are 4x cheaper to store than float32s.
		states, next_states = states / 255., next_states / 255.

		s, s_n = tf.split(states, 2, axis=-1)
		a, a_n = tf.split(actions, 2, axis=-1)
		r, r_n = tf.split(rewards, 2, axis=-1) # reward is already discounted in replay_buffer
		ns, ns_n = tf.split(next_states, 2, axis=-1)
		d, d_n = tf.split(dones, 2, axis=-1)
		a_e, a_l = tf.split(a, 2, axis=1)
		a_e_n, a_l_n = tf.split(a_n, 2, axis=1)

		# flatte them
		r, r_n = tf.reshape(r, [-1]), tf.reshape(r_n, [-1])
		d, d_n = tf.reshape(d, [-1]), tf.reshape(d_n, [-1])
		a_e, a_l = tf.reshape(a_e, [-1]), tf.reshape(a_l, [-1])
		a_e_n, a_l_n = tf.reshape(a_e_n, [-1]), tf.reshape(a_l_n, [-1])

		# ===== make sure to fit all process to compute gradients within this Tape context!! =====
		with tf.GradientTape() as tape:
			one_step_loss = self._one_step_loss(s, a_e, r, ns, d)
			n_step_loss = self._n_step_loss(s, s_n, a_e_n, r_n, d)
			large_margin_clf_loss = self._large_margin_clf_loss(a_e, a_l)
			l2_loss = tf.add_n(self.main_model.losses) * self.params.L2_reg

			# combined_loss = one_step_loss + lambda_1*n_step_loss + lambda_2*large_margin_clf_loss + lambda_3*l2_loss
			if self.pretrain_flag:
				combined_loss = one_step_loss + 1.0 * n_step_loss + 1.0 * large_margin_clf_loss + (10 ** (-5)) * l2_loss
			else:
				combined_loss = one_step_loss + 1.0 * n_step_loss + 0.0 * large_margin_clf_loss + (10 ** (-5)) * l2_loss

			# TODO: check if this is really correct..
			loss = tf.math.reduce_sum(combined_loss)

		# get gradients
		grads = tape.gradient(loss, self.main_model.trainable_weights)

		# clip gradients
		grads = self.grad_clip_fn(grads)

		# apply processed gradients to the network
		self.optimizer.apply_gradients(zip(grads, self.main_model.trainable_weights))
		return loss, combined_loss

	def _one_step_loss(self, states, actions_e, rewards, next_states, dones):
		# calculate target: max_a Q(s_{t+1}, a_{t+1})
		next_Q_main = self.main_model(next_states)
		next_Q = self.target_model(next_states)
		# calculate Q(s,a)
		q_values = self.main_model(states)
		idx_flattened = tf.range(0, tf.shape(next_Q)[0]) * tf.shape(next_Q)[1] + tf.cast(
			tf.math.argmax(next_Q_main, axis=-1), tf.int32)

		# passing [-1] to tf.reshape means flatten the array
		# using tf.gather, associate Q-values with the executed actions
		chosen_next_q = tf.gather(tf.reshape(next_Q, [-1]), idx_flattened)

		Y = rewards + self.params.gamma * chosen_next_q * (1. - dones)
		Y = tf.stop_gradient(Y)

		# get the q-values which is associated with actually taken actions in a game
		actions_one_hot = tf.one_hot(actions_e, self.num_action, 1.0, 0.0)
		chosen_q = tf.math.reduce_sum(tf.math.multiply(actions_one_hot, q_values), reduction_indices=1)
		return tf.math.subtract(Y, chosen_q)

	def _n_step_loss(self, states, states_n, actions_e, rewards_n, dones):
		# calculate target: max_a Q(s_{t+n}, a_{t+n})
		n_step_Q_main = self.main_model(states_n)
		n_step_Q = self.target_model(states_n)
		# calculate Q(s,a)
		q_values = self.main_model(states)
		idx_flattened = tf.range(0, tf.shape(n_step_Q)[0]) * tf.shape(n_step_Q)[1] + tf.cast(
			tf.math.argmax(n_step_Q_main, axis=-1), tf.int32)

		# passing [-1] to tf.reshape means flatten the array
		# using tf.gather, associate Q-values with the executed actions
		action_probs = tf.gather(tf.reshape(n_step_Q, [-1]), idx_flattened)

		# n-step discounted reward
		#TODO: check if this is correct
		G = tf.math.reduce_sum([self.params.gamma ** i * rewards_n for i in range(self.params.n_step)])

		# TD-target
		# TODO: think how to take `dones` into account in TD-target
		Y = G + self.params.gamma ** self.params.n_step * action_probs * (1. - dones)

		# get the q-values which is associated with actually taken actions in a game
		actions_one_hot = tf.one_hot(actions_e[-1], self.num_action, 1.0, 0.0)
		chosen_q = tf.math.reduce_sum(tf.math.multiply(actions_one_hot, q_values), reduction_indices=1)
		return tf.math.subtract(Y, chosen_q)

	def _large_margin_clf_loss(self, a_e, a_l):
		"""
		Logic is as below

		if a_e == a_l:
			return 0
		else:
			return 0.8

		:param a_e:
		:param a_l:
		:return:
		"""
		result = (a_e != a_l)
		return result * 0.8



class DQfD_cartpole(Agent_cartpole):
	"""
    DQfD
    """

	def __init__(self, model, optimizer, loss_fn, grad_clip_fn, num_action, params):
		self.params = params
		self.num_action = num_action
		self.pretrain_flag = True
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

		s, s_n = tf.split(states, 2, axis=-1)
		a, a_n = tf.split(actions, 2, axis=-1)
		r, r_n = tf.split(rewards, 2, axis=-1) # reward is already discounted in replay_buffer
		ns, ns_n = tf.split(next_states, 2, axis=-1)
		d, d_n = tf.split(dones, 2, axis=-1)
		a_e, a_l = tf.split(a, 2, axis=1)
		a_e_n, a_l_n = tf.split(a_n, 2, axis=1)

		# flatte them
		r, r_n = tf.reshape(r, [-1]), tf.reshape(r_n, [-1])
		d, d_n = tf.reshape(d, [-1]), tf.reshape(d_n, [-1])
		a_e, a_l = tf.reshape(a_e, [-1]), tf.reshape(a_l, [-1])
		a_e_n, a_l_n = tf.reshape(a_e_n, [-1]), tf.reshape(a_l_n, [-1])

		# ===== make sure to fit all process to compute gradients within this Tape context!! =====
		with tf.GradientTape() as tape:
			one_step_loss = self._one_step_loss(s, a_e, r, ns, d)
			n_step_loss = self._n_step_loss(s, s_n, a_e_n, r_n, d)
			large_margin_clf_loss = self._large_margin_clf_loss(a_e, a_l)
			l2_loss = tf.add_n(self.main_model.losses) * self.params.L2_reg

			# combined_loss = one_step_loss + lambda_1*n_step_loss + lambda_2*large_margin_clf_loss + lambda_3*l2_loss
			if self.pretrain_flag:
				combined_loss = one_step_loss + 1.0 * n_step_loss + 1.0 * large_margin_clf_loss + (10 ** (-5)) * l2_loss
			else:
				combined_loss = one_step_loss + 1.0 * n_step_loss + 0.0 * large_margin_clf_loss + (10 ** (-5)) * l2_loss

			# TODO: check if this is really correct..
			loss = tf.math.reduce_sum(combined_loss)

		# get gradients
		grads = tape.gradient(loss, self.main_model.trainable_weights)

		# clip gradients
		grads = self.grad_clip_fn(grads)

		# apply processed gradients to the network
		self.optimizer.apply_gradients(zip(grads, self.main_model.trainable_weights))
		return loss, combined_loss

	def _one_step_loss(self, states, actions_e, rewards, next_states, dones):
		# calculate target: max_a Q(s_{t+1}, a_{t+1})
		next_Q_main = self.main_model(next_states)
		next_Q = self.target_model(next_states)
		# calculate Q(s,a)
		q_values = self.main_model(states)
		idx_flattened = tf.range(0, tf.shape(next_Q)[0]) * tf.shape(next_Q)[1] + tf.cast(
			tf.math.argmax(next_Q_main, axis=-1), tf.int32)

		# passing [-1] to tf.reshape means flatten the array
		# using tf.gather, associate Q-values with the executed actions
		chosen_next_q = tf.gather(tf.reshape(next_Q, [-1]), idx_flattened)

		Y = rewards + self.params.gamma * chosen_next_q * (1. - dones)
		Y = tf.stop_gradient(Y)

		# get the q-values which is associated with actually taken actions in a game
		actions_one_hot = tf.one_hot(actions_e, self.num_action, 1.0, 0.0)
		chosen_q = tf.math.reduce_sum(tf.math.multiply(actions_one_hot, q_values), reduction_indices=1)
		return tf.math.subtract(Y, chosen_q)

	def _n_step_loss(self, states, states_n, actions_e, rewards_n, dones):
		# calculate target: max_a Q(s_{t+n}, a_{t+n})
		n_step_Q_main = self.main_model(states_n)
		n_step_Q = self.target_model(states_n)
		# calculate Q(s,a)
		q_values = self.main_model(states)
		idx_flattened = tf.range(0, tf.shape(n_step_Q)[0]) * tf.shape(n_step_Q)[1] + tf.cast(
			tf.math.argmax(n_step_Q_main, axis=-1), tf.int32)

		# passing [-1] to tf.reshape means flatten the array
		# using tf.gather, associate Q-values with the executed actions
		action_probs = tf.gather(tf.reshape(n_step_Q, [-1]), idx_flattened)

		# n-step discounted reward
		#TODO: check if this is correct
		G = tf.math.reduce_sum([self.params.gamma ** i * rewards_n for i in range(self.params.n_step)])

		# TD-target
		# TODO: think how to take `dones` into account in TD-target
		Y = G + self.params.gamma ** self.params.n_step * action_probs * (1. - dones)

		# get the q-values which is associated with actually taken actions in a game
		actions_one_hot = tf.one_hot(actions_e[-1], self.num_action, 1.0, 0.0)
		chosen_q = tf.math.reduce_sum(tf.math.multiply(actions_one_hot, q_values), reduction_indices=1)
		return tf.math.subtract(Y, chosen_q)

	def _large_margin_clf_loss(self, a_e, a_l):
		"""
		Logic is as below

		if a_e == a_l:
			return 0
		else:
			return 0.8

		:param a_e:
		:param a_l:
		:return:
		"""
		result = (a_e != a_l)
		return result * 0.8