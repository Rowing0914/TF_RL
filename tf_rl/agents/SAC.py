import numpy as np
import tensorflow as tf
from copy import deepcopy

class SAC:
	def __init__(self, actor, critic, num_action, params):
		self.params = params
		self.num_action = num_action
		self.eval_flg = False
		self.index_timestep = 0
		self.actor = actor(num_action)
		self.critic = critic(1)
		self.target_critic = deepcopy(self.critic)
		self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4) # used as in paper
		self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4) # used as in paper

		#  TODO: implement the checkpoints for model

		# TODO: make this available to construct graph when this class is being initialised
		# tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32)

	def predict(self, state):
		"""
		As mentioned in the topic of `policy evaluation` at sec5.2(`ablation study`) in the paper,
		for evaluation phase, using a deterministic action(choosing the mean of the policy dist) works better than
		stochastic one(Gaussian Policy).
		"""
		state = np.expand_dims(state, axis=0).astype(np.float32)
		if self.eval_flg:
			_, _, action = self._select_action(tf.constant(state))
		else:
			action, _, _ = self._select_action(tf.constant(state))
		return action.numpy()[0]

	@tf.contrib.eager.defun(autograph=False)
	def _select_action(self, state):
		return self.actor(state)

	def update(self, states, actions, rewards, next_states, dones):
		states = np.array(states, dtype=np.float32)
		next_states = np.array(next_states, dtype=np.float32)
		actions = np.array(actions, dtype=np.float32)
		rewards = np.array(rewards, dtype=np.float32)
		dones = np.array(dones, dtype=np.float32)
		return self._inner_update(states, actions, rewards, next_states, dones)


	@tf.contrib.eager.defun(autograph=False)
	def _inner_update(self, states, actions, rewards, next_states, dones):
		self.index_timestep = tf.train.get_global_step()
		# Update Critic
		with tf.GradientTape() as tape:
			# critic takes as input states, actions so that we combine them before passing them
			next_action, next_state_log_pi, _= self.actor(next_states)
			next_Q1, next_Q2 = self.target_critic(next_states, next_action)
			min_next_Q_target = tf.math.minimum(next_Q1, next_Q2) - self.params.alpha * next_state_log_pi
			q1, q2 = self.critic(states, actions)

			# compute the target discounted Q(s', a')
			Y = rewards + self.params.gamma * tf.reshape(min_next_Q_target, [-1]) * (1. - dones)
			Y = tf.stop_gradient(Y)

			# Compute critic loss
			critic_loss_q1 = tf.losses.mean_squared_error(Y, tf.reshape(q1, [-1]))
			critic_loss_q2 = tf.losses.mean_squared_error(Y, tf.reshape(q2, [-1]))

		critic_grads = tape.gradient([critic_loss_q1, critic_loss_q2], self.critic.trainable_variables)
		self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

		# Update Actor
		with tf.GradientTape() as tape:
			action, log_pi, _ = self.actor(states)
			q1, q2 = self.critic(states, action)
			actor_loss = -tf.math.reduce_mean( ( (self.params. alpha * log_pi) - tf.math.minimum(q1, q2)) )

		# get gradients
		actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

		# apply processed gradients to the network
		self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
		return tf.math.reduce_sum(critic_loss_q1 + critic_loss_q2 + actor_loss)


class SAC_debug:
	def __init__(self, actor, critic, num_action, params):
		self.params = params
		self.num_action = num_action
		self.eval_flg = False
		self.index_timestep = 0
		self.actor = actor(num_action)
		self.critic = critic(1)
		self.target_critic = deepcopy(self.critic)
		self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4) # used as in paper
		self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4) # used as in paper

		#  TODO: implement the checkpoints for model

	def predict(self, state):
		"""
		As mentioned in the topic of `policy evaluation` at sec5.2(`ablation study`) in the paper,
		for evaluation phase, using a deterministic action(choosing the mean of the policy dist) works better than
		stochastic one(Gaussian Policy).
		"""
		state = np.expand_dims(state, axis=0).astype(np.float32)
		if self.eval_flg:
			_, _, action = self._select_action(tf.constant(state))
		else:
			action, _, _ = self._select_action(tf.constant(state))
		return action.numpy()[0]

	# @tf.contrib.eager.defun(autograph=False)
	def _select_action(self, state):
		return self.actor(state)

	def update(self, states, actions, rewards, next_states, dones):
		states = np.array(states, dtype=np.float32)
		next_states = np.array(next_states, dtype=np.float32)
		actions = np.array(actions, dtype=np.float32)
		rewards = np.array(rewards, dtype=np.float32)
		dones = np.array(dones, dtype=np.float32)
		return self._inner_update(states, actions, rewards, next_states, dones)


	# @tf.contrib.eager.defun(autograph=False)
	def _inner_update(self, states, actions, rewards, next_states, dones):
		self.index_timestep = tf.train.get_global_step()
		# Update Critic
		with tf.GradientTape() as tape:
			# critic takes as input states, actions so that we combine them before passing them
			next_action, next_state_log_pi, _= self.actor(next_states)
			next_Q1, next_Q2 = self.target_critic(next_states, next_action)
			next_Q = tf.math.minimum(next_Q1, next_Q2) - self.params.alpha * next_state_log_pi
			q1, q2 = self.critic(states, actions)

			# compute the target discounted Q(s', a')
			Y = rewards + self.params.gamma * tf.reshape(next_Q, [-1]) * (1. - dones)
			Y = tf.stop_gradient(Y)

			# Compute critic loss
			critic_loss_q1 = tf.losses.mean_squared_error(Y, tf.reshape(q1, [-1]))
			critic_loss_q2 = tf.losses.mean_squared_error(Y, tf.reshape(q2, [-1]))

		critic_grads = tape.gradient([critic_loss_q1, critic_loss_q2], self.critic.trainable_variables)
		self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

		# Update Actor
		with tf.GradientTape() as tape:
			action, log_pi, _ = self.actor(states)
			q1, q2 = self.critic(states, action)
			actor_loss = -tf.math.reduce_mean( ( (self.params. alpha * log_pi) - tf.math.minimum(q1, q2)) )

		# get gradients
		actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

		# apply processed gradients to the network
		self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

		tf.contrib.summary.histogram("Y", Y, step=self.index_timestep)
		tf.contrib.summary.scalar("critic_loss_q1", critic_loss_q1, step=self.index_timestep)
		tf.contrib.summary.scalar("critic_loss_q2", critic_loss_q2, step=self.index_timestep)
		tf.contrib.summary.scalar("actor_loss", actor_loss, step=self.index_timestep)
		tf.contrib.summary.scalar("mean_next_Q", tf.math.reduce_mean(next_Q), step=self.index_timestep)
		tf.contrib.summary.scalar("max_next_Q", tf.math.reduce_max(next_Q), step=self.index_timestep)
		tf.contrib.summary.scalar("mean_q1", tf.math.reduce_mean(q1), step=self.index_timestep)
		tf.contrib.summary.scalar("mean_q2", tf.math.reduce_mean(q2), step=self.index_timestep)
		tf.contrib.summary.scalar("max_q1", tf.math.reduce_max(q1), step=self.index_timestep)
		tf.contrib.summary.scalar("max_q2", tf.math.reduce_max(q2), step=self.index_timestep)

		# print(q1.numpy(), q2.numpy(), critic_loss_q1.numpy(), critic_loss_q2.numpy(), actor_loss.numpy())
		return tf.math.reduce_sum(critic_loss_q1 + critic_loss_q2 + actor_loss)
