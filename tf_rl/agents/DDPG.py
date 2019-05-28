import numpy as np
import tensorflow as tf
from copy import deepcopy
from tf_rl.common.random_process import OrnsteinUhlenbeckProcess

class DDPG:
	def __init__(self, actor, critic, num_action, params):
		self.params = params
		self.num_action = num_action
		self.eval_flg = False
		self.index_timestep = 0
		self.actor = actor(num_action)
		self.critic = critic(1)
		self.target_actor  = deepcopy(self.actor)
		self.target_critic = deepcopy(self.critic)
		self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
		self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
		self.random_process = OrnsteinUhlenbeckProcess(size=self.num_action, theta=0.15, mu=0.0, sigma=0.2)

		# we don't save the models as in DQN.... sorry, you can implement the save/checkpoint tho.

	def predict(self, state):
		action = self.actor(tf.convert_to_tensor(state[None, :], dtype=tf.float32)).numpy()[0] + self.random_process.sample()
		return np.clip(action, self.params.action_range[0], self.params.action_range[1])

	@tf.contrib.eager.defun
	def inner_update_critic(self, next_Q, q_values, rewards, dones):
		"""
		To make this part compatible with Tensorflow Graph execution, we separate the operations in here

		:param next_Q:
		:param q_values:
		:param rewards:
		:param dones:
		:return:
		"""
		Y = tf.math.multiply(self.params.gamma, next_Q)
		Y = tf.math.multiply(Y, (1. - tf.cast(dones, tf.float32)))
		Y = tf.math.add(tf.cast(rewards, tf.float32), Y)

		# MSE or huber_loss
		critic_loss = tf.losses.mean_squared_error(Y, q_values)
		# critic_loss = tf.math.reduce_mean(tf.losses.huber_loss(Y, q_values, reduction=tf.losses.Reduction.NONE))
		weight_decay = tf.add_n([tf.nn.l2_loss(var) for var in self.critic.trainable_weights])*self.params.L2_reg
		return tf.math.add(critic_loss, weight_decay)

	def update(self, states, actions, rewards, next_states, dones):
		"""
		Update methods for Actor and Critic
		please refer to https://arxiv.org/pdf/1509.02971.pdf about the details

		"""

		# Update Critic
		self.index_timestep = tf.train.get_global_step()
		with tf.GradientTape() as tape:
			# calculate Q-values
			target_actions = self.target_actor(tf.convert_to_tensor(next_states[None, :], dtype=tf.float32))[0]

			# critic takes as input states, actions so that we combine them before passing them
			next_Q = self.target_critic(tf.cast(tf.concat([next_states, target_actions], axis=-1), dtype=tf.float32)).numpy().flatten()
			q_values = self.critic(tf.cast(tf.concat([states, actions], axis=-1), dtype=tf.float32)).numpy().flatten()

			# compute the loss in Graph Execution
			critic_loss = self.inner_update_critic(next_Q, q_values, rewards, dones)

		# get gradients
		critic_grads = tape.gradient(critic_loss, self.critic.trainable_weights)

		# apply processed gradients to the network
		self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

		# Update Actor
		with tf.GradientTape() as tape:
			# compute action
			target_actions = self.actor(tf.convert_to_tensor(states, dtype=tf.float32))
			q_values = self.critic(tf.cast(tf.concat([states, target_actions], axis=-1), dtype=tf.float32))
			actor_loss = -tf.math.reduce_mean(q_values)

		# get gradients
		actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)

		# apply processed gradients to the network
		self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_weights))
		# print(np.sum(critic_loss), np.sum(actor_loss))
		return np.sum(critic_loss + actor_loss)