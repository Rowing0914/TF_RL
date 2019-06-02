import numpy as np
import tensorflow as tf
from copy import deepcopy
from tf_rl.common.random_process import OrnsteinUhlenbeckProcess
from tf_rl.agents.core import Agent

class DDPG(Agent):
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
		self.random_process = OrnsteinUhlenbeckProcess(size=self.num_action, theta=0.15, mu=0.0, sigma=0.05)

		#  TODO: implement the checkpoints for model

	def predict(self, state):
		state = np.expand_dims(state, axis=0).astype(np.float32)
		action = self._select_action(tf.constant(state))
		return action.numpy()[0] + self.random_process.sample()

	@tf.contrib.eager.defun(autograph=False)
	def _select_action(self, state):
		return self.actor(state)

	def update(self, states, actions, rewards, next_states, dones):
		"""
		Update methods for Actor and Critic
		please refer to https://arxiv.org/pdf/1509.02971.pdf about the details

		"""
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
			next_Q = self.target_critic(next_states, self.target_actor(next_states))
			q_values = self.critic(states, actions)

			# compute the target discounted Q(s', a')
			Y = rewards + self.params.gamma * tf.reshape(next_Q, [-1]) * (1. - dones)
			Y = tf.stop_gradient(Y)

			# Compute critic loss(MSE or huber_loss) + L2 loss
			critic_loss = tf.losses.mean_squared_error(Y, tf.reshape(q_values, [-1])) + tf.add_n(self.critic.losses) * self.params.L2_reg
			# critic_loss = tf.math.reduce_mean(tf.losses.huber_loss(Y, q_values, reduction=tf.losses.Reduction.NONE)) + tf.add_n(self.critic.losses)*self.params.L2_reg

		# get gradients
		critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

		# apply processed gradients to the network
		self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

		# Update Actor
		with tf.GradientTape() as tape:
			actor_loss = -tf.math.reduce_mean(self.critic(states, self.actor(states)))

		# get gradients
		actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

		# apply processed gradients to the network
		self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
		return np.sum(critic_loss + actor_loss)


class HER_DDPG(Agent):
	"""
	DDPG for Hindsight Experience Replay

	"""
	def __init__(self, actor, critic, num_action, params):
		self.params = params
		self.num_action = num_action
		self.eval_flg = False
		self.index_timestep = 0
		self.actor = actor(num_action)
		self.critic = critic(1)
		self.target_actor  = deepcopy(self.actor)
		self.target_critic = deepcopy(self.critic)
		self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=0.001) # openai baselines
		self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=0.001) # openai baselines

		#  TODO: implement the checkpoints for model

	def predict(self, state):
		state = np.expand_dims(state, axis=0).astype(np.float32)
		action = self._select_action(tf.constant(state))
		return action.numpy()[0] * self.params.max_action

	@tf.contrib.eager.defun(autograph=False)
	def _select_action(self, state):
		return self.actor(state)

	@tf.contrib.eager.defun(autograph=False)
	def _inner_update(self, states, actions, rewards, next_states, dones):
		self.index_timestep = tf.train.get_global_step()
		# Update Critic
		with tf.GradientTape() as tape:
			# critic takes as input states, actions so that we combine them before passing them
			next_Q = self.target_critic(next_states, self.target_actor(next_states)/self.params.max_action)
			q_values = self.critic(states, actions/self.params.max_action)

			# compute the target discounted Q(s', a')
			Y = rewards + self.params.gamma * tf.reshape(next_Q, [-1]) * (1. - dones)
			Y = tf.stop_gradient(Y)

			# Compute critic loss(MSE or huber_loss) + L2 loss
			critic_loss = tf.losses.mean_squared_error(Y, tf.reshape(q_values, [-1]))
			# critic_loss = tf.math.reduce_mean(tf.losses.huber_loss(Y, q_values, reduction=tf.losses.Reduction.NONE)) + tf.add_n(self.critic.losses)*self.params.L2_reg

		# get gradients
		critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

		# apply processed gradients to the network
		self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

		# Update Actor
		with tf.GradientTape() as tape:
			action = self.actor(states)
			actor_loss = -tf.math.reduce_mean(self.critic(states, action))
			# this is where HER's original operation comes in to penalise the excessive magnitude of action
			actor_loss += self.params.action_l2 * tf.math.reduce_mean(tf.math.square(action/self.params.max_action))

		# get gradients
		actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

		# apply processed gradients to the network
		self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
		return np.sum(critic_loss + actor_loss)



class DDPG_debug(Agent):
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
		self.random_process = OrnsteinUhlenbeckProcess(size=self.num_action, theta=0.15, mu=0.0, sigma=0.05)

		#  TODO: implement the checkpoints for model

	def predict(self, state):
		state = np.expand_dims(state, axis=0).astype(np.float32)
		action = self._select_action(tf.constant(state))
		return action.numpy()[0] + self.random_process.sample()

	@tf.contrib.eager.defun(autograph=False)
	def _select_action(self, state):
		return self.actor(state)

	def update(self, states, actions, rewards, next_states, dones):
		"""
		Update methods for Actor and Critic
		please refer to https://arxiv.org/pdf/1509.02971.pdf about the details

		"""
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
			next_Q = self.target_critic(next_states, self.target_actor(next_states))
			q_values = self.critic(states, actions)

			# compute the target discounted Q(s', a')
			Y = rewards + self.params.gamma * tf.reshape(next_Q, [-1]) * (1. - dones)
			Y = tf.stop_gradient(Y)

			# Compute critic loss(MSE or huber_loss) + L2 loss
			critic_loss = tf.losses.mean_squared_error(Y, tf.reshape(q_values, [-1])) + tf.add_n(self.critic.losses) * self.params.L2_reg
			# critic_loss = tf.math.reduce_mean(tf.losses.huber_loss(Y, q_values, reduction=tf.losses.Reduction.NONE)) + tf.add_n(self.critic.losses)*self.params.L2_reg

		# get gradients
		critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

		# apply processed gradients to the network
		self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

		# Update Actor
		with tf.GradientTape() as tape:
			actor_loss = -tf.math.reduce_mean(self.critic(states, self.actor(states)))

		# get gradients
		actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

		# apply processed gradients to the network
		self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

		tf.contrib.summary.histogram("Y", Y, step=self.index_timestep)
		tf.contrib.summary.scalar("critic_loss", critic_loss, step=self.index_timestep)
		tf.contrib.summary.scalar("actor_loss", actor_loss, step=self.index_timestep)
		tf.contrib.summary.scalar("mean_next_Q", tf.math.reduce_mean(next_Q), step=self.index_timestep)
		tf.contrib.summary.scalar("max_next_Q", tf.math.reduce_max(next_Q), step=self.index_timestep)
		tf.contrib.summary.scalar("mean_q_value", tf.math.reduce_mean(q_values), step=self.index_timestep)
		tf.contrib.summary.scalar("max_q_value", tf.math.reduce_max(q_values), step=self.index_timestep)

		return np.sum(critic_loss + actor_loss)



class HER_DDPG_debug(Agent):
	"""
	DDPG for Hindsight Experience Replay

	"""
	def __init__(self, actor, critic, num_action, params):
		self.params = params
		self.num_action = num_action
		self.eval_flg = False
		self.index_timestep = 0
		self.actor = actor(num_action)
		self.critic = critic(1)
		self.target_actor  = deepcopy(self.actor)
		self.target_critic = deepcopy(self.critic)
		self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=0.001) # openai baselines
		self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=0.001) # openai baselines

		#  TODO: implement the checkpoints for model

	def predict(self, state):
		state = np.expand_dims(state, axis=0).astype(np.float32)
		action = self._select_action(tf.constant(state))
		return action.numpy()[0]

	@tf.contrib.eager.defun(autograph=False)
	def _select_action(self, state):
		return self.actor(state)

	# @tf.contrib.eager.defun(autograph=False)
	def _inner_update(self, states, actions, rewards, next_states, dones):
		self.index_timestep = tf.train.get_global_step()
		# Update Critic
		with tf.GradientTape() as tape:
			# critic takes as input states, actions so that we combine them before passing them
			next_Q = self.target_critic(next_states, self.target_actor(next_states))
			q_values = self.critic(states, actions)

			# compute the target discounted Q(s', a')
			Y = rewards + self.params.gamma * tf.reshape(next_Q, [-1]) * (1. - dones)
			Y = tf.stop_gradient(Y)

			# Compute critic loss(MSE or huber_loss) + L2 loss
			critic_loss = tf.losses.mean_squared_error(Y, tf.reshape(q_values, [-1]))
			# critic_loss = tf.math.reduce_mean(tf.losses.huber_loss(Y, q_values, reduction=tf.losses.Reduction.NONE)) + tf.add_n(self.critic.losses)*self.params.L2_reg

		# get gradients
		critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

		# apply processed gradients to the network
		self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

		# Update Actor
		with tf.GradientTape() as tape:
			action = self.actor(states)
			actor_loss = -tf.math.reduce_mean(self.critic(states, action))
			# this is where HER's original operation comes in to penalise the excessive magnitude of action
			actor_loss += self.params.action_l2 * tf.math.reduce_mean(tf.math.square(action/self.params.max_action))


		# get gradients
		actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

		# apply processed gradients to the network
		self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
		tf.contrib.summary.histogram("Y", Y, step=self.index_timestep)
		tf.contrib.summary.scalar("critic_loss", critic_loss, step=self.index_timestep)
		tf.contrib.summary.scalar("actor_loss", actor_loss, step=self.index_timestep)
		tf.contrib.summary.scalar("mean_next_Q", tf.math.reduce_mean(next_Q), step=self.index_timestep)
		tf.contrib.summary.scalar("max_next_Q", tf.math.reduce_max(next_Q), step=self.index_timestep)
		tf.contrib.summary.scalar("mean_q_value", tf.math.reduce_mean(q_values), step=self.index_timestep)
		tf.contrib.summary.scalar("max_q_value", tf.math.reduce_max(q_values), step=self.index_timestep)
		# print(critic_loss.numpy(), actor_loss.numpy())
		return np.sum(critic_loss + actor_loss)
