import numpy as np
import tensorflow as tf
from copy import deepcopy

class HER_DDPG:
	"""
	DDPG for Hindsight Experience Replay

	"""

	def __init__(self, actor, critic, num_action, params, o_norm, g_norm):
		self.params = params
		self.num_action = num_action
		self.clip_target = 1 / (1 - self.params.gamma)
		self.eval_flg = False
		self.index_timestep = 0
		self.actor = actor(num_action)
		self.critic = critic(1)
		self.target_actor = deepcopy(self.actor)
		self.target_critic = deepcopy(self.critic)
		self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)  # openai baselines
		self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)  # openai baselines

		self.o_norm = o_norm
		self.g_norm = g_norm

	#  TODO: implement the checkpoints for model

	def predict(self, obs, g):
		obs = self.o_norm.normalise(obs)
		g = self.g_norm.normalise(g)
		state = np.concatenate([obs, g], axis=-1)
		state = np.expand_dims(state, axis=0).astype(np.float32)
		action = self._select_action(tf.constant(state))
		return action.numpy()[0] * self.params.max_action

	@tf.contrib.eager.defun(autograph=False)
	def _select_action(self, state):
		return self.actor(state)

	def update(self, transitions):
		obs = self.o_norm.normalise(transitions['obs'])
		g = self.g_norm.normalise(transitions['g'])
		states = np.concatenate([obs, g], axis=-1)
		next_obs = self.o_norm.normalise(transitions['obs_next'])
		next_states = np.concatenate([next_obs, g], axis=-1)
		actions = transitions['actions']
		rewards = transitions['r'].flatten()

		states = np.array(states, dtype=np.float32)
		next_states = np.array(next_states, dtype=np.float32)
		actions = np.array(actions, dtype=np.float32)
		rewards = np.array(rewards, dtype=np.float32)
		return self._inner_update(states, actions, rewards, next_states)

	@tf.contrib.eager.defun(autograph=False)
	def _inner_update(self, states, actions, rewards, next_states):
		self.index_timestep = tf.train.get_global_step()
		# Update Critic
		with tf.GradientTape() as tape:
			# critic takes as input states, actions so that we combine them before passing them
			next_Q = self.target_critic(next_states, self.target_actor(next_states) / self.params.max_action)
			q_values = self.critic(states, actions / self.params.max_action)

			# compute the target discounted Q(s', a')
			Y = rewards + self.params.gamma * tf.reshape(next_Q, [-1])
			Y = tf.clip_by_value(Y, -self.clip_target, 0)
			Y = tf.stop_gradient(Y)

			# Compute critic loss(MSE or huber_loss)
			critic_loss = tf.losses.mean_squared_error(Y, tf.reshape(q_values, [-1]))

		# get gradients
		critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

		# apply processed gradients to the network
		self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

		# Update Actor
		with tf.GradientTape() as tape:
			action = self.actor(states)
			actor_loss = -tf.math.reduce_mean(self.critic(states, action))
			# this is where HER's original operation comes in to penalise the excessive magnitude of action
			actor_loss += self.params.action_l2 * tf.math.reduce_mean(tf.math.square(action / self.params.max_action))

		# get gradients
		actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

		# apply processed gradients to the network
		self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
		return np.sum(critic_loss + actor_loss)


class HER_DDPG_debug:
	"""
	DDPG for Hindsight Experience Replay

	"""

	def __init__(self, actor, critic, num_action, params, o_norm, g_norm):
		self.params = params
		self.num_action = num_action
		self.eval_flg = False
		self.clip_target = 1 / (1 - self.params.gamma)
		self.index_timestep = 0
		self.actor = actor(num_action)
		self.critic = critic(1)
		self.target_actor = deepcopy(self.actor)
		self.target_critic = deepcopy(self.critic)
		self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)  # openai baselines
		self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)  # openai baselines

		self.o_norm = o_norm
		self.g_norm = g_norm

	#  TODO: implement the checkpoints for model

	def predict(self, obs, g):
		obs = self.o_norm.normalise(obs)
		g = self.g_norm.normalise(g)
		state = np.concatenate([obs, g], axis=-1)
		state = np.expand_dims(state, axis=0).astype(np.float32)
		action = self._select_action(tf.constant(state))
		return action.numpy()[0]

	@tf.contrib.eager.defun(autograph=False)
	def _select_action(self, state):
		return self.actor(state)

	def update(self, transitions):
		obs = self.o_norm.normalise(transitions['obs'])
		g = self.g_norm.normalise(transitions['g'])
		states = np.concatenate([obs, g], axis=-1)
		next_obs = self.o_norm.normalise(transitions['obs_next'])
		next_states = np.concatenate([next_obs, g], axis=-1)
		actions = transitions['actions']
		rewards = transitions['r'].flatten()

		"""
		If the learning didn't go well, open this part and compare to the baselines or other repos
		And make sure that the normaliser works fine!! sometimes, it didn't update internal states
		so that nothing hasn't been normalised... yep, it occurred to me.		
		"""

		# print(self.o_norm._sum, self.o_norm._count)
		# print(self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std)
		# print(np.mean(states), np.mean(next_states), np.mean(actions), np.mean(rewards))

		states = np.array(states, dtype=np.float32)
		next_states = np.array(next_states, dtype=np.float32)
		actions = np.array(actions, dtype=np.float32)
		rewards = np.array(rewards, dtype=np.float32)
		return self._inner_update(states, actions, rewards, next_states)


	# @tf.contrib.eager.defun(autograph=False)
	def _inner_update(self, states, actions, rewards, next_states):
		self.index_timestep = tf.train.get_global_step()
		# Update Critic
		with tf.GradientTape() as tape:
			# critic takes as input states, actions so that we combine them before passing them
			next_Q = self.target_critic(next_states, self.target_actor(next_states))
			q_values = self.critic(states, actions)

			# compute the target discounted Q(s', a')
			Y = rewards + self.params.gamma * tf.reshape(next_Q, [-1])
			Y = tf.clip_by_value(Y, -self.clip_target, 0)
			Y = tf.stop_gradient(Y)

			# Compute critic loss(MSE or huber_loss)
			critic_loss = tf.losses.mean_squared_error(Y, tf.reshape(q_values, [-1]))

		# get gradients
		critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

		# apply processed gradients to the network
		self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

		# Update Actor
		with tf.GradientTape() as tape:
			action = self.actor(states)
			actor_loss = -tf.math.reduce_mean(self.critic(states, action))
			# this is where HER's original operation comes in to penalise the excessive magnitude of action
			actor_loss += self.params.action_l2 * tf.math.reduce_mean(tf.math.square(action / self.params.max_action))

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
