import numpy as np
import tensorflow as tf

class TRPO:
	def __init__(self, actor, critic, num_action, params):
		self.params = params
		self.num_action = num_action
		self.beta = 1
		self.beta_min = 1./20.
		self.beta_max = 20
		self.ksi = 10
		self.index_timestep = 0
		self.actor = actor(num_action)
		self.critic = critic(1)
		self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)  # used as in paper
		self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=1.5*1e-3)  # used as in paper

	#  TODO: implement the checkpoints for model

	def predict(self, state):
		state = np.expand_dims(state, axis=0).astype(np.float32)
		action = self._select_action(tf.constant(state))
		return action.numpy()[0]

	@tf.contrib.eager.defun(autograph=False)
	def _select_action(self, state):
		mean, std = self.actor(state)
		return tf.squeeze(mean + std * tf.random_normal(shape=tf.shape(mean)))

	def update(self, states, actions, returns, advantages, old_policy):
		states = np.array(states, dtype=np.float32)
		actions = np.array(actions, dtype=np.float32).reshape(-1, 1)
		returns = np.array(returns, dtype=np.float32)
		advantages = np.array(advantages, dtype=np.float32).reshape(-1, 1)

		for _ in range(self.params.num_updates):
			loss, kl_divergence = self._inner_update(states, actions, returns, advantages, old_policy)

			if kl_divergence.numpy() > 4 * self.params.kl_target:
				break

		''' p.4 in https://arxiv.org/pdf/1707.06347.pdf '''
		if kl_divergence.numpy() < self.params.kl_target / 1.5:
			self.beta /= 2
		elif kl_divergence.numpy() > self.params.kl_target * 1.5:
			self.beta *= 2
		self.beta = np.clip(self.beta, self.beta_min, self.beta_max)

		return loss

	@tf.contrib.eager.defun(autograph=False)
	def _inner_update(self, states, actions, returns, advantages, old_policy):
		self.index_timestep = tf.train.get_global_step()
		# Update Critic
		with tf.GradientTape() as tape:
			state_values = self.critic(states)

			# Compute critic loss
			L2 = tf.add_n(self.critic.losses) * self.params.L2_reg
			critic_loss = tf.losses.mean_squared_error(returns, tf.reshape(state_values, [-1])) + L2

		critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
		self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

		# Update Actor
		with tf.GradientTape() as tape:
			mean, std = self.actor(states)
			new_policy = tf.contrib.distributions.Normal(mean, std)
			kl_divergence = tf.contrib.distributions.kl_divergence(new_policy, old_policy)
			actor_loss  = -tf.math.reduce_mean(advantages*tf.math.exp(new_policy.log_prob(actions) - old_policy.log_prob(actions)))
			actor_loss +=  tf.math.reduce_mean(self.beta * kl_divergence)
			actor_loss +=  tf.math.reduce_mean(self.ksi * tf.math.square(tf.math.maximum(0.0, kl_divergence - 2 * self.params.kl_target)))

		# get gradients
		actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

		# apply processed gradients to the network
		self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

		return tf.math.reduce_sum(critic_loss + actor_loss), tf.math.reduce_sum(kl_divergence)


class TRPO_debug:
	def __init__(self, actor, critic, num_action, params):
		self.params = params
		self.num_action = num_action
		self.beta = 1
		self.beta_min = 1./20.
		self.beta_max = 20
		self.ksi = 10
		self.index_timestep = 0
		self.actor = actor(num_action)
		self.critic = critic(1)
		self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)  # used as in paper
		self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=1.5*1e-3)  # used as in paper

	#  TODO: implement the checkpoints for model

	def predict(self, state):
		state = np.expand_dims(state, axis=0).astype(np.float32)
		action = self._select_action(tf.constant(state))
		return action.numpy()[0]

	# @tf.contrib.eager.defun(autograph=False)
	def _select_action(self, state):
		mean, std = self.actor(state)
		return tf.squeeze(mean + std * tf.random_normal(shape=tf.shape(mean)))

	def update(self, states, actions, returns, advantages, dones, old_policy):
		states = np.array(states, dtype=np.float32)
		actions = np.array(actions, dtype=np.float32).reshape(-1, 1)
		returns = np.array(returns, dtype=np.float32)
		advantages = np.array(advantages, dtype=np.float32).reshape(-1, 1)
		dones = np.array(dones, dtype=np.float32)

		for _ in range(self.params.num_updates):
			loss, kl_divergence = self._inner_update(states, actions, returns, advantages, dones, old_policy)

			if kl_divergence.numpy() > 4 * self.params.kl_target:
				break

		''' p.4 in https://arxiv.org/pdf/1707.06347.pdf '''
		if kl_divergence.numpy() < self.params.kl_target / 1.5:
			self.beta /= 2
		elif kl_divergence.numpy() > self.params.kl_target * 1.5:
			self.beta *= 2
		self.beta = np.clip(self.beta, self.beta_min, self.beta_max)

		return loss

	# @tf.contrib.eager.defun(autograph=False)
	def _inner_update(self, states, actions, returns, advantages, dones, old_policy):
		self.index_timestep = tf.train.get_global_step()
		# Update Critic
		with tf.GradientTape() as tape:
			state_values = self.critic(states)

			# Compute critic loss
			L2 = tf.add_n(self.critic.losses) * self.params.L2_reg
			critic_loss = tf.losses.mean_squared_error(returns, tf.reshape(state_values, [-1])) + L2

		critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
		self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

		# Update Actor
		with tf.GradientTape() as tape:
			mean, std = self.actor(states)
			new_policy = tf.contrib.distributions.Normal(mean, std)
			kl_divergence = tf.contrib.distributions.kl_divergence(new_policy, old_policy)
			actor_loss  = -tf.math.reduce_mean(advantages*tf.math.exp(new_policy.log_prob(actions) - old_policy.log_prob(actions)))
			actor_loss +=  tf.math.reduce_mean(self.beta * kl_divergence)
			actor_loss +=  tf.math.reduce_mean(self.ksi * tf.math.square(tf.math.maximum(0.0, kl_divergence - 2 * self.params.kl_target)))

		# get gradients
		actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

		# apply processed gradients to the network
		self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

		tf.contrib.summary.histogram("Mean", mean, step=self.index_timestep)
		tf.contrib.summary.histogram("Std", std, step=self.index_timestep)
		tf.contrib.summary.histogram("KL Divergence", kl_divergence, step=self.index_timestep)
		tf.contrib.summary.scalar("Returns", returns, step=self.index_timestep)
		tf.contrib.summary.scalar("State values", state_values, step=self.index_timestep)
		tf.contrib.summary.scalar("Critic Loss", critic_loss, step=self.index_timestep)
		tf.contrib.summary.scalar("Actor Loss", actor_loss, step=self.index_timestep)

		return tf.math.reduce_sum(critic_loss + actor_loss), tf.math.reduce_sum(kl_divergence)
