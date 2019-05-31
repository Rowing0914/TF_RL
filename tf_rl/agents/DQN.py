import tensorflow as tf
from tf_rl.common.utils import ClipIfNotNone, AnnealingSchedule, create_checkpoint
from tf_rl.agents.core import Agent_atari, Agent_cartpole


class DQN(Agent_atari):
	"""
	A complete DQN model for training of Atari or cartpole-pixel
	and don't use this for debugging purpose.
	"""

	def __init__(self, model, optimizer, loss_fn, grad_clip_fn, num_action, gamma, model_dir):
		self.num_action = num_action
		self.gamma = gamma
		self.grad_clip_fn = grad_clip_fn
		self.loss_fn = loss_fn
		self.eval_flg = False
		self.index_timestep = 0
		self.main_model = model(num_action)
		self.target_model = model(num_action)
		self.optimizer = optimizer
		self.manager = create_checkpoint(model=self.main_model,
										 optimizer=self.optimizer,
										 model_dir=model_dir)

	@tf.contrib.eager.defun(autograph=False)
	def _select_action(self, state):
		return self.main_model(state)

	@tf.contrib.eager.defun(autograph=False)
	def inner_update(self, states, actions, rewards, next_states, dones):
		# get the current global-timestep
		self.index_timestep = tf.train.get_global_step()

		# We divide the grayscale pixel values by 255 here rather than storing
		# normalized values because uint8s are 4x cheaper to store than float32s.
		states, next_states = states/255., next_states/255.

		# ===== make sure to fit all process to compute gradients within this Tape context!! =====
		with tf.GradientTape() as tape:
			next_Q = self.target_model(next_states)
			q_values = self.main_model(states)
			Y = rewards + self.gamma * tf.math.reduce_max(next_Q, axis=-1) * (1. - dones)
			Y = tf.stop_gradient(Y)

			# get the q-values which is associated with actually taken actions in a game
			actions_one_hot = tf.one_hot(actions, self.num_action, 1.0, 0.0)
			chosen_q = tf.math.reduce_sum(tf.math.multiply(actions_one_hot, q_values), reduction_indices=1)
			batch_loss = self.loss_fn(Y, chosen_q)
			loss = tf.math.reduce_mean(batch_loss)

		# get gradients
		grads = tape.gradient(loss, self.main_model.trainable_weights)

		# clip gradients
		grads = self.grad_clip_fn(grads)

		# apply processed gradients to the network
		self.optimizer.apply_gradients(zip(grads, self.main_model.trainable_weights))
		return loss, batch_loss


class DQN_cartpole(Agent_cartpole):
	"""
	A complete DQN model for training of cartpole not for debugging purpose
	"""

	def __init__(self, model, optimizer, loss_fn, grad_clip_fn, num_action, gamma, model_dir):
		self.num_action = num_action
		self.gamma = gamma
		self.grad_clip_fn = grad_clip_fn
		self.loss_fn = loss_fn
		self.eval_flg = False
		self.index_timestep = 0
		self.main_model = model(num_action)
		self.target_model = model(num_action)
		self.optimizer = optimizer
		self.manager = create_checkpoint(model=self.main_model,
										 optimizer=self.optimizer,
										 model_dir=model_dir)

	@tf.contrib.eager.defun(autograph=False)
	def _select_action(self, state):
		return self.main_model(state)


	@tf.contrib.eager.defun(autograph=False)
	def inner_update(self, states, actions, rewards, next_states, dones):
		# get the current global-timestep
		self.index_timestep = tf.train.get_global_step()

		# ===== make sure to fit all process to compute gradients within this Tape context!! =====
		with tf.GradientTape() as tape:
			next_Q = self.target_model(next_states)
			q_values = self.main_model(states)
			Y = rewards + self.gamma * tf.math.reduce_max(next_Q, axis=-1) * (1. - dones)
			Y = tf.stop_gradient(Y)

			# get the q-values which is associated with actually taken actions in a game
			actions_one_hot = tf.one_hot(actions, self.num_action, 1.0, 0.0)
			chosen_q = tf.math.reduce_sum(tf.math.multiply(actions_one_hot, q_values), reduction_indices=1)
			batch_loss = self.loss_fn(Y, chosen_q)
			loss = tf.math.reduce_mean(batch_loss)

		# get gradients
		grads = tape.gradient(loss, self.main_model.trainable_weights)

		# clip gradients
		grads = self.grad_clip_fn(grads)

		# apply processed gradients to the network
		self.optimizer.apply_gradients(zip(grads, self.main_model.trainable_weights))

		return loss, batch_loss



class DQN_debug:
	"""
	DQN agent for debugging purpose, which means you can pass any arguments in `params` so that we can easily experiment
	with the agent without having a carefully planned design
	"""

	def __init__(self, main_model, target_model, num_action, params):
		self.num_action = num_action
		self.params = params
		self.eval_flg = False
		self.index_timestep = 0
		self.main_model = main_model(num_action)
		self.target_model = target_model(num_action)
		# self.lr = AnnealingSchedule(start=1e-3, end=1e-5, decay_steps=params.decay_steps, decay_type="linear") # learning rate decay!!
		# self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr.get_value())

		self.lr = AnnealingSchedule(start=0.0025, end=0.00025, decay_steps=params.decay_steps,
									decay_type="linear")  # learning rate decay!!
		self.optimizer = tf.train.RMSPropOptimizer(self.lr.get_value(), 0.99, 0.0, 1e-6)
		self.manager = create_checkpoint(model=self.main_model,
										 optimizer=self.optimizer,
										 model_dir=self.params.model_dir)


	def predict(self, state):
		state = tf.math.divide(state, 255.)
		return self.main_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32)).numpy()[0]

	def update(self, states, actions, rewards, next_states, dones):
		# get the current global-timestep
		self.index_timestep = tf.train.get_global_step()

		states, next_states = tf.math.divide(states, 255.), tf.math.divide(next_states, 255.)

		# ===== make sure to fit all process to compute gradients within this Tape context!! =====
		with tf.GradientTape() as tape:
			next_Q = self.target_model(tf.convert_to_tensor(next_states, dtype=tf.float32))
			# Y = rewards + self.params.gamma * tf.math.reduce_max(next_Q, axis=-1) * (1. - tf.cast(dones, tf.float32))
			Y = tf.math.multiply(self.params.gamma, tf.math.reduce_max(next_Q, axis=-1))
			Y = tf.math.multiply(Y, (1. - tf.cast(dones, tf.float32)))
			Y = tf.math.add(rewards, Y)
			Y = tf.stop_gradient(Y)

			# calculate Q(s,a)
			q_values = self.main_model(tf.convert_to_tensor(states, dtype=tf.float32))

			# at this point, instead of getting only q-values associated wit taken actions
			# we retain all values except that we update q-values associated wit taken actions by "Y"
			# Shape of Q-values matrix: (32, num_actions)
			target_values = tf.one_hot(actions, self.num_action, 1.0, 0.0) * tf.transpose(
				tf.stack([Y] * self.num_action)) + tf.one_hot(actions, self.num_action, 0.0, 1.0) * q_values
			assert tf.math.equal(target_values,
								 q_values).numpy().all() == False, "Your target values are not updated correctly"

			if self.params.loss_fn == "huber":
				# use huber loss
				batch_loss = tf.losses.huber_loss(target_values, q_values, reduction=tf.losses.Reduction.NONE)
				loss = tf.math.reduce_mean(batch_loss)
			elif self.params.loss_fn == "mse":
				# use MSE
				batch_loss = tf.math.squared_difference(target_values, q_values)
				loss = tf.math.reduce_mean(batch_loss)
			else:
				assert False

		# get gradients
		grads = tape.gradient(batch_loss, self.main_model.trainable_weights)

		# clip gradients
		if self.params.grad_clip_flg == "by_value":
			grads = [ClipIfNotNone(grad, -1., 1.) for grad in grads]
		elif self.params.grad_clip_flg == "norm":
			grads, _ = tf.clip_by_global_norm(grads, 5.0)
		elif self.params.grad_clip_flg == "None":
			pass

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
		tf.contrib.summary.scalar("learning_rate", self.lr.get_value(), step=self.index_timestep)

		return loss, batch_loss
