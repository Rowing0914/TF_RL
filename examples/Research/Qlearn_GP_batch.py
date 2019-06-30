import gym
import numpy as np
import collections
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from tf_rl.common.utils import AnnealingSchedule, eager_setup
from tf_rl.common.wrappers import DiscretisedEnv

eager_setup()

tfd = tfp.distributions

class RBFKernelFn(tf.keras.layers.Layer):
	# https://medium.com/tensorflow/regression-with-probabilistic-layers-in-tensorflow-probability-e46ff5d37baf
	def __init__(self, **kwargs):
		super(RBFKernelFn, self).__init__(**kwargs)
		dtype = kwargs.get('dtype', None)

		self._amplitude = self.add_variable(
			initializer=tf.constant_initializer(0),
			dtype=dtype,
			name='amplitude')

		self._length_scale = self.add_variable(
			initializer=tf.constant_initializer(0),
			dtype=dtype,
			name='length_scale')

	def call(self, x):
		# Never called -- this is just a layer so it can hold variables
		# in a way Keras understands.
		return x

	@property
	def kernel(self):
		return tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(
			amplitude=tf.nn.softplus(0.1 * self._amplitude),
			length_scale=tf.nn.softplus(5. * self._length_scale)
		)


class Model(tf.keras.Model):
	""" Variational Gaussian Process Network """

	def __init__(self, num_inducing_points, kernel, dtype=np.float64):
		# def __init__(self, _max, _min, num_inducing_points, kernel, dtype=np.float64): # if we need inducing_index then, open this!
		super(Model, self).__init__()
		self.dense = tf.keras.layers.Dense(1, activation='linear')
		self.gp = tfp.layers.VariationalGaussianProcess(
			num_inducing_points=num_inducing_points,
			kernel_provider=kernel(dtype=dtype),
			event_shape=[1],
			# inducing_index_points_initializer=tf.constant_initializer(
			# 	np.linspace(_min, _max, num=num_inducing_points, dtype=dtype)[..., np.newaxis]),
			unconstrained_observation_noise_variance_initializer=(
				tf.constant_initializer(np.array(0.54).astype(dtype))),
		)

	# @tf.contrib.eager.defun(autograph=False)
	def call(self, inputs):
		x = self.dense(inputs)
		return self.gp(x)


class Q_Agent:
	def __init__(self, env):
		self.env = env
		# (1, 1, 6, 12, 2) => 144 dim vector after being serialised
		self.Q = np.zeros(self.env.buckets + (env.action_space.n,))
		self.gamma = 0.995

	def choose_action(self, state, epsilon):
		return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

	def update(self, state, action, reward, next_state, alpha):
		self.Q[state][action] += alpha * (reward + 1. * np.max(self.Q[next_state]) - self.Q[state][action])

	def test(self):
		"""
		Test the agent with a visual aid!
		"""

		scores = list()
		for ep in range(10):
			current_state = self.env.reset()
			done = False
			score = 0
			while not done:
				action = self.choose_action(current_state, 0)
				obs, reward, done, _ = self.env.step(action)
				current_state = obs
				score += reward
			scores.append(score)
			print("Ep: {}, Score: {}".format(ep, score))
		scores = np.array(scores)
		print("Eval => Std: {}, Mean: {}".format(np.std(scores), np.mean(scores)))


if __name__ == '__main__':
	# DiscretisedEnv
	env = DiscretisedEnv(gym.make('CartPole-v0'))

	# hyperparameters
	num_episodes = 500
	goal_duration = 190
	decay_steps = 5000
	durations = collections.deque(maxlen=10)
	Epsilon = AnnealingSchedule(start=1.0, end=0.01, decay_steps=decay_steps)
	Alpha = AnnealingSchedule(start=1.0, end=0.01, decay_steps=decay_steps)
	agent = Q_Agent(env)


	# gp_model = Model(num_inducing_points=40, kernel=RBFKernelFn)
	# optimiser = tf.train.AdamOptimizer()
	# # https://medium.com/tensorflow/regression-with-probabilistic-layers-in-tensorflow-probability-e46ff5d37baf
	# loss_func = lambda y, rv_y: rv_y.variational_loss(y)  # temp loss func
	#
	#
	# def update(model, x, y):
	# 	""" Temp function to update the weights of GP net """
	# 	with tf.GradientTape() as tape:
	# 		pred = model(x)
	# 		print(pred)
	# 		loss = loss_func(y, pred)
	# 	grads = tape.gradient(loss, model.trainable_weights)  # get gradients
	# 	optimiser.apply_gradients(zip(grads, model.trainable_weights))  # apply gradients to the network
	# 	return model, loss

	def create_model():
		dtype = np.float64
		num_inducing_points = 40
		loss = lambda y, rv_y: rv_y.variational_loss(y)

		model = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=[144], dtype=dtype),
			tf.keras.layers.Dense(72, kernel_initializer='ones', use_bias=False),
			tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False),
			tfp.layers.VariationalGaussianProcess(
				num_inducing_points=num_inducing_points,
				kernel_provider=RBFKernelFn(dtype=dtype),
				event_shape=[1],
				# inducing_index_points_initializer=tf.constant_initializer(
				# 	np.linspace(*x_range, num=num_inducing_points,
				# 				dtype=x.dtype)[..., np.newaxis]),
				unconstrained_observation_noise_variance_initializer=(
					tf.constant_initializer(np.array(0.54).astype(dtype))),
			),
		])
		model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.01), loss=loss)
		return model


	gp_model = create_model()
	batch_size = 50
	num_epochs = 100
	num_sample = 100  # number of sampling

	policies, scores = list(), list()
	_max, _min, _means = list(), list(), list()

	global_timestep = tf.train.get_or_create_global_step()
	# === Data Collection Part ===
	for episode in range(num_episodes):
		current_state = env.reset()

		done = False
		duration = 0

		# one episode of q learning
		while not done:
			duration += 1
			global_timestep.assign_add(1)
			action = agent.choose_action(current_state, Epsilon.get_value())
			new_state, reward, done, _ = env.step(action)
			agent.update(current_state, action, reward, new_state, Alpha.get_value())
			current_state = new_state

		# == After 1 episode ===
		policies.append(agent.Q.flatten())
		scores.append(duration)
		durations.append(duration)

		if episode > batch_size:
			history = gp_model.fit(np.array(policies), np.array(scores), batch_size=batch_size, epochs=num_epochs,
								   verbose=False)
			sample_ = gp_model(agent.Q.flatten()[np.newaxis, ...]).sample(num_sample).numpy()
			print("Ep: {} | Return: {} | Mean Est Return: {:.2f} | Mean Loss: {:.3f}".format(
				episode, duration, sample_.mean(), np.mean(history.history["loss"]))
			)
			_max.append(sample_.max())
			_min.append(sample_.min())
			_means.append(sample_.mean())

		# check if our policy is good
		# if np.mean(durations) >= goal_duration and episode >= 100:
		# 	print('Ran {} episodes. Solved after {} trials'.format(episode, episode - 100))
		# 	agent.test()
		# 	env.close()
		# 	break

	# === Visualisation Part ===
	_min, _max, _means = np.array(_min), np.array(_max), np.array(_means)
	plt.title("Starting at {} Ep".format(batch_size))
	plt.plot(scores, label="Return")
	plt.plot(_means, label="Est Return")
	plt.fill_between(np.arange(len(_min)), _min, _max, facecolor='blue', alpha=0.5)
	plt.legend()
	plt.show()