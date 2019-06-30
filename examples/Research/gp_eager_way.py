import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tf.enable_eager_execution()

tfd = tfp.distributions

w0 = 0.125
b0 = 5.
x_range = [-20, 60]


def load_dataset(n=150, n_tst=150):
	np.random.seed(43)

	def s(x):
		g = (x - x_range[0]) / (x_range[1] - x_range[0])
		return 3 * (0.25 + g ** 2.)

	x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
	x = x[..., np.newaxis]
	x = np.concatenate([x, x], axis=-1)
	eps = np.random.randn(n, 2) * s(x)
	y = (w0 * x * (1. + np.sin(x)) + b0) + eps
	x_tst = np.linspace(*x_range, num=n_tst).astype(np.float32)
	x_tst = x_tst[..., np.newaxis]
	x_tst = np.concatenate([x_tst, x_tst], axis=-1)
	return y, x, x_tst

y, x, x_tst = load_dataset()

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
	def __init__(self, num_inducing_points, kernel):
		super(Model, self).__init__()
		self.dense = tf.keras.layers.Dense(1, activation='linear')
		self.gp = tfp.layers.VariationalGaussianProcess(
			num_inducing_points=num_inducing_points,
			kernel_provider=kernel(dtype=x.dtype),
			event_shape=[1],
			inducing_index_points_initializer=tf.constant_initializer(
				np.linspace(*x_range, num=num_inducing_points,
							dtype=x.dtype)[..., np.newaxis]),
			unconstrained_observation_noise_variance_initializer=(
				tf.constant_initializer(np.array(0.54).astype(x.dtype))),
		)

	# @tf.contrib.eager.defun(autograph=False)
	def call(self, inputs):
		x = self.dense(inputs)
		return self.gp(x)

num_inducing_points = 40
model = Model(num_inducing_points, kernel=RBFKernelFn)
optimiser = tf.train.AdamOptimizer()

# Do inference.
loss = lambda y, rv_y: rv_y.variational_loss(y)

for i in range(150):
	with tf.GradientTape() as tape:
		pred = model(x[i][np.newaxis, ...])
		print(pred)
		pred = model(x)
		print(pred)
		loss_ = loss(y[i], pred)

	# get gradients
	grads = tape.gradient(loss_, model.trainable_weights)

	# apply processed gradients to the network
	optimiser.apply_gradients(zip(grads, model.trainable_weights))
	print(loss_)