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

# Do inference.
batch_size = 32

def create_model():
	num_inducing_points = 40
	loss = lambda y, rv_y: rv_y.variational_loss(y)

	model = tf.keras.Sequential([
		tf.keras.layers.InputLayer(input_shape=[2], dtype=x.dtype),
		tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False),
		tfp.layers.VariationalGaussianProcess(
			num_inducing_points=num_inducing_points,
			kernel_provider=RBFKernelFn(dtype=x.dtype),
			event_shape=[1],
			inducing_index_points_initializer=tf.constant_initializer(
				np.linspace(*x_range, num=num_inducing_points,
							dtype=x.dtype)[..., np.newaxis]),
			unconstrained_observation_noise_variance_initializer=(
				tf.constant_initializer(np.array(0.54).astype(x.dtype))),
		),
	])
	model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.01), loss=loss)
	return model

model = create_model()
# model.fit(x, y, batch_size=batch_size, epochs=100, verbose=False)

for i in range(100):
	model.fit(x[i][np.newaxis, ...], y[i,0][np.newaxis, ...])

# Profit.
yhat = model(x_tst)
assert isinstance(yhat, tfd.Distribution)

y, x, _ = load_dataset()


from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.figure(figsize=[6, 1.5])  # inches
plt.plot(x, y, 'b.', label='observed');

num_samples = 7
for i in range(num_samples):
	sample_ = yhat.sample().numpy()
	plt.plot(x_tst,
			 sample_[..., 0].T,
			 'r',
			 linewidth=0.9,
			 label='ensemble means' if i == 0 else None);

plt.ylim(-0., 17);
plt.yticks(np.linspace(0, 15, 4)[1:]);
plt.xticks(np.linspace(*x_range, num=9));

ax = plt.gca();
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['left'].set_smart_bounds(True)
# ax.spines['bottom'].set_smart_bounds(True)
plt.legend(loc='center left', fancybox=True, framealpha=0., bbox_to_anchor=(1.05, 0.5))
plt.show()
