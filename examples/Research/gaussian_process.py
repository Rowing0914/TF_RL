import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
psd_kernels = tfp.positive_semidefinite_kernels

f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
observation_index_points = np.random.uniform(-1., 1., 50)[..., np.newaxis]
observations = f(observation_index_points) + np.random.normal(0., .05, 50)

amplitude = tf.math.exp(tf.Variable(np.float64(0)), name="amplitude")
length_scale = tf.math.exp(tf.Variable(np.float64(0)), name="length_scale")
kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)

observation_noise_variance = tf.math.exp(tf.Variable(np.float64(-5)), name="observation_noise_variance")

gp = tfd.GaussianProcess(
	kernel=kernel,
	index_points=observation_index_points,
	observation_noise_variance=observation_noise_variance
)

neg_log_likelihood = -gp.log_prob(observations)

optimiser = tf.train.AdamOptimizer(learning_rate=.05, beta1=.5, beta2=.99)
optimise = optimiser.minimize(neg_log_likelihood)

index_points = np.linspace(-1., 1., 100)[..., np.newaxis]

gprm = tfd.GaussianProcessRegressionModel(
	kernel=kernel,
	index_points=index_points,
	observation_index_points=observation_index_points,
	observations=observations,
	observation_noise_variance=observation_noise_variance
)

samples = gprm.sample(10)
print(samples)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(1000):
		_, neg_log_likelihood_ = sess.run([optimise, neg_log_likelihood])
		print(sess.run([amplitude, length_scale, observation_noise_variance]))
		if i % 100 == 0:
			print("Step {}: NLL: {}".format(i, neg_log_likelihood_))
	print("Final NLL: {}".format(neg_log_likelihood_))
	samples_ = sess.run(samples)

	import matplotlib.pyplot as plt
	plt.scatter(np.squeeze(observation_index_points), observations)
	plt.plot(np.stack([index_points[:, 0]]*10).T, samples_.T, c="r", alpha=.2)
	plt.show()
