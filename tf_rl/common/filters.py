"""
This is the implementations of Gaussian filters
As of 26/4/2019, I have just finished implementing Particle filter.
for test it, pls check the test/Particle_Filter_test.py

"""

import numpy as np
from scipy.stats import norm


class Particle_Filter:
	def __init__(self, N=10, type="uniform"):
		self.N = N  # number of particles
		self.weights = np.ones(self.N)/self.N  # weights of particles

		# initialise particles accordingly
		if type == "uniform":
			self.particles = self.create_uniform_particles()
		elif type == "gaussian":
			self.particles = self.create_gaussian_particles()

	def create_uniform_particles(self, pos=(-0.6, -0.4), vel=(-0.07, 0.07)):
		"""
		Initialise the particles according to Uniform distribution

		:param pos:
		:param vel:
		:return:
		"""
		particles = np.empty((self.N, 2))
		particles[:, 0] = np.random.uniform(pos[0], pos[1], size=self.N)
		particles[:, 1] = np.random.uniform(vel[0], vel[1], size=self.N)
		return particles

	def create_gaussian_particles(self, mean=(-0.6, -0.07), std=(1, 1)):
		"""
		Initialise the particles according to Gaussian distribution

		:param mean:
		:param std:
		:return:
		"""
		particles = np.empty((self.N, 2))
		particles[:, 0] = np.random.normal(mean[0], std[1], size=self.N)
		particles[:, 1] = np.random.normal(mean[0], std[1], size=self.N)
		return particles

	def predict(self, env, action):
		"""
		Move particles according to the action on the given Environment

		:param env:
		:param action:
		:return:
		"""
		original_state = env.get_state()
		for i in range(self.N):
			env.set_state(self.particles[i, :])
			next_state, _, _, _ = env.step(action)
			self.particles[i, :] = next_state
			env.set_state(original_state)

	def update(self, q_values):
		"""
		=== Sequential Importance Sampling ===
		Given a measurement(Q-values), we update the prediction of each particle to assign high probability to a particle
		that is located close to the model's prediction(Q-values) and lower probability to the one that is located far away.

		:return:
		"""
		sigma = 1  # temporarily initialisation
		distance = np.linalg.norm(self.particles - q_values, axis=1)
		self.weights *= norm(distance, sigma).pdf(q_values)
		self.weights += 1.e-300  # avoid round-off to zero
		self.weights /= np.sum(self.weights)  # normalize

	def estimate(self):
		"""
		returns mean and variance of the weighted particles

		"""

		mean = np.mean(np.average(self.particles, weights=self.weights, axis=0))
		var = np.mean(np.average((self.particles - mean) ** 2, weights=self.weights, axis=0))
		return mean, var

	def simple_resample(self):
		cumulative_sum = np.cumsum(self.weights)
		cumulative_sum[-1] = 1.  # avoid round-off error
		indexes = np.searchsorted(cumulative_sum, np.random.rand(self.N))

		# resample according to indexes
		self.particles[:] = self.particles[indexes]
		self.weights.fill(1.0 / self.N)