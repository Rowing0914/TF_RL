import numpy as np


class Policy:
	"""
	boilterplate for policy class
	"""
	def __init__(self):
		pass

	def select_action(self):
		raise NotImplementedError()


class EpsilonGreedyPolicy(Policy):
	"""
	Epsilon Greedy Policy
	"""
	def __init__(self, Epsilon_fn):
		self.Epsilon = Epsilon_fn
		self.timestep = 0

	def select_action(self, sess, agent, state):
		if np.random.uniform() < self.Epsilon.get_value(self.timestep):
			action = np.random.randint(agent.num_action)
		else:
			q_value = sess.run(agent.pred, feed_dict={agent.state: state})[0]
			action = np.argmax(q_value)
		self.timestep += 1
		return action


class BoltzmannQPolicy(Policy):
	"""
	Boltzmann Q Policy

	Original implementation: https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py

	"""
	def __init__(self, tau=1., clip=(-500., 500.)):
		self.tau = tau
		self.clip = clip

	def select_action(self, sess, agent, state):
		"""Return the selected action
		# Arguments
			q_values (np.ndarray): List of the estimations of Q for each action
		# Returns
			Selection action
		"""

		q_values = sess.run(agent.pred, feed_dict={agent.state: state})[0]
		nb_actions = q_values.shape[0]

		exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
		probs = exp_values / np.sum(exp_values)
		action = np.random.choice(range(nb_actions), p=probs)
		return action