import numpy as np

class Policy:
	"""
	boilterplate for policy class
	"""
	def __init__(self, Epsilon_fn):
		pass

	def select_action(self, sess, agent, state):
		raise NotImplementedError()


class EpsilonGreedyPolicy(Policy):
	"""
	Epsilon Greedy Policy
	"""
	def __init__(self, Epsilon_fn):
		self.Epsilon = Epsilon_fn
		self.index_episode = 0
		self.index_frame = 0

	def select_action(self, sess, agent, state):
		if np.random.uniform() < self.Epsilon.get_value(self.index_frame):
			action = np.random.randint(agent.num_action)
		else:
			q_values = sess.run(agent.pred, feed_dict={agent.state: state})[0]
			action = np.argmax(q_values)
		return action

	def current_epsilon(self):
		return self.Epsilon.get_value(self.index_episode)


class EpsilonGreedyPolicy_eager:
	"""
	Epsilon Greedy Policy for eager execution
	"""
	def __init__(self, Epsilon_fn):
		self.Epsilon = Epsilon_fn

	def select_action(self, agent, state):
		if np.random.uniform() < self.Epsilon.get_value():
			action = np.random.randint(agent.num_action)
		else:
			q_values = agent.predict(state)
			action = np.argmax(q_values)
		return action

	def current_epsilon(self):
		return self.Epsilon.get_value()


class BoltzmannQPolicy(Policy):
	"""
	Boltzmann Q Policy

	Original implementation: https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py

	"""
	def __init__(self, tau=1., clip=(-500., 500.)):
		self.tau = tau
		self.clip = clip
		self.index_episode = 0

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

	def current_epsilon(self):
		return 0


class BoltzmannQPolicy_eager:
	"""
	Boltzmann Q Policy

	Original implementation: https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py

	"""
	def __init__(self, tau=1., clip=(-500., 500.)):
		self.tau = tau
		self.clip = clip
		self.index_episode = 0

	def select_action(self, agent, state):
		"""Return the selected action
		# Arguments
			q_values (np.ndarray): List of the estimations of Q for each action
		# Returns
			Selection action
		"""

		q_values = agent.predict(state)
		nb_actions = q_values.shape[0]

		exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
		probs = exp_values / np.sum(exp_values)
		action = np.random.choice(range(nb_actions), p=probs)
		return action

	def current_epsilon(self):
		return 0


class TestPolicy:
	"""
	Policy to be used at the test phase
	"""
	def __init__(self):
		pass

	def select_action(self, agent, state):
		action = np.argmax(agent.predict(state))
		return action

	def current_epsilon(self):
		return 0