"""
Parameters for agents

"""

class Parameters:
	def __init__(self, mode=None):
		assert mode != None
		print("Loading Params for {} Environment".format(mode))
		if mode == "Atari":
			self.state_reshape = (1, 84, 84, 1)
			self.num_frames = 1000000
			self.memory_size = 10000
			self.learning_start = 10000
			self.sync_freq = 1000
			self.batch_size = 32
			self.gamma = 0.99
			self.update_hard_or_soft = "soft"
			self.soft_update_tau = 1e-2
			self.epsilon_start = 1.0
			self.epsilon_end = 0.01
			self.decay_steps = 1000
			self.prioritized_replay_alpha = 0.6
			self.prioritized_replay_beta_start = 0.4
			self.prioritized_replay_beta_end = 1.0
			self.prioritized_replay_noise = 1e-6
		elif mode == "CartPole":
			self.state_reshape = (1, 4)
			self.num_frames = 10000
			self.memory_size = 5000              # does not affect the performance
			self.learning_start = 100            # does not affect the performance
			self.sync_freq = 100                 # as you increase, a loss gets to not converge
			self.batch_size = 32
			self.gamma = 0.99                    # gamma > 1.0 or negative => does not converge!!
			self.update_hard_or_soft = "soft"    # don't use hard update method!! horrible,, cause a surge of loss
			self.soft_update_tau = 1e-2          # seems 1e-2 is the optimal ratio for tau!!
			self.epsilon_start = 1.0
			self.epsilon_end = 0.1
			self.decay_steps = 500               # this defines the frequency of the interatcion of models
			self.prioritized_replay_alpha = 0.6
			self.prioritized_replay_beta_start = 0.4
			self.prioritized_replay_beta_end = 1.0
			self.prioritized_replay_noise = 1e-6