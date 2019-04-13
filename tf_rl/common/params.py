"""
Parameters for agents

"""

class Parameters:
	def __init__(self, mode=None):
		assert mode != None
		print("Loading Params for {} Environment".format(mode))
		if mode == "Atari":
			self.state_reshape = (1, 84, 84, 1)
			self.loss_fn = "MSE"          # loss func: MSE or huber_loss
			self.policy_fn = "Eps"         # Epsilon Greedy(Eps) or Boltzmann(Boltzmann)
			self.grad_clip_flg = "norm"      # clipping by value(by_value) or global norm(norm)
			self.num_frames = 30000
			self.memory_size = 20000
			self.learning_start = 100
			self.sync_freq = 100                 # frequency of updating a target model
			self.batch_size = 32
			self.gamma = 0.99                    # gamma > 1.0 or negative => does not converge!!
			self.update_hard_or_soft = "hard"    # soft or hard update
			self.soft_update_tau = 1e-2          # seems 1e-2 is the optimal ratio for tau!!
			self.epsilon_start = 1.0
			self.epsilon_end = 0.1
			self.decay_steps = 1000               # a period for annealing a value
			self.decay_type = "curved"           # how to anneal the value: linear or curved
			self.prioritized_replay_alpha = 0.6
			self.prioritized_replay_beta_start = 0.4
			self.prioritized_replay_beta_end = 1.0
			self.prioritized_replay_noise = 1e-6
			self.tau = 1.
			self.clip = (-500., 500.)
			self.test_episodes = 10
		elif mode == "CartPole":
			self.state_reshape = (1, 4)
			self.loss_fn = "MSE"          # loss func: MSE or huber_loss
			self.policy_fn = "Eps"         # Epsilon Greedy(Eps) or Boltzmann(Boltzmann)
			self.grad_clip_flg = "norm"      # clipping by value(by_value) or global norm(norm)
			self.num_frames = 30000
			self.memory_size = 20000
			self.learning_start = 100
			self.sync_freq = 100                 # frequency of updating a target model
			self.batch_size = 32
			self.gamma = 0.99                    # gamma > 1.0 or negative => does not converge!!
			self.update_hard_or_soft = "hard"    # soft or hard update
			self.soft_update_tau = 1e-2          # seems 1e-2 is the optimal ratio for tau!!
			self.epsilon_start = 1.0
			self.epsilon_end = 0.1
			self.decay_steps = 1000               # a period for annealing a value
			self.decay_type = "curved"           # how to anneal the value: linear or curved
			self.prioritized_replay_alpha = 0.6
			self.prioritized_replay_beta_start = 0.4
			self.prioritized_replay_beta_end = 1.0
			self.prioritized_replay_noise = 1e-6
			self.tau = 1.
			self.clip = (-500., 500.)
			self.test_episodes = 10