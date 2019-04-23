"""
Parameters for agents

===== ALL PARAMS =====
state_reshape: state shape for pre-processing observations
loss_fn: types of loss function => MSE or huber_loss
policy_fn: types of policy function => Epsilon Greedy(Eps) or Boltzmann(Boltzmann)
grad_clip_flg: types of a cilpping method of gradients => by value(by_value) or global norm(norm)
num_frames: total frame in a training
num_episodes: total episode in a training
memory_size: memory size in a training => this used for Experience Replay Memory or Prioritised Experience Replay Memory
learning_start: frame number which specifies when to start updating the agent
sync_freq: frequency of updating a target model <= maybe we don't need this, bcuz we are using stochastic update method!
batch_size: batch size of each iteration of update
gamma: discount factor => gamma > 1.0 or negative => does not converge!!
update_hard_or_soft: types of synchronisation method of target and main models => soft or hard update
soft_update_tau: in soft-update tau defines the ratio of main model remains and it seems 1e-2 is the optimal!
epsilon_start: initial value of epsilon
epsilon_end: final value of epsilon
decay_steps: a period for annealing a value(epsilon or beta)
decay_type: types of annealing method => linear or curved
prioritized_replay_alpha = 0.6
prioritized_replay_beta_start = 0.4
prioritized_replay_beta_end = 1.0
prioritized_replay_noise = 1e-6
tau = 1.
clip = (-500., 500.)
test_episodes = 10
goal = 20
reward_buffer_ep = 2

"""

class Parameters:
    def __init__(self, algo=None, mode=None):
        assert algo != None, "Give me a name of the learning algorithm"
        assert mode != None, "Give me a name of Env type => CartPole or Atari??"
        print("Loading Params for {} Environment".format(mode))

        # load generic params
        self.reward_buffer_ep = 2
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.batch_size = 32
        self.test_episodes = 10
        self.tau = 1. # used in BoltzmannQPolicy
        self.clip = (-500., 500.) # used in BoltzmannQPolicy


        # load params corresponding to the algo
        if algo == "DQN":
            self._load_DQN(mode)
        elif algo == "Double_DQN":
            self._load_Double_DQN(mode)
        elif algo == "DQN_PER":
            self._load_DQN_PER(mode)
        elif algo == "Duelling_DQN":
            self._load_Duelling_DQN(mode)
        elif algo == "Duelling_Double_DQN_PER":
            self._load_Duelling_Double_DQN_PER(mode)
        elif algo == "DQfD":
            self._load_DQfD(mode)
        elif algo == "REINFORCE":
            self._load_REINFORCE(mode)


        # load params corresponding to the env type
        if mode == "CartPole":
            self.goal = 195
            self.num_frames = 30000
            self.num_episodes = 4000
            self.memory_size = 20000
            self.learning_start = 100

        elif mode == "Atari":
            self.goal = 20
            self.num_frames = 30000
            self.num_episodes = 100
            self.memory_size = 1000000
            self.learning_start = 50000


    def _load_DQN(self, mode):
        if mode == "Atari":
            self.state_reshape = (1, 84, 84, 1)
            self.loss_fn = "MSE"
            self.policy_fn = "Eps"
            self.grad_clip_flg = "by_value"
            self.sync_freq = 10000
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "curved"
            self.decay_steps = 1000
        elif mode == "CartPole":
            self.state_reshape = (1, 4)
            self.loss_fn = "MSE"
            self.policy_fn = "Eps"
            self.grad_clip_flg = "by_value"
            self.sync_freq = 100
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "linear"
            self.decay_steps = 1000

    def _load_Double_DQN(self, mode):
        if mode == "Atari":
            self.state_reshape = (1, 84, 84, 1)
            self.loss_fn = "MSE"
            self.policy_fn = "Eps"
            self.grad_clip_flg = "by_value"
            self.sync_freq = 10000
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "curved"
            self.decay_steps = 1000
        elif mode == "CartPole":
            self.state_reshape = (1, 4)
            self.loss_fn = "MSE"
            self.policy_fn = "Eps"
            self.grad_clip_flg = "by_value"
            self.sync_freq = 100
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "linear"
            self.decay_steps = 1000

    def _load_DQN_PER(self, mode):
        if mode == "Atari":
            self.state_reshape = (1, 84, 84, 1)
            self.loss_fn = "MSE"
            self.policy_fn = "Eps"
            self.grad_clip_flg = "by_value"
            self.sync_freq = 10000
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "curved"
            self.decay_steps = 1000
            self.prioritized_replay_alpha = 0.6
            self.prioritized_replay_beta_start = 0.4
            self.prioritized_replay_beta_end = 1.0
            self.prioritized_replay_noise = 1e-6
        elif mode == "CartPole":
            self.state_reshape = (1, 4)
            self.loss_fn = "MSE"
            self.policy_fn = "Eps"
            self.grad_clip_flg = "by_value"
            self.sync_freq = 100
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "linear"
            self.decay_steps = 1000
            self.prioritized_replay_alpha = 0.6
            self.prioritized_replay_beta_start = 0.4
            self.prioritized_replay_beta_end = 1.0
            self.prioritized_replay_noise = 1e-6

    def _load_Duelling_DQN(self, mode):
        if mode == "Atari":
            self.state_reshape = (1, 84, 84, 1)
            self.loss_fn = "MSE"
            self.policy_fn = "Eps"
            self.grad_clip_flg = "by_value"
            self.sync_freq = 10000
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "curved"
            self.decay_steps = 1000
        elif mode == "CartPole":
            self.state_reshape = (1, 4)
            self.loss_fn = "MSE"
            self.policy_fn = "Eps"
            self.grad_clip_flg = "by_value"
            self.sync_freq = 100
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "linear"
            self.decay_steps = 1000

    def _load_Duelling_Double_DQN_PER(self, mode):
        if mode == "Atari":
            self.state_reshape = (1, 84, 84, 1)
            self.loss_fn = "MSE"
            self.policy_fn = "Eps"
            self.grad_clip_flg = "by_value"
            self.sync_freq = 10000
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "curved"
            self.decay_steps = 1000
            self.prioritized_replay_alpha = 0.6
            self.prioritized_replay_beta_start = 0.4
            self.prioritized_replay_beta_end = 1.0
            self.prioritized_replay_noise = 1e-6
        elif mode == "CartPole":
            self.state_reshape = (1, 4)
            self.loss_fn = "MSE"
            self.policy_fn = "Eps"
            self.grad_clip_flg = "by_value"
            self.sync_freq = 100
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "linear"
            self.decay_steps = 1000
            self.prioritized_replay_alpha = 0.6
            self.prioritized_replay_beta_start = 0.4
            self.prioritized_replay_beta_end = 1.0
            self.prioritized_replay_noise = 1e-6

    def _load_DQfD(self, mode):
        if mode == "Atari":
            self.state_reshape = (1, 84, 84, 1)
            self.loss_fn = "MSE"
            self.policy_fn = "Eps"
            self.grad_clip_flg = "by_value"
            self.sync_freq = 10000
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "curved"
            self.decay_steps = 1000
            self.prioritized_replay_alpha = 0.6
            self.prioritized_replay_beta_start = 0.4
            self.prioritized_replay_beta_end = 1.0
            self.prioritized_replay_noise = 1e-6
        elif mode == "CartPole":
            self.state_reshape = (1, 4)
            # self.loss_fn = "MSE"
            self.policy_fn = "Eps"
            self.grad_clip_flg = "norm"
            self.sync_freq = 100
            self.gamma = 0.99
            self.update_hard_or_soft = "soft"
            self.soft_update_tau = 1e-2
            self.decay_type = "linear"
            self.decay_steps = 1000
            self.prioritized_replay_alpha = 0.6
            self.prioritized_replay_beta_start = 0.4
            self.prioritized_replay_beta_end = 1.0
            self.prioritized_replay_noise = 1e-6

    def _load_REINFORCE(self, mode):
        if mode == "Atari":
            self.state_reshape = (1, 84, 84, 1)
            self.loss_fn = "MSE"
            self.policy_fn = "Eps"
            self.grad_clip_flg = "by_value"
            self.sync_freq = 10000
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "curved"
            self.decay_steps = 100
        elif mode == "CartPole":
            self.state_reshape = (1, 4)
            self.loss_fn = "MSE"
            self.policy_fn = "Eps"
            self.grad_clip_flg = "by_value"
            self.sync_freq = 100
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "linear"
            self.decay_steps = 100


class logdirs:
    def __init__(self):
        self.model_DQN = "../logs/models/DQN/"
        self.log_DQN = "../logs/summaries/DQN/"
        self.model_Double_DQN = "../logs/models/Double_DQN/"
        self.log_Double_DQN = "../logs/summaries/Double_DQN/"
        self.model_DQN_PER = "../logs/models/DQN_PER/"
        self.log_DQN_PER = "../logs/summaries/DQN_PER/"
        self.model_Duelling_DQN = "../logs/models/Duelling_DQN/"
        self.log_Duelling_DQN = "../logs/summaries/Duelling_DQN/"
        self.model_Duelling_Double_DQN_PER = "../logs/models/Duelling_Double_DQN_PER/"
        self.log_Duelling_Double_DQN_PER = "../logs/summaries/Duelling_Double_DQN_PER/"
        self.model_DQfD = "../logs/models/DQfD/"
        self.log_DQfD = "../logs/summaries/DQfD/"
        self.model_DQN_afp = "../logs/models/DQN_afp/"
        self.log_DQN_afp = "../logs/summaries/DQN_afp/"