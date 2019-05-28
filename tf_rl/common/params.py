"""
TODO: maybe we don't need Parameters class anymore. Consider just rely on argparse.

Empirically Fixed Parameters for agents

===== ALL PARAMS =====
state_reshape: state shape for pre-processing observations
loss_fn: types of loss function => MSE or huber_loss
policy_fn: types of policy function => Epsilon Greedy(Eps) or Boltzmann(Boltzmann)
grad_clip_flg: types of a cilpping method of gradients => by value(by_value) or global norm(norm)
num_frames: total frame in a training
num_episodes: total episode in a training
memory_size: memory size in a training => this used for Experience Replay Memory or Prioritised Experience Replay Memory
learning_start: frame number which specifies when to start updating the agent
sync_freq: frequency of updating a target model
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
reward_buffer_ep = 10

"""

class Parameters:
    def __init__(self, algo=None, mode=None, env_name=None):
        assert algo != None, "Give me a name of the learning algorithm"
        assert mode != None, "Give me a name of Env type => CartPole or Atari??"
        print("Loading Params for {} Environment".format(mode))


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
        elif algo == "DDPG":
            self._load_DDPG(mode)
        elif algo == "HER":
            self._load_HER(mode)


        # load params corresponding to the env type
        if mode == "Atari":
            self.goal = ENV_LIST_NATURE["{}NoFrameskip-v4".format(env_name)]
            self.num_frames = 200_000_000
            self.memory_size = 400_000
            self.learning_start = 20_000
        elif mode == "CartPole":
            self.goal = 195
            self.num_frames = 10_000
            self.memory_size = 5_000
            self.learning_start = 100


    def _load_DQN(self, mode):
        if mode == "Atari":
            self.state_reshape = (1, 84, 84, 1)
            self.loss_fn = "MSE"
            self.grad_clip_flg = "norm"
            self.sync_freq = 1000
            self.train_interval = 4
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "curved"
            self.decay_steps = 1_000_000
            self.reward_buffer_ep = 10
            self.epsilon_start = 1.0
            self.epsilon_end = 0.1
            self.batch_size = 32
            self.test_episodes = 10
        elif mode == "CartPole":
            self.state_reshape = (1, 4)
            self.loss_fn = "huber_loss"
            self.grad_clip_flg = "None"
            self.sync_freq = 1000
            self.train_interval = 1
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "linear"
            self.decay_steps = 3_000
            self.reward_buffer_ep = 10
            self.epsilon_start = 1.0
            self.epsilon_end = 0.02
            self.batch_size = 32
            self.test_episodes = 10

    def _load_Double_DQN(self, mode):
        if mode == "Atari":
            self.state_reshape = (1, 84, 84, 1)
            self.loss_fn = "MSE"
            self.grad_clip_flg = "norm"
            self.sync_freq = 10000
            self.train_interval = 4
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "curved"
            self.decay_steps = 1_000_000
            self.reward_buffer_ep = 10
            self.epsilon_start = 1.0
            self.epsilon_end = 0.1
            self.batch_size = 32
            self.test_episodes = 10
        elif mode == "CartPole":
            self.state_reshape = (1, 4)
            self.loss_fn = "huber_loss"
            self.grad_clip_flg = "None"
            self.sync_freq = 1000
            self.train_interval = 1
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "linear"
            self.decay_steps = 10_000
            self.reward_buffer_ep = 10
            self.epsilon_start = 1.0
            self.epsilon_end = 0.02
            self.batch_size = 32
            self.test_episodes = 10

    def _load_DQN_PER(self, mode):
        if mode == "Atari":
            self.state_reshape = (1, 84, 84, 1)
            self.loss_fn = "MSE"
            self.policy_fn = "Eps"
            self.grad_clip_flg = "by_value"
            self.sync_freq = 10000
            self.train_interval = 4
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "curved"
            self.decay_steps = 1000
            self.reward_buffer_ep = 10
            self.epsilon_start = 1.0
            self.epsilon_end = 0.1
            self.batch_size = 32
            self.test_episodes = 10
            self.tau = 1. # used in BoltzmannQPolicy
            self.clip = (-500., 500.) # used in BoltzmannQPolicy
            self.prioritized_replay_alpha = 0.6
            self.prioritized_replay_beta_start = 0.4
            self.prioritized_replay_beta_end = 1.0
            self.prioritized_replay_noise = 1e-6
        elif mode == "CartPole":
            self.state_reshape = (1, 4)
            self.loss_fn = "huber_loss"
            self.grad_clip_flg = "None"
            self.sync_freq = 1000
            self.train_interval = 1
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "linear"
            self.decay_steps = 10_000
            self.reward_buffer_ep = 10
            self.epsilon_start = 1.0
            self.epsilon_end = 0.02
            self.batch_size = 32
            self.test_episodes = 10
            self.prioritized_replay_alpha = 0.6
            self.prioritized_replay_beta_start = 0.4
            self.prioritized_replay_beta_end = 1.0
            self.prioritized_replay_noise = 1e-6

    def _load_Duelling_DQN(self, mode):
        if mode == "Atari":
            self.state_reshape = (1, 84, 84, 1)
            self.loss_fn = "MSE"
            self.policy_fn = "Eps"
            self.grad_clip_flg = "norm"
            self.sync_freq = 10000
            self.train_interval = 4
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "curved"
            self.decay_steps = 1_000_000
            self.reward_buffer_ep = 10
            self.epsilon_start = 1.0
            self.epsilon_end = 0.1
            self.batch_size = 32
            self.test_episodes = 10
            self.tau = 1. # used in BoltzmannQPolicy
            self.clip = (-500., 500.) # used in BoltzmannQPolicy
        elif mode == "CartPole":
            self.state_reshape = (1, 4)
            self.loss_fn = "huber_loss"
            self.grad_clip_flg = "None"
            self.sync_freq = 1000
            self.train_interval = 1
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "linear"
            self.decay_steps = 10_000
            self.reward_buffer_ep = 10
            self.epsilon_start = 1.0
            self.epsilon_end = 0.02
            self.batch_size = 32
            self.test_episodes = 10

    def _load_Duelling_Double_DQN_PER(self, mode):
        if mode == "Atari":
            self.state_reshape = (1, 84, 84, 1)
            self.loss_fn = "MSE"
            self.policy_fn = "Eps"
            self.grad_clip_flg = "by_value"
            self.sync_freq = 10000
            self.train_interval = 4
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "curved"
            self.decay_steps = 1000
            self.reward_buffer_ep = 10
            self.epsilon_start = 1.0
            self.epsilon_end = 0.1
            self.batch_size = 32
            self.test_episodes = 10
            self.tau = 1.  # used in BoltzmannQPolicy
            self.clip = (-500., 500.)  # used in BoltzmannQPolicy
            self.prioritized_replay_alpha = 0.6
            self.prioritized_replay_beta_start = 0.4
            self.prioritized_replay_beta_end = 1.0
            self.prioritized_replay_noise = 1e-6
        elif mode == "CartPole":
            self.state_reshape = (1, 4)
            self.loss_fn = "huber_loss"
            self.grad_clip_flg = "None"
            self.sync_freq = 1000
            self.train_interval = 1
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "linear"
            self.decay_steps = 10_000
            self.reward_buffer_ep = 10
            self.epsilon_start = 1.0
            self.epsilon_end = 0.02
            self.batch_size = 32
            self.test_episodes = 10
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
            self.train_interval = 4
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "curved"
            self.decay_steps = 1000
            self.reward_buffer_ep = 10
            self.epsilon_start = 1.0
            self.epsilon_end = 0.1
            self.batch_size = 32
            self.test_episodes = 10
            self.tau = 1. # used in BoltzmannQPolicy
            self.clip = (-500., 500.) # used in BoltzmannQPolicy
            self.prioritized_replay_alpha = 0.6
            self.prioritized_replay_beta_start = 0.4
            self.prioritized_replay_beta_end = 1.0
            self.prioritized_replay_noise = 1e-6
        elif mode == "CartPole":
            self.state_reshape = (1, 4)
            self.loss_fn = "huber_loss"
            self.grad_clip_flg = "None"
            self.sync_freq = 1000
            self.train_interval = 1
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "linear"
            self.decay_steps = 10_000
            self.reward_buffer_ep = 10
            self.epsilon_start = 1.0
            self.epsilon_end = 0.02
            self.batch_size = 32
            self.test_episodes = 10
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
            self.reward_buffer_ep = 10
            self.epsilon_start = 1.0
            self.epsilon_end = 0.1
            self.batch_size = 32
            self.test_episodes = 10
            self.tau = 1. # used in BoltzmannQPolicy
            self.clip = (-500., 500.) # used in BoltzmannQPolicy
        elif mode == "CartPole":
            self.state_reshape = (1, 4)
            self.loss_fn = "huber_loss"
            self.grad_clip_flg = "None"
            self.sync_freq = 1000
            self.train_interval = 1
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "linear"
            self.decay_steps = 10_000
            self.reward_buffer_ep = 10
            self.epsilon_start = 1.0
            self.epsilon_end = 0.02
            self.batch_size = 32
            self.test_episodes = 10

    def _load_DDPG(self, mode):
        if mode == "Atari": # not used
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
            self.reward_buffer_ep = 10
            self.epsilon_start = 1.0
            self.epsilon_end = 0.1
            self.batch_size = 32
            self.test_episodes = 10
            self.tau = 1. # used in BoltzmannQPolicy
            self.clip = (-500., 500.) # used in BoltzmannQPolicy
        elif mode == "CartPole": # not cartpole but pendulum
            self.state_reshape = (1, 4)
            self.loss_fn = "huber_loss"
            self.grad_clip_flg = "None"
            self.sync_freq = 1000
            self.train_interval = 1
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "linear"
            self.decay_steps = 10_000
            self.reward_buffer_ep = 10
            self.epsilon_start = 1.0
            self.epsilon_end = 0.02
            self.batch_size = 32
            self.test_episodes = 10

    def _load_HER(self, mode):
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
            self.reward_buffer_ep = 10
            self.epsilon_start = 1.0
            self.epsilon_end = 0.1
            self.batch_size = 32
            self.test_episodes = 10
            self.tau = 1. # used in BoltzmannQPolicy
            self.clip = (-500., 500.) # used in BoltzmannQPolicy
        elif mode == "CartPole":
            self.state_reshape = (1, 4)
            self.loss_fn = "huber_loss"
            self.grad_clip_flg = "None"
            self.sync_freq = 1000
            self.train_interval = 1
            self.gamma = 0.99
            self.update_hard_or_soft = "hard"
            self.soft_update_tau = 1e-2
            self.decay_type = "linear"
            self.decay_steps = 10_000
            self.reward_buffer_ep = 10
            self.epsilon_start = 1.0
            self.epsilon_end = 0.02
            self.batch_size = 32
            self.test_episodes = 10

"""
Env list and scores

"""


# https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
ENV_LIST_NIPS = {
    "BeamRiderNoFrameskip-v4": 6846.0,
    "BreakoutNoFrameskip-v4": 401.2,
    "EnduroNoFrameskip-v4": 301.8,
    "PongNoFrameskip-v4": 18.9,
    "QbertNoFrameskip-v4": 10596,
    "SeaquestNoFrameskip-v4": 5286.0,
    "SpaceInvadersNoFrameskip-v4": 1976.0
}


# https://www.nature.com/articles/nature14236.pdf
ENV_LIST_NATURE = {
    "VideoPinballNoFrameskip-v4": 42684.0,
    "BoxingNoFrameskip-v4": 71.8,
    "BreakoutNoFrameskip-v4": 401.2,
    "StarGunnerNoFrameskip-v4": 57997.0,
    "RobotankNoFrameskip-v4": 51.6,
    "AtlantisNoFrameskip-v4": 85641.0,
    "CrazyClimberNoFrameskip-v4": 114103.0,
    "GopherNoFrameskip-v4": 8520.0,
    "DemonAttackNoFrameskip-v4": 9711.0,
    "NameThisGameNoFrameskip-v4": 7257.0,
    "KrullNoFrameskip-v4": 3805.0,
    "AssaultNoFrameskip-v4": 3359.0,
    "RoadRunnerNoFrameskip-v4": 18257.0,
    "KangarooNoFrameskip-v4": 6740.0,
    "JamesbondNoFrameskip-v4": 576.7,
    "TennisNoFrameskip-v4": -2.5,
    "PongNoFrameskip-v4": 18.9,
    "SpaceInvadersNoFrameskip-v4": 1976.0,
    "BeamRiderNoFrameskip-v4": 6846.0,
    "TutankhamNoFrameskip-v4": 186.7,
    "KungFuMasterNoFrameskip-v4": 23270.0,
    "FreewayNoFrameskip-v4": 30.3,
    "TimePilotNoFrameskip-v4": 5947.0,
    "EnduroNoFrameskip-v4": 301.8,
    "FishingDerbyNoFrameskip-v4": -0.8,
    "UpNDownNoFrameskip-v4": 8456.0,
    "IceHockeyNoFrameskip-v4": -1.6,
    "QbertNoFrameskip-v4": 10596.0,
    "HeroNoFrameskip-v4": 19950.0,
    "AsterixNoFrameskip-v4": 6012.0,
    "BattleZoneNoFrameskip-v4": 26300.0,
    "WizardOfWorNoFrameskip-v4": 4757.0,
    "ChopperCommandNoFrameskip-v4": 6687.0,
    "CentipedeNoFrameskip-v4": 8309.0,
    "BankHeistNoFrameskip-v4": 429.7,
    "RiverraidNoFrameskip-v4": 8316.0,
    "ZaxxonNoFrameskip-v4": 4977.0,
    "AmidarNoFrameskip-v4": 739.5,
    "AlienNoFrameskip-v4": 3069.0,
    "VentureNoFrameskip-v4": 380.0,
    "SeaquestNoFrameskip-v4": 5286.0,
    "DoubleDunkNoFrameskip-v4": -18.1,
    "BowlingNoFrameskip-v4": 42.4,
    "MsPacmanNoFrameskip-v4": 2311.0,
    "AsteroidsNoFrameskip-v4": 1629.0,
    "FrostbiteNoFrameskip-v4": 328.3,
    "GravitarNoFrameskip-v4": 306.7,
    "PrivateEyeNoFrameskip-v4": 1788.0,
    "MontezumaRevengeNoFrameskip-v4": 0.0
}


# https://gym.openai.com/envs/#mujoco
# https://github.com/openai/baselines-results/blob/master/param-noise/mujoco.md
DDPG_ENV_LIST = {
    "Ant-v2": 3500,
    "HalfCheetah-v2": 7000,
    "Hopper-v2": 1500,
    "Humanoid-v2": 2000,
    "HumanoidStandup-v2": 0,
    "InvertedDoublePendulum-v2": 6000,
    "InvertedPendulum-v2": 800,
    "Reacher-v2": -6,
    "Swimmer-v2": 40,
    "Walker2d-v2": 2500
}