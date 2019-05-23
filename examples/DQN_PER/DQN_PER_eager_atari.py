import os
import argparse
from datetime import datetime
import tensorflow as tf
from collections import deque
from tf_rl.common.wrappers import wrap_deepmind, make_atari
from tf_rl.common.params import ENV_LIST_NATURE
from tf_rl.common.memory import PrioritizedReplayBuffer
from tf_rl.common.utils import AnnealingSchedule
from tf_rl.common.policy import EpsilonGreedyPolicy_eager
from tf_rl.common.train import train_DQN_PER
from tf_rl.agents.DQN import DQN, DQN_debug

tf.enable_eager_execution()
tf.random.set_random_seed(123)

class Model(tf.keras.Model):
	def __init__(self, num_action):
		super(Model, self).__init__()
		self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
		self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
		self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
		self.flat = tf.keras.layers.Flatten()
		self.fc1 = tf.keras.layers.Dense(512, activation='relu')
		self.pred = tf.keras.layers.Dense(num_action, activation='linear')

	@tf.contrib.eager.defun
	def call(self, inputs):
		x = self.conv1(inputs)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flat(x)
		x = self.fc1(x)
		return self.pred(x)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", default="Atari", help="game env type => Atari or CartPole")
	parser.add_argument("--env_name", default="Breakout", help="game title")
	parser.add_argument("--loss_fn", default="MSE", help="types of loss function => MSE or huber_loss")
	parser.add_argument("--grad_clip_flg", default="norm", help="types of a clipping method of gradients => by value(by_value) or global norm(norm) or None")
	parser.add_argument("--num_frames", default=10_000_000, type=int, help="total frame in a training")
	parser.add_argument("--train_interval", default=5, type=int, help="a frequency of training occurring in training phase")
	parser.add_argument("--eval_interval", default=250_000, type=int, help="a frequency of evaluation occurring in training phase")
	parser.add_argument("--memory_size", default=500_000, type=int, help="memory size in a training => this used for Experience Replay Memory or Prioritised Experience Replay Memory")
	parser.add_argument("--learning_start", default=20_000, type=int, help="frame number which specifies when to start updating the agent")
	parser.add_argument("--sync_freq", default=1_000, type=int, help="frequency of updating a target model")
	parser.add_argument("--batch_size", default=32, type=int, help="batch size of each iteration of update")
	parser.add_argument("--reward_buffer_ep", default=100, type=int, help="reward_buffer size")
	parser.add_argument("--gamma", default=0.99, type=float, help="discount factor => gamma > 1.0 or negative => does not converge!!")
	parser.add_argument("--update_hard_or_soft", default="hard", help="types of synchronisation method of target and main models => soft or hard update")
	parser.add_argument("--soft_update_tau", default=1e-2, type=float, help="in soft-update tau defines the ratio of main model remains and it seems 1e-2 is the optimal!")
	parser.add_argument("--epsilon_start", default=1.0, type=float, help="initial value of epsilon")
	parser.add_argument("--epsilon_end", default=0.02, type=float, help="final value of epsilon")
	parser.add_argument("--decay_steps", default=250_000, type=int, help="a period for annealing a value(epsilon or beta)")
	parser.add_argument("--decay_type", default="linear", help="types of annealing method => linear or curved")
	parser.add_argument("--log_dir", default="../../logs/logs/DQN/", help="directory for log")
	parser.add_argument("--model_dir", default="../../logs/models/DQN/", help="directory for trained model")
	parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
	parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
	params = parser.parse_args()
	params.goal = ENV_LIST_NATURE["{}NoFrameskip-v4".format(params.env_name)]
	params.test_episodes = 10
	params.prioritized_replay_alpha = 0.6
	params.prioritized_replay_beta_start = 0.4
	params.prioritized_replay_beta_end = 1.0
	params.prioritized_replay_noise = 1e-6

	env = wrap_deepmind(make_atari("{}NoFrameskip-v4".format(params.env_name)))
	now = datetime.now()

	now = datetime.now()

	if params.google_colab:
		# mount your drive on google colab
		from google.colab import drive
		drive.mount("/content/gdrive")
		params.log_dir = "/content/gdrive/My Drive/logs/logs/DQN_PER/{}".format(params.env_name)
		params.model_dir = "/content/gdrive/My Drive/logs/models/DQN_PER/{}".format(params.env_name)
		os.makedirs(params.log_dir)
		os.makedirs(params.model_dir)
		assert os.path.isdir(params.log_dir), "Faild to create a directory on your My Drive, pls check it"
		assert os.path.isdir(params.model_dir), "Faild to create a directory on your My Drive, pls check it"
		if params.debug_flg:
			agent = DQN_debug(Model, Model, env.action_space.n, params)
		else:
			agent = DQN(Model, Model, env.action_space.n, params)
	else:
		# run on the local machine
		if params.debug_flg:
			params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "DQN_PER_debug/"
			params.model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "DQN_PER_debug/"
			agent = DQN_debug(Model, Model, env.action_space.n, params)
		else:
			params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "DQN_PER/"
			params.model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "DQN_PER/"
			agent = DQN(Model, Model, env.action_space.n, params)

	replay_buffer = PrioritizedReplayBuffer(params.memory_size, alpha=params.prioritized_replay_alpha)
	Beta = AnnealingSchedule(start=params.prioritized_replay_beta_start, end=params.prioritized_replay_beta_end,
							 decay_steps=params.decay_steps)
	Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end,
								decay_steps=params.decay_steps)
	policy = EpsilonGreedyPolicy_eager(Epsilon_fn=Epsilon)

	reward_buffer = deque(maxlen=params.reward_buffer_ep)
	summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)
	train_DQN_PER(agent, env, policy, replay_buffer, reward_buffer, params, Beta, summary_writer)