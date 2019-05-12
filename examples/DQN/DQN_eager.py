import os
import gym
import argparse
from datetime import datetime
import tensorflow as tf
from collections import deque
from tf_rl.common.wrappers import MyWrapper, wrap_deepmind, make_atari, CartPole_Pixel
from tf_rl.common.params import Parameters
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import AnnealingSchedule
from tf_rl.common.policy import EpsilonGreedyPolicy_eager
from tf_rl.common.train import train_DQN
from tf_rl.agents.DQN import DQN, DQN_new

tf.enable_eager_execution()
tf.random.set_random_seed(123)

class Model(tf.keras.Model):
	def __init__(self, env_type, num_action):
		super(Model, self).__init__()
		self.env_type = env_type
		if self.env_type == "CartPole":
			self.dense1 = tf.keras.layers.Dense(16, activation='relu')
			self.dense2 = tf.keras.layers.Dense(16, activation='relu')
			self.dense3 = tf.keras.layers.Dense(16, activation='relu')
			self.pred = tf.keras.layers.Dense(num_action, activation='linear')
		elif self.env_type == "Atari":
			self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
			self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
			self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
			self.flat = tf.keras.layers.Flatten()
			self.fc1 = tf.keras.layers.Dense(512, activation='relu')
			self.pred = tf.keras.layers.Dense(num_action, activation='linear')

	def call(self, inputs):
		if self.env_type == "CartPole":
			x = self.dense1(inputs)
			x = self.dense2(x)
			x = self.dense3(x)
			pred = self.pred(x)
			return pred
		elif self.env_type == "Atari":
			x = self.conv1(inputs)
			x = self.conv2(x)
			x = self.conv3(x)
			x = self.flat(x)
			x = self.fc1(x)
			pred = self.pred(x)
			return pred


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument("--mode", default="CartPole", help="game env type => Atari or CartPole")
	# parser.add_argument("--loss_fn", default="MSE", help="types of loss function => MSE or huber_loss")
	# parser.add_argument("--grad_clip_flg", default="norm", help="types of a clipping method of gradients => by value(by_value) or global norm(norm) or None")
	# parser.add_argument("--num_frames", default=10000, type=int, help="total frame in a training")
	# parser.add_argument("--train_interval", default=4, type=int, help="a frequency of training occurring in training phase")
	# parser.add_argument("--eval_interval", default=2000, type=int, help="a frequency of evaluation occurring in training phase")
	# parser.add_argument("--memory_size", default=5000, type=int, help="memory size in a training => this used for Experience Replay Memory or Prioritised Experience Replay Memory")
	# parser.add_argument("--learning_start", default=200, type=int, help="frame number which specifies when to start updating the agent")
	# parser.add_argument("--sync_freq", default=100, type=int, help="frequency of updating a target model")
	# parser.add_argument("--batch_size", default=128, type=int, help="batch size of each iteration of update")
	# parser.add_argument("--gamma", default=0.99, type=float, help="discount factor => gamma > 1.0 or negative => does not converge!!")
	# parser.add_argument("--update_hard_or_soft", default="hard", help="types of synchronisation method of target and main models => soft or hard update")
	# parser.add_argument("--soft_update_tau", default=1e-2, type=float, help="in soft-update tau defines the ratio of main model remains and it seems 1e-2 is the optimal!")
	# parser.add_argument("--epsilon_start", default=1.0, type=float, help="initial value of epsilon")
	# parser.add_argument("--epsilon_end", default=0.02, type=float, help="final value of epsilon")
	# parser.add_argument("--decay_steps", default=4000, type=int, help="a period for annealing a value(epsilon or beta)")
	# parser.add_argument("--decay_type", default="linear", help="types of annealing method => linear or curved")
	# parser.add_argument("--log_dir", default="../../logs/logs/DQN/", help="directory for log")
	# parser.add_argument("--model_dir", default="../../logs/models/DQN/", help="directory for trained model")
	# parser.add_argument("--new_or_old", default="old", help="temp")


	# ====== Params for Atari ======
	parser.add_argument("--mode", default="Atari", help="game env type => Atari or CartPole")
	parser.add_argument("--env_name", default="Breakout", help="game title")
	parser.add_argument("--loss_fn", default="MSE", help="types of loss function => MSE or huber_loss")
	parser.add_argument("--grad_clip_flg", default="norm", help="types of a clipping method of gradients => by value(by_value) or global norm(norm) or None")
	parser.add_argument("--num_frames", default=20_000_000, type=int, help="total frame in a training")
	parser.add_argument("--train_interval", default=4, type=int, help="a frequency of training occurring in training phase")
	parser.add_argument("--eval_interval", default=250_000, type=int, help="a frequency of evaluation occurring in training phase")
	parser.add_argument("--memory_size", default=500_000, type=int, help="memory size in a training => this used for Experience Replay Memory or Prioritised Experience Replay Memory")
	parser.add_argument("--learning_start", default=20_000, type=int, help="frame number which specifies when to start updating the agent")
	parser.add_argument("--sync_freq", default=1_000, type=int, help="frequency of updating a target model")
	parser.add_argument("--batch_size", default=128, type=int, help="batch size of each iteration of update")
	parser.add_argument("--gamma", default=0.99, type=float, help="discount factor => gamma > 1.0 or negative => does not converge!!")
	parser.add_argument("--update_hard_or_soft", default="hard", help="types of synchronisation method of target and main models => soft or hard update")
	parser.add_argument("--soft_update_tau", default=1e-2, type=float, help="in soft-update tau defines the ratio of main model remains and it seems 1e-2 is the optimal!")
	parser.add_argument("--epsilon_start", default=1.0, type=float, help="initial value of epsilon")
	parser.add_argument("--epsilon_end", default=0.02, type=float, help="final value of epsilon")
	parser.add_argument("--decay_steps", default=250_000, type=int, help="a period for annealing a value(epsilon or beta)")
	parser.add_argument("--decay_type", default="linear", help="types of annealing method => linear or curved")
	parser.add_argument("--log_dir", default="../../logs/logs/DQN/", help="directory for log")
	parser.add_argument("--model_dir", default="../../logs/models/DQN/", help="directory for trained model")
	parser.add_argument("--new_or_old", default="new", help="new(vectorised back-prop), old(single-value back-prop)")
	parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")

	args = parser.parse_args()

	# I know this is not beautiful, but for the sake of ease of dev and finding the best params,
	# i will leave this for a while
	# TODO: you need to amend this design to the one only args, instead of params
	params = Parameters(algo="DQN", mode=args.mode, env_name=args.env_name)
	params.loss_fn = args.loss_fn
	params.grad_clip_flg = args.grad_clip_flg
	params.num_frames = args.num_frames
	params.memory_size = args.memory_size
	params.learning_start = args.learning_start
	params.train_interval = args.train_interval
	params.eval_interval = args.eval_interval
	params.sync_freq = args.sync_freq
	params.batch_size = args.batch_size
	params.gamma = args.gamma
	params.update_hard_or_soft = args.update_hard_or_soft
	params.soft_update_tau = args.soft_update_tau
	params.epsilon_start = args.epsilon_start
	params.epsilon_end = args.epsilon_end
	params.decay_steps = args.decay_steps
	params.decay_type = args.decay_type


	if args.mode == "CartPole":
		env = MyWrapper(gym.make("CartPole-v0"))
	elif args.mode == "Atari":
		env = wrap_deepmind(make_atari("{}NoFrameskip-v4".format(args.env_name)))
	elif args.mode == "CartPole-p":
		env = CartPole_Pixel(gym.make("CartPole-v0"))

	replay_buffer = ReplayBuffer(params.memory_size)

	now = datetime.now()

	if args.google_colab:
		# mount your drive on google colab
		from google.colab import drive
		drive.mount("/content/gdrive")
		args.log_dir = "/content/gdrive/My Drive/logs/logs/DQN/{}".format(args.env_name)
		args.model_dir = "/content/gdrive/My Drive/logs/models/DQN/{}".format(args.env_name)
		os.makedirs(args.log_dir)
		os.makedirs(args.model_dir)
		assert os.path.isdir(args.log_dir), "Faild to create a directory on your My Drive, pls check it"
		assert os.path.isdir(args.model_dir), "Faild to create a directory on your My Drive, pls check it"
		if args.new_or_old == "new":
			agent = DQN_new(args.mode, Model, Model, env.action_space.n, params, args.model_dir)
		else:
			agent = DQN(args.mode, Model, Model, env.action_space.n, params, args.model_dir)
	else:
		if args.new_or_old == "new":
			# args.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "DQN_new_not_pixel/"
			args.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "DQN_new/"
			agent = DQN_new(args.mode, Model, Model, env.action_space.n, params, args.model_dir)
		else:
			# args.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "DQN_old_not_pixel/"
			args.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "DQN_old/"
			agent = DQN(args.mode, Model, Model, env.action_space.n, params, args.model_dir)

	Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
	policy = EpsilonGreedyPolicy_eager(Epsilon_fn=Epsilon)

	reward_buffer = deque(maxlen=params.reward_buffer_ep)
	summary_writer = tf.contrib.summary.create_file_writer(args.log_dir)
	train_DQN(agent, env, policy, replay_buffer, reward_buffer, params, summary_writer)
