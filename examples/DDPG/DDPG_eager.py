import gym
import argparse
import tensorflow as tf
from collections import deque
from tf_rl.common.memory import ReplayBuffer
from tf_rl.agents.DDPG import DDPG
from tf_rl.common.train import train_DDPG
from tf_rl.common.params import DDPG_ENV_LIST

config = tf.ConfigProto(allow_soft_placement=True,
						intra_op_parallelism_threads=1,
						inter_op_parallelism_threads=1)
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
tf.enable_resource_variables()

regulariser = tf.keras.regularizers.l2(1e-2)
kernel_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)

class Actor(tf.keras.Model):
	def __init__(self, num_action=1):
		super(Actor, self).__init__()
		self.dense1 = tf.keras.layers.Dense(400, activation='relu', kernel_initializer=kernel_init)
		self.batch1 = tf.keras.layers.BatchNormalization()
		self.dense2 = tf.keras.layers.Dense(300, activation='relu', kernel_initializer=kernel_init)
		self.batch2 = tf.keras.layers.BatchNormalization()
		self.pred = tf.keras.layers.Dense(num_action, activation='tanh', kernel_initializer=kernel_init)

	@tf.contrib.eager.defun(autograph=False)
	def call(self, inputs):
		x = self.dense1(inputs)
		# x = self.batch1(x) # without batch norm, it works better than with it
		x = self.dense2(x)
		# x = self.batch2(x) # without batch norm, it works better than with it
		pred = self.pred(x)
		return pred


class Critic(tf.keras.Model):
	def __init__(self, output_shape):
		super(Critic, self).__init__()
		self.dense1 = tf.keras.layers.Dense(400, activation='relu', kernel_regularizer=regulariser, bias_regularizer=regulariser, kernel_initializer=kernel_init)
		self.batch1 = tf.keras.layers.BatchNormalization()
		self.dense2 = tf.keras.layers.Dense(300, activation='relu', kernel_regularizer=regulariser, bias_regularizer=regulariser, kernel_initializer=kernel_init)
		self.batch2 = tf.keras.layers.BatchNormalization()
		self.pred = tf.keras.layers.Dense(output_shape, activation='linear', kernel_regularizer=regulariser, bias_regularizer=regulariser, kernel_initializer=kernel_init)

	@tf.contrib.eager.defun(autograph=False)
	def call(self, obs, act):
		x = self.dense1(obs)
		# x = self.batch1(x) # without batch norm, it works better than with it
		x = self.dense2(tf.concat([x, act], axis=-1))
		# x = self.batch2(x) # without batch norm, it works better than with it
		pred = self.pred(x)
		return pred


# ======= Atari-like High Dim input Models ========

# class Actor(tf.keras.Model):
# 	"""
# 	For Atari
#
# 	"""
# 	def __init__(self, env_type, num_action):
# 		super(Actor, self).__init__()
# 		self.env_type = env_type
# 		self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
# 		self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
# 		self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
# 		self.flat = tf.keras.layers.Flatten()
# 		self.fc1 = tf.keras.layers.Dense(512, activation='relu')
# 		self.pred = tf.keras.layers.Dense(num_action, activation='linear')
#
# 	@tf.contrib.eager.defun
# 	def call(self, inputs):
# 		x = self.conv1(inputs)
# 		x = self.conv2(x)
# 		x = self.conv3(x)
# 		x = self.flat(x)
# 		x = self.fc1(x)
# 		pred = self.pred(x)
# 		return pred


# class Critic(tf.keras.Model):
# 	"""
#
# 	For Atari
#
# 	"""
# 	def __init__(self, env_type, num_action):
# 		super(Critic, self).__init__()
# 		self.env_type = env_type
# 		self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
# 		self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
# 		self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
# 		self.flat = tf.keras.layers.Flatten()
# 		self.fc1 = tf.keras.layers.Dense(512, activation='relu')
# 		self.pred = tf.keras.layers.Dense(num_action, activation='linear')
#
# 	@tf.contrib.eager.defun
# 	def call(self, inputs):
# 		x = self.conv1(inputs)
# 		x = self.conv2(x)
# 		x = self.conv3(x)
# 		x = self.flat(x)
# 		x = self.fc1(x)
# 		pred = self.pred(x)
# 		return pred


if __name__ == '__main__':
	"""
	this is defined in params.py
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
	"""

	parser = argparse.ArgumentParser()
	parser.add_argument("--env_name", default="HalfCheetah-v2", help="Env title")
	parser.add_argument("--seed", default=123, type=int, help="seed for randomness")
	parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
	parser.add_argument("--train_interval", default=100, type=int, help="a frequency of training occurring in training phase")
	parser.add_argument("--nb_train_steps", default=50, type=int, help="a number of training, which occurs once in train_interval above!!")
	parser.add_argument("--eval_interval", default=5_000, type=int, help="a frequency of evaluation occurring in training phase")
	parser.add_argument("--memory_size", default=100_000, type=int, help="memory size in a training")
	parser.add_argument("--learning_start", default=10_000, type=int, help="frame number which specifies when to start updating the agent")
	parser.add_argument("--batch_size", default=100, type=int, help="batch size of each iteration of update")
	parser.add_argument("--reward_buffer_ep", default=5, type=int, help="reward_buffer size")
	parser.add_argument("--gamma", default=0.99, type=float, help="discount factor => gamma > 1.0 or negative => does not converge!!")
	parser.add_argument("--soft_update_tau", default=1e-2, type=float, help="soft-update needs tau to define the ratio of main model remains")
	parser.add_argument("--L2_reg", default=1e-2, type=float, help="magnitude of L2 regularisation")
	parser.add_argument("--action_range", default=[-1., 1.], type=list, help="magnitude of L2 regularisation")
	parser.add_argument("--log_dir", default="../../logs/logs/DDPG/", help="directory for log")
	parser.add_argument("--model_dir", default="../../logs/models/DDPG/", help="directory for trained model")
	parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
	parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
	params = parser.parse_args()

	from datetime import datetime
	now = datetime.now()
	params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-DDPG/"
	params.model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-DDPG/"

	params.test_episodes = 10

	env = gym.make(params.env_name)
	# set seed
	env.seed(params.seed)
	tf.random.set_random_seed(params.seed)

	params.goal = DDPG_ENV_LIST[params.env_name]
	agent = DDPG(Actor, Critic, env.action_space.shape[0], params)
	replay_buffer = ReplayBuffer(params.memory_size)
	reward_buffer = deque(maxlen=params.reward_buffer_ep)
	summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)
	train_DDPG(agent, env, replay_buffer, reward_buffer, params, summary_writer)