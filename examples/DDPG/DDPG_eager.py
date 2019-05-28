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
tf.random.set_random_seed(123)

class Actor(tf.keras.Model):
	def __init__(self, num_action):
		super(Actor, self).__init__()
		self.dense1 = tf.keras.layers.Dense(400, activation='relu')
		self.batch = tf.keras.layers.BatchNormalization()
		self.dense2 = tf.keras.layers.Dense(300, activation='relu')
		self.pred = tf.keras.layers.Dense(num_action, activation='linear')

	@tf.contrib.eager.defun
	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.batch(x)
		x = self.dense2(x)
		pred = self.pred(x)
		return pred


class Critic(tf.keras.Model):
	def __init__(self, output_shape):
		super(Critic, self).__init__()
		self.dense1 = tf.keras.layers.Dense(400, activation='relu')
		self.batch = tf.keras.layers.BatchNormalization()
		self.dense2 = tf.keras.layers.Dense(300, activation='relu')
		self.pred = tf.keras.layers.Dense(output_shape, activation='linear')

	@tf.contrib.eager.defun
	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.batch(x)
		x = self.dense2(x)
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
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_name", default="HalfCheetah-v2", help="Env title")
	parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
	parser.add_argument("--train_interval", default=100, type=int, help="a frequency of training occurring in training phase")
	parser.add_argument("--nb_train_steps", default=50, type=int, help="a number of training, which occurs once in train_interval above!!")
	parser.add_argument("--eval_interval", default=50_000, type=int, help="a frequency of evaluation occurring in training phase")
	parser.add_argument("--memory_size", default=1_000_000, type=int, help="memory size in a training")
	parser.add_argument("--learning_start", default=2_000, type=int, help="frame number which specifies when to start updating the agent")
	parser.add_argument("--batch_size", default=32, type=int, help="batch size of each iteration of update")
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
	params.test_episodes = 10

	env = gym.make(params.env_name)
	params.goal = DDPG_ENV_LIST[params.env_name]
	agent = DDPG(Actor, Critic, env.action_space.shape[0], params)
	replay_buffer = ReplayBuffer(params.memory_size)
	reward_buffer = deque(maxlen=params.reward_buffer_ep)
	summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)
	train_DDPG(agent, env, replay_buffer, reward_buffer, params, summary_writer)