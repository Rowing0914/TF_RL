import gym
import argparse
import os
import tensorflow as tf
from collections import deque
from tf_rl.common.wrappers import MyWrapper, wrap_deepmind, make_atari
from examples.params.params import Parameters, logdirs
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import AnnealingSchedule
from tf_rl.common.policy import EpsilonGreedyPolicy_eager, BoltzmannQPolicy_eager
from tf_rl.common.train import train_DQN
from tf_rl.agents.DQN import DQN

tf.enable_eager_execution()

class Model_CartPole(tf.keras.Model):
	def __init__(self, num_action):
		super(Model_CartPole, self).__init__()
		self.dense1 = tf.keras.layers.Dense(16, activation='relu')
		self.dense2 = tf.keras.layers.Dense(16, activation='relu')
		self.dense3 = tf.keras.layers.Dense(16, activation='relu')
		self.pred = tf.keras.layers.Dense(num_action, activation='linear')

	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.dense2(x)
		x = self.dense3(x)
		pred = self.pred(x)
		return pred


class Model_Atari(tf.keras.Model):
	def __init__(self, num_action):
		super(Model_Atari, self).__init__()
		self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
		self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
		self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
		self.flat = tf.keras.layers.Flatten()
		self.fc1 = tf.keras.layers.Dense(512, activation='relu')
		self.pred = tf.keras.layers.Dense(num_action, activation='linear')

	def call(self, inputs):
		x = self.conv1(inputs)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flat(x)
		x = self.fc1(x)
		pred = self.pred(x)
		return pred


if __name__ == '__main__':
	logdirs = logdirs()
	try:
		os.system("rm -rf {}".format(logdirs.log_DQN))
	except:
		pass

	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", default="CartPole", help="game env type")
	args = parser.parse_args()

	if args.mode == "CartPole":
		env = MyWrapper(gym.make("CartPole-v0"))
		params = Parameters(algo="DQN", mode="CartPole")
		replay_buffer = ReplayBuffer(params.memory_size)
		agent = DQN(Model_CartPole, Model_CartPole, env.action_space.n, params, logdirs.model_DQN)
		if params.policy_fn == "Eps":
			Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end,
										decay_steps=params.decay_steps)
			policy = EpsilonGreedyPolicy_eager(Epsilon_fn=Epsilon)
		elif params.policy_fn == "Boltzmann":
			policy = BoltzmannQPolicy_eager()
	elif args.mode == "Atari":
		env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
		params = Parameters(algo="DQN", mode="Atari")
		replay_buffer = ReplayBuffer(params.memory_size)
		agent = DQN(Model_Atari, Model_Atari, env.action_space.n, params, logdirs.model_DQN)
		if params.policy_fn == "Eps":
			Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end,
										decay_steps=params.decay_steps)
			policy = EpsilonGreedyPolicy_eager(Epsilon_fn=Epsilon)
		elif params.policy_fn == "Boltzmann":
			policy = BoltzmannQPolicy_eager()
	else:
		print("Select 'mode' either 'Atari' or 'CartPole' !!")

	reward_buffer = deque(maxlen=params.reward_buffer_ep)
	summary_writer = tf.contrib.summary.create_file_writer(logdirs.log_DQN)
	train_DQN(agent, env, policy, replay_buffer, reward_buffer, params, summary_writer)
