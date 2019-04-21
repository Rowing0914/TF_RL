import gym
import argparse
import os
import tensorflow as tf
from collections import deque
from tf_rl.common.wrappers import MyWrapper, wrap_deepmind, make_atari
from tf_rl.common.params import Parameters, logdirs
from tf_rl.common.memory import PrioritizedReplayBuffer
from tf_rl.common.utils import AnnealingSchedule
from tf_rl.common.policy import EpsilonGreedyPolicy_eager, BoltzmannQPolicy_eager
from tf_rl.common.train import train_DQN_PER
from tf_rl.agents.Double_DQN import Double_DQN

tf.enable_eager_execution()

class Model(tf.keras.Model):
	def __init__(self, env_type, num_action, duelling_type="avg"):
		super(Model, self).__init__()
		self.env_type = env_type
		self.duelling_type = duelling_type
		if self.env_type == "CartPole":
			self.dense1 = tf.keras.layers.Dense(16, activation='relu')
			self.dense2 = tf.keras.layers.Dense(16, activation='relu')
			self.dense3 = tf.keras.layers.Dense(16, activation='relu')
			self.q_value = tf.keras.layers.Dense(num_action, activation='linear')
			self.v_value = tf.keras.layers.Dense(1, activation='linear')
		elif self.env_type == "Atari":
			self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
			self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
			self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
			self.flat = tf.keras.layers.Flatten()
			self.fc1 = tf.keras.layers.Dense(512, activation='relu')
			self.q_value = tf.keras.layers.Dense(num_action, activation='linear')
			self.v_value = tf.keras.layers.Dense(1, activation='linear')

	def call(self, inputs):
		if self.env_type == "CartPole":
			x = self.dense1(inputs)
			x = self.dense2(x)
			x = self.dense3(x)
			q_value = self.q_value(x)
			v_value = self.v_value(x)
		elif self.env_type == "Atari":
			x = self.conv1(inputs)
			x = self.conv2(x)
			x = self.conv3(x)
			x = self.flat(x)
			x = self.fc1(x)
			q_value = self.q_value(x)
			v_value = self.v_value(x)


		if self.duelling_type == "avg":
			# Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
			output = tf.math.add(v_value, tf.math.subtract(q_value, tf.reduce_mean(q_value)))
		elif self.duelling_type == "max":
			# Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
			output = tf.math.add(v_value, tf.math.subtract(q_value, tf.math.reduce_max(q_value)))
		elif self.duelling_type == "naive":
			# Q(s,a;theta) = V(s;theta) + A(s,a;theta)
			output = tf.math.add(v_value, q_value)
		else:
			assert False, "dueling_type must be one of {'avg','max','naive'}"
		return output


if __name__ == '__main__':
	logdirs = logdirs()
	try:
		os.system("rm -rf {}".format(logdirs.log_Duelling_Double_DQN_PER))
	except:
		pass

	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", default="CartPole", help="game env type")
	parser.add_argument("--num_episodes", default=100, type=int, help="game env type")
	args = parser.parse_args()

	if args.mode == "CartPole":
		env = MyWrapper(gym.make("CartPole-v0"))
	elif args.mode == "Atari":
		env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))

	params = Parameters(algo="DQN_PER", mode=args.mode)
	params.num_episodes = args.num_episodes
	replay_buffer = PrioritizedReplayBuffer(params.memory_size, alpha=params.prioritized_replay_alpha)
	Beta = AnnealingSchedule(start=params.prioritized_replay_beta_start, end=params.prioritized_replay_beta_end,
							 decay_steps=params.decay_steps)
	agent = Double_DQN(args.mode, Model, Model, env.action_space.n, params, logdirs.model_DQN)
	if params.policy_fn == "Eps":
		Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end,
									decay_steps=params.decay_steps)
		policy = EpsilonGreedyPolicy_eager(Epsilon_fn=Epsilon)
	elif params.policy_fn == "Boltzmann":
		policy = BoltzmannQPolicy_eager()

	reward_buffer = deque(maxlen=params.reward_buffer_ep)
	summary_writer = tf.contrib.summary.create_file_writer(logdirs.log_Duelling_Double_DQN_PER)
	train_DQN_PER(agent, env, policy, replay_buffer, reward_buffer, params, Beta, summary_writer)