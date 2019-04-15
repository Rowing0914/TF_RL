import gym
import argparse
import os
import numpy as np
import tensorflow as tf
from collections import deque
from tf_rl.common.wrappers import MyWrapper, wrap_deepmind, make_atari
from tf_rl.common.params import Parameters
from tf_rl.common.memory import PrioritizedReplayBuffer
from tf_rl.common.utils import AnnealingSchedule, huber_loss, ClipIfNotNone
from tf_rl.common.policy import EpsilonGreedyPolicy_eager, BoltzmannQPolicy_eager
from tf_rl.common.train import train_DQN_PER

tf.enable_eager_execution()

class Model_CartPole(tf.keras.Model):
	def __init__(self, num_action, duelling_type="avg"):
		super(Model_CartPole, self).__init__()
		self.duelling_type = duelling_type
		self.dense1 = tf.keras.layers.Dense(16, activation='relu')
		self.dense2 = tf.keras.layers.Dense(16, activation='relu')
		self.dense3 = tf.keras.layers.Dense(16, activation='relu')
		self.q_value = tf.keras.layers.Dense(num_action, activation='linear')
		self.v_value = tf.keras.layers.Dense(1, activation='linear')

	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.dense2(x)
		x = self.dense3(x)
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


class Model_Atari(tf.keras.Model):
	def __init__(self, num_action, duelling_type="avg"):
		super(Model_Atari, self).__init__()
		self.duelling_type = duelling_type
		self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
		self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
		self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
		self.flat = tf.keras.layers.Flatten()
		self.fc1 = tf.keras.layers.Dense(512, activation='relu')
		self.q_value = tf.keras.layers.Dense(num_action, activation='linear')
		self.v_value = tf.keras.layers.Dense(1, activation='linear')


	def call(self, inputs):
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


class Duelling_Double_DQN_PER:
	"""
    Duelling_Double_DQN_PER
    """
	def __init__(self, main_model, target_model, num_action, params):
		self.num_action = num_action
		self.params = params
		self.main_model = main_model(num_action)
		self.target_model = target_model(num_action)
		self.optimizer = tf.train.AdamOptimizer()
		# self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
		self.index_episode = 0

	def predict(self, state):
		return self.main_model(tf.convert_to_tensor(state[None,:], dtype=tf.float32)).numpy()[0]

	def update(self, states, actions, rewards, next_states, dones):
		with tf.GradientTape() as tape:
			# make sure to fit all process to compute gradients within this Tape context!!

			# calculate target: R + gamma * max_a Q(s',a')
			next_Q_main = self.main_model(tf.convert_to_tensor(next_states, dtype=tf.float32))
			next_Q = self.target_model(tf.convert_to_tensor(next_states, dtype=tf.float32))
			idx_flattened = tf.range(0, tf.shape(next_Q)[0]) * tf.shape(next_Q)[1] + np.argmax(next_Q_main, axis=-1)

			# passing [-1] to tf.reshape means flatten the array
			# using tf.gather, associate Q-values with the executed actions
			action_probs = tf.gather(tf.reshape(next_Q, [-1]), idx_flattened)

			Y = rewards + self.params.gamma * action_probs * np.logical_not(dones)

			# calculate Q(s,a)
			q_values = self.main_model(tf.convert_to_tensor(states, dtype=tf.float32))

			# get the q-values which is associated with actually taken actions in a game
			actions_one_hot = tf.one_hot(actions, self.num_action, 1.0, 0.0)
			action_probs = tf.reduce_sum(actions_one_hot * q_values, reduction_indices=-1)

			if self.params.loss_fn == "huber_loss":
				# use huber loss
				loss = huber_loss(tf.subtract(Y, action_probs))
				batch_loss = loss
			elif self.params.loss_fn == "MSE":
				# use MSE
				batch_loss = tf.squared_difference(Y, action_probs)
				loss = tf.reduce_mean(batch_loss)
			else:
				assert False

		# get gradients
		grads = tape.gradient(loss, self.main_model.trainable_weights)

		# clip gradients
		if self.params.grad_clip_flg == "by_value":
			grads = [ClipIfNotNone(grad, -1., 1.) for grad in grads]
		elif self.params.grad_clip_flg == "norm":
			grads, _ = tf.clip_by_global_norm(grads, 5.0)

		# apply processed gradients to the network
		self.optimizer.apply_gradients(zip(grads, self.main_model.trainable_weights))

		# for log purpose
		for index, grad in enumerate(grads):
			tf.contrib.summary.histogram("layer_grad_{}".format(index), grad, step=self.index_episode)
		tf.contrib.summary.scalar("loss", loss, step=self.index_episode)
		tf.contrib.summary.histogram("next_Q", next_Q, step=self.index_episode)
		tf.contrib.summary.scalar("mean_q_value", tf.math.reduce_mean(next_Q), step=self.index_episode)
		tf.contrib.summary.scalar("var_q_value", tf.math.reduce_variance(next_Q), step=self.index_episode)
		tf.contrib.summary.scalar("max_q_value", tf.reduce_max(next_Q), step=self.index_episode)

		return loss, batch_loss

if __name__ == '__main__':

	logdir = "../logs/summary_Duelling_Double_DQN_PER_eager"
	try:
		os.system("rm -rf {}".format(logdir))
	except:
		pass

	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", default="CartPole", help="game env type")
	args = parser.parse_args()

	if args.mode == "CartPole":
		env = MyWrapper(gym.make("CartPole-v0"))
		params = Parameters(mode="CartPole")
		replay_buffer = PrioritizedReplayBuffer(params.memory_size, alpha=params.prioritized_replay_alpha)
		agent = Duelling_Double_DQN_PER(Model_CartPole, Model_CartPole, env.action_space.n, params)
		Beta = AnnealingSchedule(start=params.prioritized_replay_beta_start, end=params.prioritized_replay_beta_end,
								 decay_steps=params.decay_steps)
		if params.policy_fn == "Eps":
			Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end,
										decay_steps=params.decay_steps)
			policy = EpsilonGreedyPolicy_eager(Epsilon_fn=Epsilon)
		elif params.policy_fn == "Boltzmann":
			policy = BoltzmannQPolicy_eager()
	elif args.mode == "Atari":
		env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
		params = Parameters(mode="Atari")
		replay_buffer = PrioritizedReplayBuffer(params.memory_size, alpha=params.prioritized_replay_alpha)
		agent = Duelling_Double_DQN_PER(Model_Atari, Model_Atari, env.action_space.n, params)
		Beta = AnnealingSchedule(start=params.prioritized_replay_beta_start, end=params.prioritized_replay_beta_end,
								 decay_steps=params.decay_steps)
		if params.policy_fn == "Eps":
			Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end,
										decay_steps=params.decay_steps)
			policy = EpsilonGreedyPolicy_eager(Epsilon_fn=Epsilon)
		elif params.policy_fn == "Boltzmann":
			policy = BoltzmannQPolicy_eager()
	else:
		print("Select 'mode' either 'Atari' or 'CartPole' !!")

	reward_buffer = deque(maxlen=5)
	summary_writer = tf.contrib.summary.create_file_writer(logdir)
	train_DQN_PER(agent, env, policy, replay_buffer, reward_buffer, params, Beta, summary_writer)