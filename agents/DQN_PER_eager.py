import gym
import argparse
import os
import time
import itertools
import numpy as np
import tensorflow as tf
from collections import deque
from common.wrappers import MyWrapper, wrap_deepmind, make_atari
from common.params import Parameters
from common.memory import PrioritizedReplayBuffer
from common.utils import AnnealingSchedule, soft_target_model_update_eager, logging, huber_loss, ClipIfNotNone
from common.policy import EpsilonGreedyPolicy_eager, BoltzmannQPolicy_eager
from common.visualise import plot_Q_values

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


class DQN_PER:
	"""
    DQN with PER
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
			next_Q = self.target_model(tf.convert_to_tensor(next_states, dtype=tf.float32))
			Y = rewards + self.params.gamma * np.max(next_Q, axis=-1).flatten() * np.logical_not(dones)

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

	logdir = "../logs/summary_DQN_PER_eager"
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
		agent = DQN_PER(Model_CartPole, Model_CartPole, env.action_space.n, params)
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
		agent = DQN_PER(Model_Atari, Model_Atari, env.action_space.n, params)
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

	with summary_writer.as_default():
		# for summary purpose, we put all codes in this context
		with tf.contrib.summary.always_record_summaries():

			global_timestep = 0
			for i in range(4000):
				state = env.reset()
				total_reward = 0
				start = time.time()
				cnt_action = list()
				policy.index_episode = i
				agent.index_episode = i
				for t in itertools.count():
					# env.render()
					action = policy.select_action(agent, state)
					next_state, reward, done, info = env.step(action)
					replay_buffer.add(state, action, reward, next_state, done)

					total_reward += reward
					state = next_state
					cnt_action.append(action)

					if done:
						tf.contrib.summary.scalar("reward", total_reward, step=global_timestep)

						if global_timestep > params.learning_start:
							# PER returns: state, action, reward, next_state, done, weights(a weight for an episode), indices(indices for a batch of episode)
							states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(
								params.batch_size, Beta.get_value(i))

							loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)
							logging(global_timestep, params.num_frames, i, time.time() - start, total_reward, np.mean(loss),
									policy.current_epsilon(), cnt_action)

							# add noise to the priorities
							batch_loss = np.abs(batch_loss) + params.prioritized_replay_noise

							# Update a prioritised replay buffer using a batch of losses associated with each timestep
							replay_buffer.update_priorities(indices, batch_loss)

							if np.random.rand() > 0.5:
								if params.update_hard_or_soft == "hard":
									agent.target_model.set_weights(agent.main_model.get_weights())
								elif params.update_hard_or_soft == "soft":
									soft_target_model_update_eager(agent.target_model, agent.main_model, tau=params.soft_update_tau)
						break

					global_timestep += 1

				# store the episode reward
				reward_buffer.append(total_reward)
				# check the stopping condition
				if np.mean(reward_buffer) > 195:
					print("GAME OVER!!")
					break

	env.close()
