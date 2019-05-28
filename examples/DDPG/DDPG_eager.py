"""

Deep Deterministic Policy Gradient algorithm
URL: https://arxiv.org/pdf/1509.02971.pdf

Problem Setting: Pendulum
URL: https://gym.openai.com/envs/Pendulum-v0/
Description:
The inverted pendulum swingup problem is a classic problem in the control literature.
In this version of the problem, the pendulum starts in a random position,
and the goal is to swing it up so it stays upright.


"""

import gym
import argparse
import numpy as np
import time
import tensorflow as tf
from copy import deepcopy
from collections import deque
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import AnnealingSchedule, logger, soft_target_model_update_eager, test_Agent
from tf_rl.common.random_process import OrnsteinUhlenbeckProcess

tf.enable_eager_execution()
tf.random.set_random_seed(123)

class Actor(tf.keras.Model):
	def __init__(self, env_type, num_action):
		super(Actor, self).__init__()
		self.env_type = env_type
		if self.env_type == "CartPole":
			self.dense1 = tf.keras.layers.Dense(16, activation='relu')
			self.dense2 = tf.keras.layers.Dense(16, activation='relu')
			self.dense3 = tf.keras.layers.Dense(32, activation='relu')
			self.batch  = tf.keras.layers.BatchNormalization()
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
			x = self.batch(x)
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


class Critic(tf.keras.Model):
	def __init__(self, env_type, num_action):
		super(Critic, self).__init__()
		self.env_type = env_type
		if self.env_type == "CartPole":
			self.dense1 = tf.keras.layers.Dense(32, activation='relu')
			self.dense2 = tf.keras.layers.Dense(32, activation='relu')
			self.dense3 = tf.keras.layers.Dense(32, activation='relu')
			self.batch = tf.keras.layers.BatchNormalization()
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
			x = self.batch(x)
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


class DDPG:
	def __init__(self, env_type, actor, critic, num_action, params):
		self.params = params
		self.env_type = env_type
		self.num_action = num_action
		self.eval_flg = False
		self.index_timestep = 0
		self.actor = actor(env_type, num_action)
		self.critic = critic(env_type, num_action)
		self.target_actor  = deepcopy(self.actor)
		self.target_critic = deepcopy(self.critic)
		self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=10e-4)
		self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=10e-3)
		self.random_process = OrnsteinUhlenbeckProcess(size=self.num_action, theta=0.15, mu=0.0, sigma=0.2)

	def predict(self, state, epsilon=0.02):
		if np.random.rand(1) > epsilon:
			return self.actor(tf.convert_to_tensor(state[None, :], dtype=tf.float32)).numpy()[0]*self.random_process.sample()
		else:
			return np.random.uniform(-2.0, 2.0, 1)*self.random_process.sample()

	def update(self, states, actions, rewards, next_states, dones):
		"""

		Update Critic

		"""
		self.index_timestep = tf.train.get_global_step()
		with tf.GradientTape() as tape:
			# calculate Q-values
			target_actions = self.target_actor(tf.convert_to_tensor(next_states[None, :], dtype=tf.float32))[0]
			# critic takes as input states, actions so that we combine them before passing them
			next_Q = self.target_critic( tf.cast(tf.concat([next_states, target_actions], axis = -1), dtype=tf.float32) )
			q_values = self.critic( tf.cast(tf.concat([states, actions], axis = -1), dtype=tf.float32) )

			Y = tf.math.multiply(self.params.gamma, tf.math.reduce_max(next_Q, axis=-1))
			Y = tf.math.multiply(Y, (1. - tf.cast(dones, tf.float32)))
			Y = tf.math.add(tf.cast(rewards, tf.float32), Y)
			Y = tf.stop_gradient(tf.reshape(Y, (32,1)))

			# huber loss
			critic_loss = tf.losses.huber_loss(Y, q_values, reduction=tf.losses.Reduction.NONE)

		# get gradients
		critic_grads = tape.gradient(critic_loss, self.critic.trainable_weights)

		# apply processed gradients to the network
		self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

		"""

		Update Actor

		"""

		with tf.GradientTape() as tape:
			# compute action
			target_actions = self.actor(tf.convert_to_tensor(states[None, :], dtype=tf.float32))
			q_values = self.critic( tf.cast(tf.concat([states[None, :], target_actions], axis = -1), dtype=tf.float32) )

			actor_loss = tf.reduce_mean(-q_values*critic_loss)

		# get gradients
		actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)

		# apply processed gradients to the network
		self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_weights))

		return np.sum(critic_loss + actor_loss)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", default="Atari", help="game env type => Atari or CartPole")
	parser.add_argument("--env_name", default="Breakout", help="game title")
	parser.add_argument("--loss_fn", default="MSE", help="types of loss function => MSE or huber_loss")
	parser.add_argument("--grad_clip_flg", default="norm", help="types of a clipping method of gradients => by value(by_value) or global norm(norm) or None")
	parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
	parser.add_argument("--train_interval", default=1, type=int, help="a frequency of training occurring in training phase")
	parser.add_argument("--eval_interval", default=50_000, type=int, help="a frequency of evaluation occurring in training phase")
	parser.add_argument("--memory_size", default=100_000, type=int, help="memory size in a training => this used for Experience Replay Memory or Prioritised Experience Replay Memory")
	parser.add_argument("--learning_start", default=500, type=int, help="frame number which specifies when to start updating the agent")
	parser.add_argument("--sync_freq", default=1_000, type=int, help="frequency of updating a target model")
	parser.add_argument("--batch_size", default=32, type=int, help="batch size of each iteration of update")
	parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
	parser.add_argument("--gamma", default=0.99, type=float, help="discount factor => gamma > 1.0 or negative => does not converge!!")
	parser.add_argument("--update_hard_or_soft", default="hard", help="types of synchronisation method of target and main models => soft or hard update")
	parser.add_argument("--soft_update_tau", default=1e-3, type=float, help="in soft-update tau defines the ratio of main model remains and it seems 1e-2 is the optimal!")
	parser.add_argument("--epsilon_start", default=1.0, type=float, help="initial value of epsilon")
	parser.add_argument("--epsilon_end", default=0.02, type=float, help="final value of epsilon")
	parser.add_argument("--decay_steps", default=100_000, type=int, help="a period for annealing a value(epsilon or beta)")
	parser.add_argument("--decay_type", default="linear", help="types of annealing method => linear or curved")
	parser.add_argument("--log_dir", default="../../logs/logs/DDPG/", help="directory for log")
	parser.add_argument("--model_dir", default="../../logs/models/DDPG/", help="directory for trained model")
	parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
	parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
	params = parser.parse_args()
	params.goal = 0
	params.test_episodes = 10


	env = gym.make("Ant-v2")
	# env = gym.make("MountainCarContinuous-v0")

	agent = DDPG("CartPole", Actor, Critic, 1, params)
	replay_buffer = ReplayBuffer(params.memory_size)
	reward_buffer = deque(maxlen=params.reward_buffer_ep)
	Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)

	global_timestep = tf.train.get_or_create_global_step()
	time_buffer = list()
	log = logger(params)

	for i in range(2000):
		state = env.reset()
		total_reward = 0
		start = time.time()
		agent.random_process.reset_states()
		done = False
		while not done:
			action = agent.predict(state, Epsilon.get_value())
			next_state, reward, done, info = env.step(action)
			replay_buffer.add(state, action, reward, next_state, done)

			tf.assign(global_timestep, global_timestep.numpy() + 1, name='update_global_step')
			total_reward += reward
			state = next_state

			# for evaluation purpose
			if global_timestep.numpy() % params.eval_interval == 0:
				agent.eval_flg = True

			if (global_timestep.numpy() > params.learning_start) and (global_timestep.numpy() % params.train_interval == 0):
				states, actions, rewards, next_states, dones = replay_buffer.sample(params.batch_size)

				loss = agent.update(states, actions, rewards, next_states, dones)

			# synchronise the target and main models by hard or soft update
			if (global_timestep.numpy() > params.learning_start) and (global_timestep.numpy() % params.sync_freq == 0):
				soft_target_model_update_eager(agent.target_actor, agent.actor, tau=params.soft_update_tau)
				soft_target_model_update_eager(agent.target_critic, agent.critic, tau=params.soft_update_tau)

		"""
		===== After 1 Episode is Done =====
		"""

		tf.contrib.summary.scalar("reward", total_reward, step=i)
		tf.contrib.summary.scalar("exec time", time.time() - start, step=i)
		if i >= params.reward_buffer_ep:
			tf.contrib.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=i)

		# store the episode reward
		reward_buffer.append(total_reward)
		time_buffer.append(time.time() - start)

		if global_timestep.numpy() > params.learning_start and i % params.reward_buffer_ep == 0:
			log.logging(global_timestep.numpy(), i, np.sum(time_buffer), reward_buffer, np.mean(loss),0, [0])
			time_buffer = list()

		if agent.eval_flg:
			test_Agent(agent, env)
			agent.eval_flg = False

		# check the stopping condition
		if global_timestep.numpy() > params.num_frames:
			print("=== Training is Done ===")
			test_Agent(agent, env, n_trial=params.test_episodes)
			env.close()
			break