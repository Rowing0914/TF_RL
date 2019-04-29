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
import itertools
import numpy as np
import time
import tensorflow as tf
from copy import deepcopy
from collections import deque
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.params import Parameters
from tf_rl.common.utils import AnnealingSchedule, logging, soft_target_model_update_eager, huber_loss
from tf_rl.common.random_process import OrnsteinUhlenbeckProcess

tf.enable_eager_execution()


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
		self.actor = actor(env_type, num_action)
		self.critic = critic(env_type, num_action)
		self.target_actor  = deepcopy(self.actor)
		self.target_critic = deepcopy(self.critic)
		self.actor_optimizer = tf.train.AdamOptimizer()
		self.critic_optimizer = tf.train.AdamOptimizer()
		self.random_process = OrnsteinUhlenbeckProcess(size=self.num_action, theta=0.15, mu=0.0, sigma=0.2)

	def predict(self, state, epsilon):
		if np.random.rand(1) > epsilon:
			return self.actor(tf.convert_to_tensor(state[None, :], dtype=tf.float32)).numpy()[0]*self.random_process.sample()
		else:
			return np.random.uniform(-2.0, 2.0, 1)*self.random_process.sample()

	def update(self, states, actions, rewards, next_states, dones):
		"""

		Update Critic

		"""

		with tf.GradientTape() as tape:
			# calculate Q-values
			target_actions = self.target_actor(tf.convert_to_tensor(next_states[None, :], dtype=tf.float32))[0]
			# critic takes as input states, actions so that we combine them before passing them
			next_q_values = self.target_critic( tf.cast(tf.concat([next_states, target_actions], axis = -1), dtype=tf.float32) )
			q_values = self.critic( tf.cast(tf.concat([states, actions], axis = -1), dtype=tf.float32) )

			Y = rewards + self.params.gamma*next_q_values * np.logical_not(dones)

			# MSE loss function: (1/N)*sum(Y - Q(s,a))^2
			critic_loss = tf.reduce_mean(tf.squared_difference(Y, q_values))
			critic_loss = huber_loss(critic_loss)

		# get gradients
		critic_grads = tape.gradient(critic_loss, self.critic.trainable_weights)

		# apply processed gradients to the network
		self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

		if np.random.rand() > 0.4:
			# soft update
			soft_target_model_update_eager(self.target_critic, self.critic)

		"""

		Update Actor

		"""

		with tf.GradientTape() as tape:
			# compute q-values
			target_actions = self.actor(tf.convert_to_tensor(states[None, :], dtype=tf.float32))
			q_values = self.critic( tf.cast(tf.concat([states[None, :], target_actions], axis = -1), dtype=tf.float32) )

			actor_loss = tf.reduce_mean(-q_values)
			actor_loss = huber_loss(actor_loss)

		# get gradients
		actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)

		# apply processed gradients to the network
		self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_weights))

		if np.random.rand() > 0.4:
			# soft update
			soft_target_model_update_eager(self.target_actor, self.actor)

		return np.sum(critic_loss + actor_loss)


env = gym.make("Pendulum-v0")
# env = gym.make("MountainCarContinuous-v0")

params = Parameters(algo="DDPG", mode="CartPole")
agent = DDPG("CartPole", Actor, Critic, 1, params)
replay_buffer = ReplayBuffer(params.memory_size)
reward_buffer = deque(maxlen=params.reward_buffer_ep)
Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)

global_timestep = 0

for i in range(params.num_episodes):
	state = env.reset()
	memory = list()
	total_reward = 0
	start = time.time()
	agent.random_process.reset_states()
	action_list = list()

	# generate an episode
	for t in itertools.count():
		# if i > 200:
		# 	env.render()
		action = agent.predict(state, Epsilon.get_value(i))
		next_state, reward, done, info = env.step(action)
		replay_buffer.add(state, action, reward, next_state, done)

		state = next_state
		total_reward += reward
		action_list.append(action)

		if done or t > 200:
			if global_timestep > params.learning_start:
				states, actions, rewards, next_states, dones = replay_buffer.sample(params.batch_size)

				# update the networks according to the current episode
				loss = agent.update(states, actions, rewards, next_states, dones)

				logging(global_timestep, params.num_frames, i, time.time() - start, total_reward, loss, Epsilon.get_value(i), [np.mean(np.array(action_list))])
			total_reward = 0
			break

		global_timestep += 1

	reward_buffer.append(total_reward)
	# stopping condition: if the agent has achieved the goal twice successively then we stop this!!
	if np.mean(reward_buffer) > params.goal:
		break

env.close()