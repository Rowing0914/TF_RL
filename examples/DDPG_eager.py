# Deep Deterministic Policy Gradient algorithm
# https://arxiv.org/pdf/1509.02971.pdf

import gym
import itertools
import numpy as np
import time
from collections import deque
from tf_rl.common.wrappers import MyWrapper
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.policy import BoltzmannQPolicy_eager, EpsilonGreedyPolicy_eager
from tf_rl.common.params import Parameters
from tf_rl.common.utils import AnnealingSchedule, logging
import tensorflow as tf

tf.enable_eager_execution()


class Actor(tf.keras.Model):
	def __init__(self, env_type, num_action):
		super(Actor, self).__init__()
		self.env_type = env_type
		if self.env_type == "CartPole":
			self.dense1 = tf.keras.layers.Dense(16, activation='relu')
			self.dense2 = tf.keras.layers.Dense(16, activation='relu')
			self.dense3 = tf.keras.layers.Dense(16, activation='relu')
			self.pred = tf.keras.layers.Dense(num_action, activation='softmax')
		elif self.env_type == "Atari":
			self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
			self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
			self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
			self.flat = tf.keras.layers.Flatten()
			self.fc1 = tf.keras.layers.Dense(512, activation='relu')
			self.pred = tf.keras.layers.Dense(num_action, activation='softmax')

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


class Critic(tf.keras.Model):
	def __init__(self, env_type, num_action):
		super(Critic, self).__init__()
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


class Actor_Critic:
	def __init__(self, env_type, actor, critic, num_action, params):
		self.params = params
		self.env_type = env_type
		self.num_action = num_action
		self.actor = actor(env_type, num_action)
		self.critic = critic(env_type, num_action)
		self.actor_optimizer = tf.train.AdamOptimizer()
		self.critic_optimizer = tf.train.AdamOptimizer()

	def predict(self, state):
		return self.actor(tf.convert_to_tensor(state[None, :], dtype=tf.float32)).numpy()[0]

	def update(self, states, actions, rewards, next_states, dones):
		"""

		Update Critic

		"""

		with tf.GradientTape() as tape:
			# calculate Q-values
			next_Q_actor = self.actor(tf.convert_to_tensor(next_states[None, :], dtype=tf.float32))
			next_Q_critic = self.critic(tf.convert_to_tensor(next_states[None, :], dtype=tf.float32))
			Q_critic = self.critic(tf.convert_to_tensor(states[None, :], dtype=tf.float32))

			# create indices for gathering action values according to Actor assessment
			idx_flattened = tf.range(0, tf.shape(next_Q_critic)[0]) * tf.shape(next_Q_critic)[1] + np.argmax(next_Q_actor, axis=-1)

			# same as Double DQN
			target_action_probs = tf.gather(tf.reshape(next_Q_critic, [-1]), idx_flattened)

			# get the q-values which is associated with actually taken actions in a game
			actions_one_hot = tf.one_hot(actions, self.num_action, 1.0, 0.0)
			action_probs = tf.reduce_sum(actions_one_hot * Q_critic, reduction_indices=-1)

			Y = reward + self.params.gamma*target_action_probs

			# MSE loss function: (1/N)*sum(Y - Q(s,a))^2
			critic_loss = tf.reduce_mean(tf.squared_difference(Y, action_probs))

		# get gradients
		critic_grads = tape.gradient(critic_loss, self.critic.trainable_weights)

		# apply processed gradients to the network
		self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

		"""

		Update Actor

		"""

		with tf.GradientTape() as tape:
			# compute q-values
			q_values = self.actor(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
			next_Q_critic = self.critic(tf.convert_to_tensor(next_states[None, :], dtype=tf.float32))

			Y = rewards + self.params.gamma * np.max(next_Q_critic, axis=-1).flatten() * np.logical_not(dones)

			# get the q-values which is associated with actually taken actions in a game
			actions_one_hot = tf.one_hot(actions, self.num_action, 1.0, 0.0)
			action_probs = tf.reduce_sum(actions_one_hot * q_values, reduction_indices=-1)

			actor_loss = tf.reduce_mean(tf.squared_difference(Y, action_probs))

		# get gradients
		actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)

		# apply processed gradients to the network
		self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_weights))


# env_type = "CartPole"
env = MyWrapper(gym.make("CartPole-v0"))
params = Parameters(algo="DQN", mode="CartPole")
agent = Actor_Critic("CartPole", Actor, Critic, env.action_space.n, params)
replay_buffer = ReplayBuffer(params.memory_size)
reward_buffer = deque(maxlen=params.reward_buffer_ep)

if params.policy_fn == "Eps":
	Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end,
								decay_steps=params.decay_steps)
	policy = EpsilonGreedyPolicy_eager(Epsilon_fn=Epsilon)
elif params.policy_fn == "Boltzmann":
	policy = BoltzmannQPolicy_eager()

global_timestep = 0

for i in range(params.num_episodes):
	state = env.reset()
	memory = list()
	total_reward = 0
	cnt_action = list()
	policy.index_episode = i
	start = time.time()

	# generate an episode
	for t in itertools.count():
		# env.render()
		action = policy.select_action(agent, state)
		next_state, reward, done, info = env.step(action)
		replay_buffer.add(state, action, reward, next_state, done)

		state = next_state
		total_reward += reward
		cnt_action.append(action)

		if done:
			if global_timestep > params.learning_start:
				states, actions, rewards, next_states, dones = replay_buffer.sample(params.batch_size)

				# update the networks according to the current episode
				agent.update(states, actions, rewards, next_states, dones)

				logging(global_timestep, params.num_frames, i, time.time() - start, total_reward, 0, policy.current_epsilon(), cnt_action)
				total_reward = 0
			break

		global_timestep += 1

	reward_buffer.append(total_reward)
	# stopping condition: if the agent has achieved the goal twice successively then we stop this!!
	if np.mean(reward_buffer) > params.goal:
		break

env.close()