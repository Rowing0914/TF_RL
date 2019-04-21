# Algorithm: on page 333 of the Sutton's RL book
# Design pattern: followed Dennybritz => https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20REINFORCE%20with%20Baseline%20Solution.ipynb

import gym
import itertools
import numpy as np
from collections import deque
from tf_rl.common.wrappers import MyWrapper
from tf_rl.common.policy import BoltzmannQPolicy
from tf_rl.common.params import Parameters
import tensorflow as tf

class Policy_Network:
	def __init__(self, num_action, scope="policy_net"):
		with tf.variable_scope(scope):
			self.num_action = num_action
			self.state = tf.placeholder(shape=[None, 4], dtype=tf.float32, name="X")
			self.Y = tf.placeholder(shape=[None], dtype=tf.float32, name="Y")
			self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")

			x = tf.keras.layers.Dense(16, activation=tf.nn.relu)(self.state)
			x = tf.keras.layers.Dense(16, activation=tf.nn.relu)(x)
			x = tf.keras.layers.Dense(16, activation=tf.nn.relu)(x)
			self.pred = tf.keras.layers.Dense(self.num_action, activation=tf.nn.relu)(x)

			self.action_probs = tf.squeeze(tf.nn.softmax(self.pred))
			self.picked_action_prob = tf.gather(self.action_probs, self.action)

			# Loss and train op
			self.loss = -tf.log(self.picked_action_prob) * self.Y

			self.optimizer = tf.train.AdamOptimizer()
			self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

	def predict(self, sess, state):
		return sess.run(self.pred, feed_dict={self.state: state})[0]

	def update(self, sess, state, action, target):
		_, loss = sess.run([self.train_op, self.loss], feed_dict={self.state: state, self.action: action, self.Y: target})
		return loss


class Value_Network:
	def __init__(self, scope="value_net"):
		with tf.variable_scope(scope):
			self.state = tf.placeholder(shape=[None, 4], dtype=tf.float32, name="X")
			self.Y = tf.placeholder(shape=[None], dtype=tf.float32, name="Y")

			x = tf.keras.layers.Dense(16, activation=tf.nn.relu)(self.state)
			x = tf.keras.layers.Dense(16, activation=tf.nn.relu)(x)
			x = tf.keras.layers.Dense(16, activation=tf.nn.relu)(x)
			self.pred = tf.keras.layers.Dense(1, activation=tf.nn.relu)(x)

			# use MSE
			self.losses = tf.squared_difference(self.Y, self.pred)
			self.loss = tf.reduce_mean(self.losses)

			self.optimizer = tf.train.AdamOptimizer()
			self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

	def predict(self, sess, state):
		return sess.run(self.pred, feed_dict={self.state: state})[0]

	def update(self, sess, state, target):
		_, loss = sess.run([self.train_op, self.loss], feed_dict={self.state: state, self.Y: target})
		return loss


env = MyWrapper(gym.make("CartPole-v0"))
policy_net = Policy_Network(num_action=env.action_space.n)
value_net = Value_Network()
params = Parameters(algo="DQN", mode="CartPole")
reward_buffer = deque(maxlen=params.reward_buffer_ep)

# please use Boltzmann policy instead!!
policy = BoltzmannQPolicy()

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

episode_reward, total_reward = [], 0

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for i in range(400):
		state = env.reset()
		memory = list()
		policy.index_episode = i

		# generate an episode
		for t in itertools.count():
			# env.render()
			action = policy.select_action(sess, policy_net, state.reshape(params.state_reshape))
			next_state, reward, done, info = env.step(action)
			memory.append([state, action, next_state, reward, done])
			if done:
				print("Episode {} finished after {} timesteps".format(i, t+1))
				break
			state = next_state
			total_reward += reward


		# logging purpose
		episode_reward.append(total_reward)
		reward_buffer.append(total_reward)
		total_reward = 0

		# after an episode, we update networks(Policy and Value)
		for step, data in enumerate(memory):
			state, action, next_state, reward, done = data

			# calculate discounted G
			total_return = sum(params.gamma**i * t[3] for i, t in enumerate(memory[step:]))

			# calculate an advantage
			advantage = total_return - value_net.predict(sess, state.reshape(params.state_reshape))

			# update models
			value_net.update(sess, state.reshape(params.state_reshape), advantage)
			policy_net.update(sess, state.reshape(params.state_reshape), [action], advantage)

		# stopping condition: if the agent has achieved the goal twice successively then we stop this!!
		if np.mean(reward_buffer) > params.goal:
			break

	env.close()

np.save("../logs/value/rewards_REINFORCE.npy", episode_reward)