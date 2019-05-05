# TODO: fix the error, and get correct params
import gym
import numpy as np
import tensorflow as tf
from collections import deque

# ==== import below from my repo ====
from tf_rl.common.wrappers import MyWrapper   # just a wrapper to set a reward at the terminal state -1
from tf_rl.common.params import Parameters    # params for training
from tf_rl.common.memory import ReplayBuffer  # Experience Replay Buffer

tf.enable_eager_execution()

class Model(tf.keras.Model):
	def __init__(self, num_action):
		super(Model, self).__init__()
		self.dense1 = tf.keras.layers.Dense(16, activation='relu')
		self.dense2 = tf.keras.layers.Dense(16, activation='relu')
		self.dense3 = tf.keras.layers.Dense(16, activation='relu')
		self.pred = tf.keras.layers.Dense(num_action, activation='relu')

	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.dense2(x)
		x = self.dense3(x)
		pred = self.pred(x)
		return pred


class DQN:
	"""
    On policy DQN
    """
	def __init__(self, num_action):
		self.num_action = num_action
		self.model = Model(num_action)
		self.optimizer = tf.train.AdamOptimizer()

	def predict(self, state):
		return self.model(tf.convert_to_tensor(state[None, :], dtype=tf.float32)).numpy()[0]

	def update(self, state, action, next_state):
		# calculate target: R + gamma * max_a Q(s',a')
		next_Q = self.model(tf.convert_to_tensor(next_state[None, :], dtype=tf.float32))
		Y = rewards + params.gamma * tf.reduce_max(next_Q, axis=-1) * np.logical_not(dones)

		# calculate Q(s,a)
		q_values = self.model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
		actions_one_hot = tf.one_hot(action, self.num_action, 1.0, 0.0)
		action_probs = tf.reduce_sum(actions_one_hot * q_values, reduction_indices=-1)

		# Minibatch MSE => (1/batch_size) * (R + gamma * Q(s',a') - Q(s,a))^2
		loss = tf.reduce_mean(tf.squared_difference(Y, action_probs))
		return loss


if __name__ == '__main__':
	env = MyWrapper(gym.make("CartPole-v0"))
	replay_buffer = ReplayBuffer(5000)
	params = Parameters(mode="CartPole")
	reward_buffer = deque(maxlen=params.reward_buffer_ep)
	agent = DQN(env.action_space.n)

	for i in range(1000):
		state = env.reset()

		total_reward = 0
		for t in range(210):
			# env.render()
			action = np.argmax(agent.predict(state)) # behave greedily
			next_state, reward, done, info = env.step(action)
			replay_buffer.add(state, action, reward, next_state, done)

			total_reward += reward
			state = next_state

			if done:
				print("Episode {0} finished after {1} timesteps".format(i, t + 1))

				if i > 10:
					with tf.GradientTape() as tape:
						# make sure to fit all process to compute gradients within this Tape context!!
						states, actions, rewards, next_states, dones = replay_buffer.sample(params.batch_size)
						loss = agent.update(states, actions, next_states)

					grads = tape.gradient(loss, agent.model.trainable_weights)
					agent.optimizer.apply_gradients(zip(grads, agent.model.trainable_weights))
				break

		# store the episode reward
		reward_buffer.append(total_reward)

		# check the stopping condition
		if np.mean(reward_buffer) > 195:
			print("GAME OVER!!")
			break

	env.close()
