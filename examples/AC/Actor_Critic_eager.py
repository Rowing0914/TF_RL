# one step Actor-Critic Algorithm

import gym, argparse
import itertools
import numpy as np
import time
from collections import deque
from tf_rl.common.wrappers import MyWrapper
import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement=True,
						intra_op_parallelism_threads=1,
						inter_op_parallelism_threads=1)
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
tf.random.set_random_seed(123)

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
	def __init__(self, env_type):
		super(Critic, self).__init__()
		self.env_type = env_type
		if self.env_type == "CartPole":
			self.dense1 = tf.keras.layers.Dense(16, activation='relu')
			self.dense2 = tf.keras.layers.Dense(16, activation='relu')
			self.dense3 = tf.keras.layers.Dense(16, activation='relu')
			self.pred = tf.keras.layers.Dense(1, activation='linear')
		elif self.env_type == "Atari":
			self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
			self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
			self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
			self.flat = tf.keras.layers.Flatten()
			self.fc1 = tf.keras.layers.Dense(512, activation='relu')
			self.pred = tf.keras.layers.Dense(1, activation='linear')

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
		self.critic = critic(env_type)
		self.actor_optimizer = tf.train.AdamOptimizer()
		self.critic_optimizer = tf.train.AdamOptimizer()

	def predict(self, state):
		# we take an action according to the action distribution produced by policy network
		return np.random.choice(np.arange(self.num_action), p=self.actor(tf.convert_to_tensor(state[None,:], dtype=tf.float32)).numpy()[0])

	def update(self, state, action, reward, next_state, done):
		"""

		Update Critic

		"""

		with tf.GradientTape() as tape:
			# calculate one-step advantage
			state_value = self.critic(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
			next_state_value = self.critic(tf.convert_to_tensor(next_state[None, :], dtype=tf.float32))
			advantage = reward + self.params.gamma*next_state_value - state_value

			# MSE loss function: (1/N)*sum(Advantage - V(s))^2
			critic_loss = tf.reduce_mean(tf.squared_difference(advantage, state_value))

		# get gradients
		critic_grads = tape.gradient(critic_loss, self.critic.trainable_weights)

		# apply processed gradients to the network
		self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

		"""

		Update Actor

		"""

		with tf.GradientTape() as tape:
			# compute action probability distirbution
			action_probs = self.actor(tf.convert_to_tensor(state[None, :], dtype=tf.float32))

			# get the probability according to the taken action in an episode
			actions_one_hot = tf.one_hot(action, self.num_action, 1.0, 0.0)
			action_probs = tf.reduce_sum(actions_one_hot * action_probs, reduction_indices=-1)

			# loss for policy network: TD_error * log p(a|s)
			actor_loss = -tf.log(action_probs) * advantage

		# get gradients
		actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)

		# apply processed gradients to the network
		self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_weights))

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="CartPole", help="game env type: Atari or CartPole")
parser.add_argument("--seed", default=123, help="seed of randomness")
parser.add_argument("--loss_fn", default="huber", help="types of loss function: MSE or huber")
parser.add_argument("--grad_clip_flg", default="", help="gradient clippings: by value(by_value) or global norm(norm) or nothing")
parser.add_argument("--num_episodes", default=500, type=int, help="total episodes in a training")
parser.add_argument("--train_interval", default=1, type=int, help="a frequency of training occurring in training phase")
parser.add_argument("--eval_interval", default=2500, type=int, help="a frequency of evaluation occurring in training phase") # temp
parser.add_argument("--memory_size", default=5000, type=int, help="memory size in a training")
parser.add_argument("--learning_start", default=100, type=int, help="frame number which specifies when to start updating the agent")
parser.add_argument("--sync_freq", default=1000, type=int, help="frequency of updating a target model")
parser.add_argument("--batch_size", default=32, type=int, help="batch size of each iteration of update")
parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.99, type=float, help="discount factor: gamma > 1.0 or negative => does not converge!!")
parser.add_argument("--tau", default=1e-2, type=float, help="soft update tau")
parser.add_argument("--ep_start", default=1.0, type=float, help="initial value of epsilon")
parser.add_argument("--ep_end", default=0.02, type=float, help="final value of epsilon")
parser.add_argument("--lr_start", default=0.0025, type=float, help="initial value of lr")
parser.add_argument("--lr_end", default=0.00025, type=float, help="final value of lr")
parser.add_argument("--decay_steps", default=3000, type=int, help="a period for annealing a value(epsilon or beta)")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.goal = 195

# env_type = "CartPole"
env = MyWrapper(gym.make("CartPole-v0"))

agent = Actor_Critic("CartPole", Actor, Critic, env.action_space.n, params)
reward_buffer = deque(maxlen=params.reward_buffer_ep)
global_timestep = 0

for i in range(params.num_episodes):
	state = env.reset()
	memory = list()
	total_reward = 0
	cnt_action = list()
	start = time.time()

	# generate an episode
	for t in itertools.count():
		# env.render()
		action = agent.predict(state)
		next_state, reward, done, info = env.step(action)

		# update the networks according to the current episode
		agent.update(state, action, reward, next_state, done)

		state = next_state
		total_reward += reward
		cnt_action.append(action)

		if done:
			# logging purpose
			reward_buffer.append(total_reward)

			print("Reward: {}".format(total_reward))
			total_reward = 0
			break

	# stopping condition: if the agent has achieved the goal twice successively then we stop this!!
	if np.mean(reward_buffer) > params.goal:
		break

env.close()