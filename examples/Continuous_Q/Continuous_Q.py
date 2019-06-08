import gym, argparse
import numpy as np
import itertools
import tensorflow as tf
from tf_rl.common.utils import AnnealingSchedule, eager_setup
from tf_rl.common.filters import Particle_Filter
from tf_rl.common.wrappers import MyWrapper_revertable

eager_setup()

class Model(tf.keras.Model):
	def __init__(self, num_action):
		super(Model, self).__init__()
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

class Continuous_Q_Agent:
	def __init__(self, env, params, policy_type="Eps"):
		self.env = env
		self.num_action = 1
		self.model = Model(num_action=self.num_action)
		self.params = params
		self.policy_type = policy_type
		self.optimizer = tf.train.AdamOptimizer()

	def estimate_Q(self, state, epsilon):
		if (np.random.random() <= epsilon):
			return self.env.action_space.sample()
		else:
			return self.model(tf.convert_to_tensor(state[None,:], dtype=tf.float32)).numpy()[0]

	def update(self, state, action, reward, next_state, done):
		with tf.GradientTape() as tape:
			# make sure to fit all process to compute gradients within this Tape context!!

			# calculate target: R + gamma * max_a' Q(s', a')
			next_Q = self.model(tf.convert_to_tensor(next_state[None,:], dtype=tf.float32))
			Y = reward + self.params.gamma * np.max(next_Q, axis=-1).flatten() * np.logical_not(done)

			# calculate Q(s,a)
			q_values = self.model(tf.convert_to_tensor(state[None,:], dtype=tf.float32))

			# use MSE
			batch_loss = tf.squared_difference(Y, q_values)
			loss = tf.reduce_mean(batch_loss)

		# get gradients
		grads = tape.gradient(loss, self.model.trainable_weights)

		# apply processed gradients to the network
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

		return loss, batch_loss


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", default="CartPole", help="game env type: Atari or CartPole")
	parser.add_argument("--seed", default=123, help="seed of randomness")
	parser.add_argument("--loss_fn", default="huber", help="types of loss function: MSE or huber")
	parser.add_argument("--grad_clip_flg", default="",
						help="gradient clippings: by value(by_value) or global norm(norm) or nothing")
	parser.add_argument("--num_episodes", default=500, type=int, help="total episodes in a training")
	parser.add_argument("--train_interval", default=1, type=int,
						help="a frequency of training occurring in training phase")
	parser.add_argument("--eval_interval", default=2500, type=int,
						help="a frequency of evaluation occurring in training phase")  # temp
	parser.add_argument("--memory_size", default=5000, type=int, help="memory size in a training")
	parser.add_argument("--learning_start", default=100, type=int,
						help="frame number which specifies when to start updating the agent")
	parser.add_argument("--sync_freq", default=1000, type=int, help="frequency of updating a target model")
	parser.add_argument("--batch_size", default=32, type=int, help="batch size of each iteration of update")
	parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
	parser.add_argument("--gamma", default=0.99, type=float,
						help="discount factor: gamma > 1.0 or negative => does not converge!!")
	parser.add_argument("--tau", default=1e-2, type=float, help="soft update tau")
	parser.add_argument("--ep_start", default=1.0, type=float, help="initial value of epsilon")
	parser.add_argument("--ep_end", default=0.02, type=float, help="final value of epsilon")
	parser.add_argument("--lr_start", default=0.0025, type=float, help="initial value of lr")
	parser.add_argument("--lr_end", default=0.00025, type=float, help="final value of lr")
	parser.add_argument("--decay_steps", default=3000, type=int, help="a period for annealing a value(epsilon or beta)")
	parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
	parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
	params = parser.parse_args()

	env = MyWrapper_revertable(gym.make('MountainCarContinuous-v0'))

	# hyperparameters
	all_rewards = list()
	global_timestep = tf.train.get_or_create_global_step()
	anneal_ep = tf.train.polynomial_decay(params.ep_start, global_timestep, params.decay_steps, params.ep_end)
	agent = Continuous_Q_Agent(env, params)
	pf = Particle_Filter(N=10,type="uniform")
	global_step = 0

	for episode in range(params.num_episodes):
		state = env.reset()
		episode_loss = 0
		total_reward = 0

		for t in itertools.count():
			# estimate
			mean, var = pf.estimate()
			action = np.random.normal(mean, var, 1)

			global_timestep.assign_add(1)

			if episode > 100:
				env.render()

			# predict and update particles
			pf.predict(env, action)
			q_values = agent.estimate_Q(state, anneal_ep().numpy())
			pf.update(q_values=q_values)
			pf.simple_resample()

			next_state, reward, done, _ = env.step(action)
			loss, batch_loss = agent.update(state, action, reward, next_state, done)

			episode_loss += loss
			total_reward += reward
			state = next_state
			global_step += 1

			if t >= 300 or done:
				print("Reward: {}".format(total_reward))
				break
