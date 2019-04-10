import gym
import os
import time
import itertools
import numpy as np
import tensorflow as tf
from collections import deque
from common.wrappers import MyWrapper
from common.params import Parameters
from common.memory import ReplayBuffer
from common.utils import AnnealingSchedule, soft_target_model_update_eager, logging, huber_loss, ClipIfNotNone
from common.policy import EpsilonGreedyPolicy_eager, BoltzmannQPolicy_eager
from common.visualise import plot_Q_values

tf.enable_eager_execution()

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

class DQN:
	"""
    DQN
    """
	def __init__(self, num_action, params):
		self.num_action = num_action
		self.params = params
		self.main_model = Model(num_action)
		self.target_model = Model(num_action)
		# self.optimizer = tf.train.AdamOptimizer()
		self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
		self.index_episode = 0

	def predict(self, state):
		return self.main_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32)).numpy()[0]

	def update(self, state, action, next_state):
		with tf.GradientTape() as tape:
			# make sure to fit all process to compute gradients within this Tape context!!

			# calculate target: R + gamma * max_a Q(s',a')
			next_Q = self.predict(next_state)
			Y = rewards + params.gamma * np.max(next_Q, axis=1).flatten() * np.logical_not(dones)

			# calculate Q(s,a)
			q_values = self.main_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))

			# check Q values distribution
			# for q in q_values.numpy()[0]:
			# 	plot_Q_values(q, ymax=1)

			actions_one_hot = tf.one_hot(action, self.num_action, 1.0, 0.0)
			action_probs = tf.reduce_sum(actions_one_hot * q_values, reduction_indices=-1)

			if self.params.loss_fn == "huber_loss":
				# use huber loss
				loss = huber_loss(tf.subtract(Y, action_probs))
			elif self.params.loss_fn == "MSE":
				# use MSE
				loss = tf.reduce_mean(tf.squared_difference(Y, action_probs))
			else:
				assert False

		grads = tape.gradient(loss, agent.main_model.trainable_weights)

		if self.params.grad_clip_flg == "by_value":
			grads = [ClipIfNotNone(grad, -1., 1.) for grad in grads]
		elif self.params.grad_clip_flg == "norm":
			grads, _ = tf.clip_by_global_norm(grads, 5.0)

		agent.optimizer.apply_gradients(zip(grads, agent.main_model.trainable_weights))


		for index, grad in enumerate(grads):
			tf.contrib.summary.histogram("layer_grad_{}".format(index), grad, step=self.index_episode)
		tf.contrib.summary.scalar("loss", loss, step=self.index_episode)
		tf.contrib.summary.histogram("next_Q", next_Q, step=self.index_episode)
		tf.contrib.summary.scalar("mean_q_value", tf.math.reduce_mean(next_Q), step=self.index_episode)
		tf.contrib.summary.scalar("var_q_value", tf.math.reduce_variance(next_Q), step=self.index_episode)
		tf.contrib.summary.scalar("max_q_value", tf.reduce_max(next_Q), step=self.index_episode)

		return loss


if __name__ == '__main__':
	try:
		os.system("rm -rf ../logs/summary_DQN_eager")
	except:
		pass

	reward_buffer = deque(maxlen=5)
	env = MyWrapper(gym.make("CartPole-v0"))
	replay_buffer = ReplayBuffer(5000)
	params = Parameters(mode="CartPole")
	agent = DQN(env.action_space.n, params)
	Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps, decay_type=params.decay_type)
	policy = EpsilonGreedyPolicy_eager(Epsilon)
	# policy = BoltzmannQPolicy_eager()

	summary_writer = tf.contrib.summary.create_file_writer("../logs/summary_DQN_eager")

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
							states, actions, rewards, next_states, dones = replay_buffer.sample(params.batch_size)

							loss = agent.update(states, actions, next_states)
							logging(global_timestep, params.num_frames, i, time.time() - start, total_reward, np.mean(loss),
									policy.current_epsilon(), cnt_action)

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
