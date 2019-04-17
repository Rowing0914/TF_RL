import gym
import argparse
import os
import time
import itertools
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tf_rl.common.wrappers import MyWrapper, wrap_deepmind, make_atari
from examples.params import Parameters, logdirs
from examples.DQN_eager import Model_CartPole as Model_CartPole_DQN, Model_Atari as Model_Atari_DQN
from tf_rl.common.memory import PrioritizedReplayBuffer
from tf_rl.common.utils import AnnealingSchedule, soft_target_model_update_eager, logging, huber_loss, ClipIfNotNone
from tf_rl.common.policy import EpsilonGreedyPolicy_eager, BoltzmannQPolicy_eager, TestPolicy
from tf_rl.agents.DQN import DQN


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


def getting_demo():
	pass

class DQfD:
	"""
    DQfD
    """
	def __init__(self, main_model, target_model, num_action, params, checkpoint_dir):
		self.num_action = num_action
		self.params = params
		self.main_model = main_model(num_action)
		self.target_model = target_model(num_action)
		self.optimizer = tf.train.AdamOptimizer()
		# self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
		self.index_episode = 0

		# TF: checkpoint vs Saver => https://stackoverflow.com/questions/53569622/difference-between-tf-train-checkpoint-and-tf-train-saver
		self.checkpoint_dir = checkpoint_dir
		self.check_point = tf.train.Checkpoint(optimizer=self.optimizer,
											   model=self.main_model,
											   optimizer_step=tf.train.get_or_create_global_step())
		self.manager = tf.train.CheckpointManager(self.check_point, checkpoint_dir, max_to_keep=3)

	def predict(self, state):
		return self.main_model(tf.convert_to_tensor(state[None,:], dtype=tf.float32)).numpy()[0]

	def update(self, states, actions_e, actions_l, rewards, next_states, dones):
		with tf.GradientTape() as tape:
			# make sure to fit all process to compute gradients within this Tape context!!

			one_step_loss = self._one_step_loss(states, actions_e, rewards, next_states, dones)

			n_step_loss = self._n_step_loss()

			large_margin_clf_loss = self._large_margin_clf_loss(actions_e, actions_l)

			l2_loss = self._l2_loss()

			# combined_loss = one_step_loss + lambda_1*n_step_loss + lambda_2*large_margin_clf_loss + lambda_3*l2_loss
			combined_loss = one_step_loss + 1.0*n_step_loss + 1.0*large_margin_clf_loss + (10**(-5))*l2_loss

			loss = huber_loss(combined_loss)

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
		# for index, grad in enumerate(grads):
		# 	tf.contrib.summary.histogram("layer_grad_{}".format(index), grad, step=self.index_episode)
		# tf.contrib.summary.scalar("loss", loss, step=self.index_episode)
		# tf.contrib.summary.histogram("next_Q", next_Q, step=self.index_episode)
		# tf.contrib.summary.scalar("mean_q_value", tf.math.reduce_mean(next_Q), step=self.index_episode)
		# tf.contrib.summary.scalar("var_q_value", tf.math.reduce_variance(next_Q), step=self.index_episode)
		# tf.contrib.summary.scalar("max_q_value", tf.reduce_max(next_Q), step=self.index_episode)

		return loss, 0

	def _one_step_loss(self, states, actions_e, rewards, next_states, dones):
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
		actions_one_hot = tf.one_hot(actions_e, self.num_action, 1.0, 0.0)
		action_probs = tf.reduce_sum(actions_one_hot * q_values, reduction_indices=-1)

		return tf.subtract(Y, action_probs) # one step TD-error

	def _n_step_loss(self):
		return 0

	def _large_margin_clf_loss(self, a_e, a_l):
		"""
		Logic is formed as below

		if a_e == a_l:
			return 0
		else:
			return 0.8

		:param a_e:
		:param a_l:
		:return:
		"""
		result = (a_e != a_l).astype(int)
		return result * 0.8

	def _l2_loss(self):
		vars = self.main_model.get_weights()
		lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 10e-5 # in paper, they used the fixed amount
		return lossL2


if __name__ == '__main__':

	logdirs = logdirs()

	try:
		os.system("rm -rf {}".format(logdirs.log_DQfD))
	except:
		pass

	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", default="CartPole", help="game env type")
	args = parser.parse_args()

	if args.mode == "CartPole":
		env = MyWrapper(gym.make("CartPole-v0"))
		params = Parameters(mode="CartPole")
		replay_buffer = PrioritizedReplayBuffer(params.memory_size, alpha=params.prioritized_replay_alpha)
		agent = DQfD(Model_CartPole, Model_CartPole, env.action_space.n, params, logdirs.model_DQfD)
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
		agent = DQfD(Model_Atari, Model_Atari, env.action_space.n, params, logdirs.model_DQfD)
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

	reward_buffer = deque(maxlen=params.reward_buffer_ep)
	summary_writer = tf.contrib.summary.create_file_writer(logdirs.log_DQfD)




	"""
	
	Populating the memory with demonstrations
	
	"""

	expert = DQN(Model_CartPole_DQN, Model_CartPole_DQN, env.action_space.n, params, logdirs.model_DQN)
	expert_policy = TestPolicy()
	expert.check_point.restore(expert.manager.latest_checkpoint)
	print("Restore the model from disk")

	batch_experience = list()
	batches = list()

	print("Pupulating a memory with demonstrations")
	for _ in range(5):
		state = env.reset()
		done = False
		episode_reward = 0

		while not done:
			action_e = expert_policy.select_action(expert, state)
			action_l = policy.select_action(agent, state)

			next_state, reward, done, _ = env.step(action_e)
			batch_experience.append([state, action_e, action_l, reward, next_state, done])
			state = next_state
			episode_reward += reward

			if len(batch_experience) == 10:
				batches.append(batch_experience)
				batch_experience = list()

		print("Game Over with score: {0}".format(episode_reward))

	print(len(batches))




	"""
	
	Pre-train the agent with collected demonstrations
	
	"""

	for i in range(10):
		sample = random.sample(batches, 1)

		states, actions_e, actions_l, rewards, next_states, dones = [], [], [], [], [], []

		for row in sample[0]:
			states.append(row[0])
			actions_e.append(row[1])
			actions_l.append(row[2])
			rewards.append(row[3])
			next_states.append(row[4])
			dones.append(row[5])
		states, actions_e, actions_l, rewards, next_states, dones = np.array(states), np.array(actions_e), np.array(actions_l), np.array(rewards), np.array(next_states), np.array(dones)
		agent.update(states, actions_e, actions_l, rewards, next_states, dones)

	print("to be implemented soon!!")
	error()



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
				if np.mean(reward_buffer) > params.goal:
					print("GAME OVER!!")
					break

	env.close()
