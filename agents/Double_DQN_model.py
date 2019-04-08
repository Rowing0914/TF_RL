import numpy as np
import time
import tensorflow as tf
from common.utils import sync_main_target, soft_target_model_update, logging


def train_Double_DQN(main_model, target_model, env, replay_buffer, policy, params):
	"""
	Train Double DQN agent

	:param main_model:
	:param target_model:
	:param env:
	:param replay_buffer:
	:param Epsilon:
	:param params:
	:return:
	"""

	# log purpose
	losses, all_rewards, cnt_action = [], [], []
	episode_reward, index_episode = 0, 0

	# Create a glboal step variable
	global_step = tf.Variable(0, name='global_step', trainable=False)

	with tf.Session() as sess:
		# initialise all variables used in the model
		sess.run(tf.global_variables_initializer())
		state = env.reset()
		start = time.time()
		global_step = sess.run(tf.train.get_global_step())
		for frame_idx in range(1, params.num_frames + 1):
			action = policy.select_action(sess, target_model, state.reshape(params.state_reshape))
			cnt_action.append(action)
			next_state, reward, done, _ = env.step(action)
			replay_buffer.add(state, action, reward, next_state, done)

			state = next_state
			episode_reward += reward
			global_step += 1

			if done:
				index_episode += 1
				state = env.reset()
				all_rewards.append(episode_reward)

				if frame_idx > params.learning_start and len(replay_buffer) > params.batch_size:
					states, actions, rewards, next_states, dones = replay_buffer.sample(params.batch_size)
					next_Q_main = main_model.predict(sess, next_states)
					next_Q = target_model.predict(sess, next_states)
					Y = rewards + params.gamma * next_Q[
						np.arange(params.batch_size), np.argmax(next_Q_main, axis=1)] * np.logical_not(dones)
					loss = main_model.update(sess, states, actions, Y)

					# Logging and refreshing log purpose values
					losses.append(loss)
					logging(frame_idx, params.num_frames, index_episode, time.time()-start, episode_reward, np.mean(loss), cnt_action)

					episode_summary = tf.Summary()
					episode_summary.value.add(simple_value=episode_reward, node_name="episode_reward",
											  tag="episode_reward")
					episode_summary.value.add(simple_value=index_episode, node_name="episode_length",
											  tag="episode_length")
					main_model.summary_writer.add_summary(episode_summary, global_step)

					episode_reward = 0
					cnt_action = []
					start = time.time()

			if frame_idx > params.learning_start and frame_idx % params.sync_freq == 0:
				# soft update means we partially add the original weights of target model instead of completely
				# sharing the weights among main and target models
				if params.update_hard_or_soft == "hard":
					sync_main_target(sess, main_model, target_model)
				elif params.update_hard_or_soft == "soft":
					soft_target_model_update(sess, main_model, target_model, tau=params.soft_update_tau)

	return all_rewards, losses