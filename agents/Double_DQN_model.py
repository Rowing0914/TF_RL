import numpy as np
import tensorflow as tf
from common.utils import sync_main_target, soft_target_model_update


def train_Double_DQN(main_model, target_model, env, replay_buffer, Epsilon, params):
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
	losses = []
	all_rewards = []
	episode_reward = 0

	with tf.Session() as sess:
		# initialise all variables used in the model
		sess.run(tf.global_variables_initializer())
		state = env.reset()
		for frame_idx in range(1, params.num_frames + 1):
			action = target_model.act(sess, state.reshape(params.state_reshape), Epsilon.get_value(frame_idx))

			next_state, reward, done, _ = env.step(action)
			replay_buffer.add(state, action, reward, next_state, done)

			state = next_state
			episode_reward += reward

			if done:
				state = env.reset()
				all_rewards.append(episode_reward)
				episode_reward = 0

				if frame_idx > params.learning_start and len(replay_buffer) > params.batch_size:
					states, actions, rewards, next_states, dones = replay_buffer.sample(params.batch_size)
					next_Q_main = main_model.predict(sess, next_states)
					next_Q = target_model.predict(sess, next_states)
					Y = rewards + params.gamma * next_Q[
						np.arange(params.batch_size), np.argmax(next_Q_main, axis=1)] * dones
					loss = main_model.update(sess, states, actions, Y)
					print("GAME OVER AT STEP: {0}, SCORE: {1}, LOSS: {2}".format(frame_idx, episode_reward, loss))
					losses.append(loss)

			if frame_idx > params.learning_start and frame_idx % params.sync_freq == 0:
				# soft update means we partially add the original weights of target model instead of completely
				# sharing the weights among main and target models
				if params.update_hard_or_soft == "hard":
					sync_main_target(sess, main_model, target_model)
				elif params.update_hard_or_soft == "soft":
					soft_target_model_update(sess, main_model, target_model, tau=params.soft_update_tau)

	return all_rewards, losses