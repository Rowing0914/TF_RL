import time, itertools
import tensorflow as tf
import numpy as np
from tf_rl.common.utils import soft_target_model_update_eager, logging

"""
TODO: think about incorporating PER's memory updating procedure into the model
so that, we can unify train_DQN and train_DQN_PER 
"""

def train_DQN(agent, env, policy, replay_buffer, reward_buffer, params, summary_writer):
	"""
	Training script for DQN and other advanced models without PER

	:param agent:
	:param env:
	:param policy:
	:param replay_buffer:
	:param reward_buffer:
	:param params:
	:param summary_writer:
	:return:
	"""
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

							loss = agent.update(states, actions, rewards, next_states, dones)
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


def train_DQN_PER(agent, env, policy, replay_buffer, reward_buffer, params, Beta, summary_writer):
	"""
	Training scripts for DQN with PER

	:param agent:
	:param env:
	:param policy:
	:param replay_buffer:
	:param reward_buffer:
	:param params:
	:param summary_writer:
	:return:
	"""
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
				if np.mean(reward_buffer) > 195:
					print("GAME OVER!!")
					break

	env.close()
