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
			for i in itertools.count():
				state = env.reset()
				total_reward = 0
				start = time.time()
				cnt_action = list()
				policy.index_episode = i
				agent.index_episode = i
				for t in itertools.count():
					# if i > 100:
					# 	env.render()
					action = policy.select_action(agent, state)
					next_state, reward, done, info = env.step(action)
					replay_buffer.add(state, action, reward, next_state, done)

					global_timestep += 1
					total_reward += reward
					state = next_state
					cnt_action.append(action)

					if (global_timestep > params.learning_start) and (global_timestep % params.train_interval == 0):
						states, actions, rewards, next_states, dones = replay_buffer.sample(params.batch_size)

						loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)

						# synchronise the target and main models by hard or soft update
						if global_timestep % params.sync_freq == 0:
							agent.manager.save()
							if params.update_hard_or_soft == "hard":
								agent.target_model.set_weights(agent.main_model.get_weights())
							elif params.update_hard_or_soft == "soft":
								soft_target_model_update_eager(agent.target_model, agent.main_model,
															   tau=params.soft_update_tau)

					if done:
						tf.contrib.summary.scalar("reward", total_reward, step=i)
						# store the episode reward
						reward_buffer.append(total_reward)

						if global_timestep > params.learning_start:
							try:
								logging(global_timestep, params.num_frames, i, time.time() - start, total_reward, np.mean(loss), policy.current_epsilon(), cnt_action)
							except:
								pass

						break

				# check the stopping condition
				if np.mean(reward_buffer) > params.goal or global_timestep > params.num_frames:
					print("GAME OVER!!")
					env.close()
					break




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
			for i in itertools.count():
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
					global_timestep += 1

					if (global_timestep > params.learning_start) and (global_timestep % params.train_interval == 0):
						# PER returns: state, action, reward, next_state, done, weights(a weight for an episode), indices(indices for a batch of episode)
						states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(
							params.batch_size, Beta.get_value(i))

						loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)

						# add noise to the priorities
						batch_loss = np.abs(batch_loss) + params.prioritized_replay_noise

						# Update a prioritised replay buffer using a batch of losses associated with each timestep
						replay_buffer.update_priorities(indices, batch_loss)

						if global_timestep % params.sync_freq == 0:
							agent.manager.save()
							if params.update_hard_or_soft == "hard":
								agent.target_model.set_weights(agent.main_model.get_weights())
							elif params.update_hard_or_soft == "soft":
								soft_target_model_update_eager(agent.target_model, agent.main_model,
															   tau=params.soft_update_tau)

					if done:
						tf.contrib.summary.scalar("reward", total_reward, step=i)
						# store the episode reward
						reward_buffer.append(total_reward)

						if global_timestep > params.learning_start:
							try:
								logging(global_timestep, params.num_frames, i, time.time() - start, total_reward, np.mean(loss), policy.current_epsilon(), cnt_action)
							except:
								pass

						break

				# check the stopping condition
				if np.mean(reward_buffer) > params.goal or global_timestep > params.num_frames:
					print("GAME OVER!!")
					env.close()
					break


def train_DQN_afp(agent, expert, env, agent_policy, expert_policy, replay_buffer, reward_buffer, params, summary_writer):
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
			for i in itertools.count():
				state = env.reset()
				total_reward = 0
				start = time.time()
				cnt_action = list()
				agent_policy.index_episode = i
				agent.index_episode = i
				for t in itertools.count():
					# env.render()
					action = agent_policy.select_action(agent, state)

					# where the AFP comes in
					# if learning agent is not sure about his decision, then he asks for expert's help
					if action <= 0.5:
						action = expert_policy.select_action(expert, state)

					next_state, reward, done, info = env.step(action)
					replay_buffer.add(state, action, reward, next_state, done)

					total_reward += reward
					state = next_state
					cnt_action.append(action)
					global_timestep += 1

					if global_timestep > params.learning_start:
						states, actions, rewards, next_states, dones = replay_buffer.sample(params.batch_size)

						loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)
						logging(global_timestep, params.num_frames, i, time.time() - start, total_reward, np.mean(loss), 0, cnt_action)

						if np.random.rand() > 0.5:
							agent.manager.save()
							if params.update_hard_or_soft == "hard":
								agent.target_model.set_weights(agent.main_model.get_weights())
							elif params.update_hard_or_soft == "soft":
								soft_target_model_update_eager(agent.target_model, agent.main_model, tau=params.soft_update_tau)

					if done:
						tf.contrib.summary.scalar("reward", total_reward, step=global_timestep)
						reward_buffer.append(total_reward)
						break

				# check the stopping condition
				if np.mean(reward_buffer) > params.goal:
					print("GAME OVER!!")
					env.close()
					break


def train_DRQN(agent, env, policy, replay_buffer, reward_buffer, params, summary_writer):
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
			for i in itertools.count():
				state = env.reset()
				total_reward = 0
				start = time.time()
				cnt_action = list()
				policy.index_episode = i
				agent.index_episode = i
				episode_memory = list()
				for t in itertools.count():
					# env.render()
					action = policy.select_action(agent, state.reshape(1, 4))
					next_state, reward, done, info = env.step(action)
					episode_memory.append((state, action, reward, next_state, done))

					total_reward += reward
					state = next_state
					cnt_action.append(action)
					global_timestep += 1

					if global_timestep > params.learning_start:
						states, actions, rewards, next_states, dones = replay_buffer.sample(params.batch_size)
						_states, _actions, _rewards, _next_states, _dones = [], [], [], [], []
						for index, data in enumerate(zip(states, actions, rewards, next_states, dones)):
							s1, a, r, s2, d = data
							ep_start = np.random.randint(0, len(s1) + 1 - 4)
							# states[i] = s1[ep_start:ep_start+4, :]
							# actions[i] = a[ep_start:ep_start+4]
							# rewards[i] = r[ep_start:ep_start+4]
							# next_states[i] = s2[ep_start:ep_start+4, :]
							# dones[i] = d[ep_start:ep_start+4]
							_states.append(s1[ep_start:ep_start + 4, :])
							_actions.append(a[ep_start:ep_start + 4])
							_rewards.append(r[ep_start:ep_start + 4])
							_next_states.append(s2[ep_start:ep_start + 4, :])
							_dones.append(d[ep_start:ep_start + 4])

						_states, _actions, _rewards, _next_states, _dones = np.array(_states), np.array(
							_actions), np.array(_rewards), np.array(_next_states), np.array(_dones)

						# loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)
						loss, batch_loss = agent.update(_states, _actions, _rewards, _next_states, _dones)
						logging(global_timestep, params.num_frames, i, time.time() - start, total_reward, np.mean(loss),
								policy.current_epsilon(), cnt_action)

						if np.random.rand() > 0.5:
							agent.manager.save()
							if params.update_hard_or_soft == "hard":
								agent.target_model.set_weights(agent.main_model.get_weights())
							elif params.update_hard_or_soft == "soft":
								soft_target_model_update_eager(agent.target_model, agent.main_model,
															   tau=params.soft_update_tau)

					if done:
						tf.contrib.summary.scalar("reward", total_reward, step=global_timestep)
						reward_buffer.append(total_reward)

						s1, a, r, s2, d = [], [], [], [], []
						for data in episode_memory:
							s1.append(data[0])
							a.append(data[1])
							r.append(data[2])
							s2.append(data[3])
							d.append(data[4])

						replay_buffer.add(s1, a, r, s2, d)
						break

				# check the stopping condition
				if np.mean(reward_buffer) > params.goal:
					print("GAME OVER!!")
					env.close()
					break


"""

Test Methods

"""

def test_Agent(agent, env, policy):
	"""
	Test the agent with a visual aid!

	:return:
	"""
	state = env.reset()
	done = False
	episode_reward = 0

	# TODO: properly implement the q-values visualisation tool
	# xmax = 2
	# xmin = -1
	# ymax = np.amax(self.Q) + 30
	# ymin = 0

	while not done:
		env.render()
		action = policy.select_action(agent, state)
		# plot_Q_values(self.Q[current_state], xmin, xmax, ymin, ymax)
		# print(self.Q[current_state])
		next_state, reward, done, _ = env.step(action)
		state = next_state
		episode_reward += reward
	print("Game Over with score: {0}".format(episode_reward))
	env.close()
	return

