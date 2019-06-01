from collections import deque
import time, itertools
import tensorflow as tf
import numpy as np
from tf_rl.common.utils import soft_target_model_update_eager, logger, test_Agent, test_Agent_policy_gradient, her_strategy, state_unpacker, get_ready, test_Agent_HER

"""
===== Value Based Algorithm =====

TODO: think about incorporating PER's memory updating procedure into the model
so that, we can unify train_DQN and train_DQN_PER 
"""

def train_DQN(agent, env, policy, replay_buffer, reward_buffer, summary_writer):
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
	get_ready(agent.params)
	global_timestep = tf.train.get_or_create_global_step()
	time_buffer = list()
	log = logger(agent.params)
	with summary_writer.as_default():
		# for summary purpose, we put all codes in this context
		with tf.contrib.summary.always_record_summaries():

			for i in itertools.count():
				state = env.reset()
				total_reward = 0
				start = time.time()
				cnt_action = list()
				done = False
				while not done:
					action = policy.select_action(agent, state)
					next_state, reward, done, info = env.step(action)
					replay_buffer.add(state, action, reward, next_state, done)

					global_timestep.assign_add(1)
					total_reward += reward
					state = next_state
					cnt_action.append(action)

					# for evaluation purpose
					if global_timestep.numpy() % agent.params.eval_interval == 0:
						agent.eval_flg = True

					if (global_timestep.numpy() > agent.params.learning_start) and (global_timestep.numpy() % agent.params.train_interval == 0):
						states, actions, rewards, next_states, dones = replay_buffer.sample(agent.params.batch_size)

						loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)

					# synchronise the target and main models by hard or soft update
					if (global_timestep.numpy() > agent.params.learning_start) and (global_timestep.numpy() % agent.params.sync_freq == 0):
						agent.manager.save()
						agent.target_model.set_weights(agent.main_model.get_weights())

				"""
				===== After 1 Episode is Done =====
				"""

				tf.contrib.summary.scalar("reward", total_reward, step=i)
				tf.contrib.summary.scalar("exec time", time.time() - start, step=i)
				if i >= agent.params.reward_buffer_ep:
					tf.contrib.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=i)
				tf.contrib.summary.histogram("taken actions", cnt_action, step=i)

				# store the episode reward
				reward_buffer.append(total_reward)
				time_buffer.append(time.time() - start)

				if global_timestep.numpy() > agent.params.learning_start and i % agent.params.reward_buffer_ep == 0:
					log.logging(global_timestep.numpy(), i, np.sum(time_buffer), reward_buffer, np.mean(loss), policy.current_epsilon(), cnt_action)
					time_buffer = list()

				if agent.eval_flg:
					test_Agent(agent, env)
					agent.eval_flg = False

				# check the stopping condition
				if global_timestep.numpy() > agent.params.num_frames:
					print("=== Training is Done ===")
					test_Agent(agent, env, n_trial=agent.params.test_episodes)
					env.close()
					break


def train_DQN_PER(agent, env, policy, replay_buffer, reward_buffer, Beta, summary_writer):
	"""
	Training script for DQN with PER

	:param agent:
	:param env:
	:param policy:
	:param replay_buffer:
	:param reward_buffer:
	:param params:
	:param summary_writer:
	:return:
	"""
	get_ready(agent.params)
	global_timestep = tf.train.get_or_create_global_step()
	time_buffer = list()
	log = logger(agent.params)
	with summary_writer.as_default():
		# for summary purpose, we put all codes in this context
		with tf.contrib.summary.always_record_summaries():

			for i in itertools.count():
				state = env.reset()
				total_reward = 0
				start = time.time()
				cnt_action = list()
				done = False
				while not done:
					action = policy.select_action(agent, state)
					next_state, reward, done, info = env.step(action)
					replay_buffer.add(state, action, reward, next_state, done)

					global_timestep.assign_add(1)
					total_reward += reward
					state = next_state
					cnt_action.append(action)

					# for evaluation purpose
					if global_timestep.numpy() % agent.params.eval_interval == 0:
						agent.eval_flg = True

					if (global_timestep.numpy() > agent.params.learning_start) and (global_timestep.numpy() % agent.params.train_interval == 0):
						# PER returns: state, action, reward, next_state, done, weights(a weight for an episode), indices(indices for a batch of episode)
						states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(
							agent.params.batch_size, Beta.get_value())

						loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)

						# add noise to the priorities
						batch_loss = np.abs(batch_loss) + agent.params.prioritized_replay_noise

						# Update a prioritised replay buffer using a batch of losses associated with each timestep
						replay_buffer.update_priorities(indices, batch_loss)

					# synchronise the target and main models by hard or soft update
					if (global_timestep.numpy() > agent.params.learning_start) and (global_timestep.numpy() % agent.params.sync_freq == 0):
						agent.manager.save()
						if agent.params.update_hard_or_soft == "hard":
							agent.target_model.set_weights(agent.main_model.get_weights())
						elif agent.params.update_hard_or_soft == "soft":
							soft_target_model_update_eager(agent.target_model, agent.main_model, tau=agent.params.soft_update_tau)

				"""
				===== After 1 Episode is Done =====
				"""

				tf.contrib.summary.scalar("reward", total_reward, step=i)
				tf.contrib.summary.scalar("exec time", time.time() - start, step=i)
				if i >= agent.params.reward_buffer_ep:
					tf.contrib.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=i)
				tf.contrib.summary.histogram("taken actions", cnt_action, step=i)

				# store the episode reward
				reward_buffer.append(total_reward)
				time_buffer.append(time.time() - start)

				if global_timestep.numpy() > agent.params.learning_start and i % agent.params.reward_buffer_ep == 0:
					log.logging(global_timestep.numpy(), i, np.sum(time_buffer), reward_buffer, np.mean(loss), policy.current_epsilon(), cnt_action)
					time_buffer = list()

				if agent.eval_flg:
					test_Agent(agent, env)
					agent.eval_flg = False

				# check the stopping condition
				if global_timestep.numpy() > agent.params.num_frames:
					print("=== Training is Done ===")
					test_Agent(agent, env, n_trial=agent.params.test_episodes)
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
	get_ready(params)
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


					if (global_timestep > params.learning_start) and (global_timestep % params.train_interval == 0):
						states, actions, rewards, next_states, dones = replay_buffer.sample(params.batch_size)

						loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)

					# synchronise the target and main models by hard or soft update
					if (global_timestep > params.learning_start) and (global_timestep % params.sync_freq == 0):
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
								logging(global_timestep, params.num_frames, i, time.time() - start, total_reward, np.mean(loss), 0, cnt_action)
							except:
								pass

						break

				# check the stopping condition
				# if np.mean(reward_buffer) > params.goal or global_timestep.numpy() > params.num_frames:
				if global_timestep.numpy() > params.num_frames:
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
	get_ready(params)
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

===== Policy Based Algorithm =====

"""


# I have referred to https://github.com/laket/DDPG_Eager in terms of design of algorithm
# I know this is not precisely accurate to the original algo, but this works better than that... lol
def train_DDPG(agent, env, replay_buffer, reward_buffer, summary_writer):
	get_ready(agent.params)

	global_timestep = tf.train.get_or_create_global_step()
	time_buffer = deque(maxlen=agent.params.reward_buffer_ep)
	log = logger(agent.params)

	with summary_writer.as_default():
		# for summary purpose, we put all codes in this context
		with tf.contrib.summary.always_record_summaries():

			for i in itertools.count():
				state = env.reset()
				total_reward = 0
				start = time.time()
				agent.random_process.reset_states()
				done = False
				episode_len = 0
				while not done:
					# env.render()
					if global_timestep.numpy() < agent.params.learning_start:
						action = env.action_space.sample()
					else:
						action = agent.predict(state)
					# scale for execution in env (in DDPG, every action is clipped between [-1, 1] in agent.predict)
					next_state, reward, done, info = env.step(action * env.action_space.high)
					replay_buffer.add(state, action, reward, next_state, done)

					global_timestep.assign_add(1)
					episode_len += 1
					total_reward += reward
					state = next_state

					# for evaluation purpose
					if global_timestep.numpy() % agent.params.eval_interval == 0:
						agent.eval_flg = True

				"""
				===== After 1 Episode is Done =====
				"""

				# train the model at this point
				for t_train in range(episode_len): # in mujoco, this will be 1,000 iterations!
					states, actions, rewards, next_states, dones = replay_buffer.sample(agent.params.batch_size)
					loss = agent.update(states, actions, rewards, next_states, dones)
					soft_target_model_update_eager(agent.target_actor, agent.actor, tau=agent.params.soft_update_tau)
					soft_target_model_update_eager(agent.target_critic, agent.critic, tau=agent.params.soft_update_tau)

				tf.contrib.summary.scalar("reward", total_reward, step=i)
				tf.contrib.summary.scalar("exec time", time.time() - start, step=i)
				if i >= agent.params.reward_buffer_ep:
					tf.contrib.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=i)

				# store the episode reward
				reward_buffer.append(total_reward)
				time_buffer.append(time.time() - start)

				if global_timestep.numpy() > agent.params.learning_start and i % agent.params.reward_buffer_ep == 0:
					log.logging(global_timestep.numpy(), i, np.sum(time_buffer), reward_buffer, np.mean(loss), 0, [0])

				if agent.eval_flg:
					test_Agent_policy_gradient(agent, env)
					agent.eval_flg = False

				# check the stopping condition
				if global_timestep.numpy() > agent.params.num_frames:
					print("=== Training is Done ===")
					test_Agent_policy_gradient(agent, env, n_trial=agent.params.test_episodes)
					env.close()
					break


def train_SAC(agent, env, replay_buffer, reward_buffer, summary_writer):
	get_ready(agent.params)

	global_timestep = tf.train.get_or_create_global_step()
	log = logger(agent.params)
	time_buffer = deque(maxlen=agent.params.reward_buffer_ep)

	with summary_writer.as_default():
		# for summary purpose, we put all codes in this context
		with tf.contrib.summary.always_record_summaries():

			for i in itertools.count():
				state = env.reset()
				total_reward = 0
				start = time.time()
				agent.random_process.reset_states()
				done = False
				episode_len = 0
				while not done:
					# env.render()
					if global_timestep.numpy() < agent.params.learning_start:
						action = env.action_space.sample()
					else:
						action = agent.predict(state)
					# scale for execution in env (in DDPG, every action is clipped between [-1, 1] in agent.predict)
					next_state, reward, done, info = env.step(action * env.action_space.high)
					replay_buffer.add(state, action, reward, next_state, done)

					global_timestep.assign_add(1)
					episode_len += 1
					total_reward += reward
					state = next_state

					# for evaluation purpose
					if global_timestep.numpy() % agent.params.eval_interval == 0:
						agent.eval_flg = True

				"""
				===== After 1 Episode is Done =====
				"""

				# train the model at this point
				for t_train in range(episode_len): # in mujoco, this will be 1,000 iterations!
					states, actions, rewards, next_states, dones = replay_buffer.sample(agent.params.batch_size)
					loss = agent.update(states, actions, rewards, next_states, dones)
					soft_target_model_update_eager(agent.target_actor, agent.actor, tau=agent.params.soft_update_tau)
					soft_target_model_update_eager(agent.target_critic, agent.critic, tau=agent.params.soft_update_tau)

				tf.contrib.summary.scalar("reward", total_reward, step=i)
				tf.contrib.summary.scalar("exec time", time.time() - start, step=i)
				if i >= agent.params.reward_buffer_ep:
					tf.contrib.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=i)

				# store the episode reward
				reward_buffer.append(total_reward)

				if global_timestep.numpy() > agent.params.learning_start and i % agent.params.reward_buffer_ep == 0:
					log.logging(global_timestep.numpy(), i, time.time() - start, reward_buffer, np.mean(loss), 0, [0])

				if agent.eval_flg:
					test_Agent_policy_gradient(agent, env)
					agent.eval_flg = False

				# check the stopping condition
				if global_timestep.numpy() > agent.params.num_frames:
					print("=== Training is Done ===")
					test_Agent_policy_gradient(agent, env, n_trial=agent.params.test_episodes)
					env.close()
					break


def train_HER_bit(agent, env, policy, replay_buffer, summary_writer):
	get_ready(agent.params)

	global_timestep = tf.train.get_or_create_global_step()
	log = logger(agent.params)

	with summary_writer.as_default():
		# for summary purpose, we put all codes in this context
		with tf.contrib.summary.always_record_summaries():

			for i in itertools.count():
				state = env.reset()
				current_goal = env.goal
				total_reward = 0
				start = time.time()
				episodes = list()
				cnt_action = list()
				done = False
				while not done:
					action = policy.select_action(agent, np.concatenate([state, current_goal]))
					next_state, reward, done, info = env.step(action)
					episodes.append((np.concatenate([state, current_goal]), action, reward, np.concatenate([next_state, current_goal]), done))

					global_timestep.assign_add(1)
					total_reward += reward
					state = next_state
					cnt_action.append(action)

					# if the game has ended, then break
					if done:
						break

				# Replay THE episode step-by-step while choosing "k" time-steps at random to get another goal(next_state of selected time-step)
				for t in range(len(episodes)):
					s_and_g, a, r, ns_and_g, d = episodes[t] # unpack the trajectory
					for k in her_strategy(n=len(episodes), k=4): # "future" strategy
						new_goal = episodes[k][-2][:agent.params.bit_len] # find the new goal, which is the next_state of randomly selected state
						new_reward = env.compute_reward(s_and_g[:agent.params.bit_len], new_goal)[1] # find the new reward accordingly
						episodes.append((np.concatenate([s_and_g[:agent.params.bit_len], new_goal]), a, new_reward, np.concatenate([ns_and_g[:agent.params.bit_len], new_goal]), d))

				# put the constructed episode into Replay Memory
				# if you want, you can use Prioritised Experience Replay at this point!
				for data in episodes:
					replay_buffer.add(*data)

				# Update Loop
				for _ in range(10):
					states, actions, rewards, next_states, dones = replay_buffer.sample(agent.params.batch_size)

					loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)

					if np.random.rand() > 0.5:
						agent.manager.save()
						if agent.params.update_hard_or_soft == "hard":
							agent.target_model.set_weights(agent.main_model.get_weights())
						elif agent.params.update_hard_or_soft == "soft":
							soft_target_model_update_eager(agent.target_model, agent.main_model, tau=agent.params.soft_update_tau)

				log.logging(global_timestep.numpy(), agent.params.num_frames, time.time() - start, [total_reward], np.mean(loss),
						policy.current_epsilon(), cnt_action)

			test_Agent(agent, env)



def train_HER(agent, env, replay_buffer, reward_buffer, summary_writer):
	get_ready(agent.params)

	global_timestep = tf.train.get_or_create_global_step()
	total_ep = 0

	with summary_writer.as_default():
		# for summary purpose, we put all codes in this context
		with tf.contrib.summary.always_record_summaries():

			for epoch in range(agent.params.num_epochs):
				for cycle in range(agent.params.num_cycles):
					episodes = list()
					for ep in range(agent.params.num_episodes):
						state = env.reset()
						agent.random_process.reset_states()
						# obs, achieved_goal, desired_goal in `numpy.ndarray`
						obs, ag, dg = state_unpacker(state)
						total_reward = 0
						done = False
						while not done:
							# env.render()
							# in the paper, they used this stochastic behavioural policy
							if np.random.random() > 0.2:
								action = env.action_space.sample()
							else:
								action = agent.predict(np.concatenate([obs, ag], axis=-1))

							next_state, reward, done, info = env.step(action)
							# obs, achieved_goal, desired_goal in `numpy.ndarray`
							next_obs, next_ag, next_dg = state_unpacker(state)
							episodes.append((obs, ag, action, reward, next_obs, next_ag, done))

							global_timestep.assign_add(1)
							total_reward += reward
							state = next_state

							# for evaluation purpose
							if global_timestep.numpy() % agent.params.eval_interval == 0:
								agent.eval_flg = True

							# if the game has ended, then break
							if done:
								break

					"""
					===== Outside Episodes =====
					"""

					# Replay ONE episode step-by-step while choosing "k" time-steps at random to get another goal(next_state of selected time-step)
					short_memory = list()
					for t in range(len(episodes)):
						s, ag, a, r, ns, nag, d = episodes[t] # unpack the trajectory
						for k in her_strategy(n=len(episodes), k=4): # "future" strategy
							_s, _ag, _a, _r, _ns, _nag, _d = episodes[k] # unpack the trajectory
							new_goal = _nag
							new_reward = env.compute_reward(ag, _ag, "") # find the new reward given currently achieved goal
							_sg  = np.concatenate([s, new_goal], axis=-1)
							_nsg = np.concatenate([ns, new_goal], axis=-1)
							short_memory.append(( _sg, a, new_reward, _nsg, d))

					# put the constructed episode into Replay Memory
					# if you want, you can use Prioritised Experience Replay at this point!
					for data in short_memory:
						replay_buffer.add(*data)

					# Update Loop
					for _ in range(agent.params.num_updates):
						states, actions, rewards, next_states, dones = replay_buffer.sample(agent.params.batch_size)
						loss = agent.update(states, actions, rewards, next_states, dones)
						soft_target_model_update_eager(agent.target_actor, agent.actor, tau=agent.params.soft_update_tau)
						soft_target_model_update_eager(agent.target_critic, agent.critic, tau=agent.params.soft_update_tau)

					# store the episode reward
					reward_buffer.append(total_reward)
					total_ep += ep+1
					print("Epoch: {}/{} | Cycle: {}/{} | Ep: {} | MEAN R: {} | MAX R: {}".format(
						epoch+1, agent.params.num_epochs, cycle+1, agent.params.num_cycles, total_ep, np.mean(reward_buffer), np.max(reward_buffer)
					))

					if agent.eval_flg:
						test_Agent_HER(agent, env)
						agent.eval_flg = False

			print("=== Training is Done ===")
			test_Agent_HER(agent, env, n_trial=agent.params.test_episodes)
			env.close()