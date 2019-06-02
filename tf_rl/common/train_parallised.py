import itertools, os, ray
import tensorflow as tf
import numpy as np
from tf_rl.common.utils import get_ready, her_strategy, test_Agent_HER, state_unpacker, soft_target_model_update_eager

@ray.remote
class Env:
	def __init__(self, env):
		os.environ["MKL_NUM_THREADS"] = "1"
		self.env = env

	def one_episode(self, agent, global_timestep):
		episodes = list()
		state = self.env.reset()
		agent.random_process.reset_states()
		# obs, achieved_goal, desired_goal in `numpy.ndarray`
		obs, ag, dg = state_unpacker(state)
		total_reward = 0
		for t in itertools.count():
			# env.render()
			# in the paper, they used this stochastic behavioural policy
			if np.random.random() > 0.2:
				action = self.env.action_space.sample()
			else:
				action = agent.predict(np.concatenate([obs, dg], axis=-1))

			next_state, reward, done, info = self.env.step(action)
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
		return episodes


def train_HER(agent, env, replay_buffer, reward_buffer, summary_writer):
	ray.init()
	get_ready(agent.params)

	global_timestep = tf.train.get_or_create_global_step()
	total_ep = 0
	num_workers = 4

	envs = [Env.remote(env) for _ in range(num_workers)]

	with summary_writer.as_default():
		# for summary purpose, we put all codes in this context
		with tf.contrib.summary.always_record_summaries():

			for epoch in range(agent.params.num_epochs):
				for cycle in range(agent.params.num_cycles):
					episodes = list()
					# agent_id = ray.put(agent)
					for ep in range(int(agent.params.num_episodes/num_workers)):
						actions = list()
						for i in range(num_workers):
							# action_id = envs[i].one_episode(agent_id, global_timestep)
							action_id = envs[i].one_episode.remote(agent, global_timestep)
							actions.append(action_id)
						for i in range(num_workers):
							action_id, remaining_ids = ray.wait(actions)
							episode = ray.get(action_id)
							episodes.append(episode)
					print(episodes)
					asdf

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