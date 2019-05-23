import time
import numpy as np
import tensorflow as tf
from tf_rl.common.params import Parameters
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import AnnealingSchedule, soft_target_model_update_eager, logging
from tf_rl.common.policy import EpsilonGreedyPolicy_eager
from tf_rl.agents.DQN import DQN
from examples.DQN.DQN_eager_cartpole import Model

tf.enable_eager_execution()
tf.random.set_random_seed(123)

class Env:
	def __init__(self, size):
		self.size = size

	def reset(self):
		state = np.random.randint(2, size=self.size)
		goal = np.random.randint(2, size=self.size)

		while np.sum(state == goal) == self.size:
			goal = np.random.randint(2, size=self.size)
		self.goal = goal
		self.state = state
		return state

	def step(self, action):
		self.state[action] = 1 - self.state[action]
		return self.compute_reward(self.state, self.goal)

	def compute_reward(self, state, goal):
		if not self.check_success(state, goal):
			return state, -1, False
		else:
			return state, 0, True

	def check_success(self, state, goal):
		return np.sum(state == goal) == self.size


def strategy(n, k):
	"""
	Future Strategy
	randomly select k time-steps which come from the same episode and observed after it

	:param n:
	:param k:
	:return:
	"""
	if k > n:
		return np.random.choice(n, k, replace=True)
	else:
		return np.random.choice(n, k, replace=False)


if __name__ == '__main__':
	num_episodes = 1000
	num_steps = 50
	num_action = 5
	mode = "CartPole"

	replay_buffer = ReplayBuffer(100)
	env = Env(num_action)
	params = Parameters(algo="HER", mode=mode)
	Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end,
								decay_steps=params.decay_steps)
	policy = EpsilonGreedyPolicy_eager(Epsilon_fn=Epsilon)
	agent = DQN(mode, Model, Model, num_action, params, "temp")

	global_timestep = 0

	for i in range(num_episodes):
		state = env.reset()
		current_goal = env.goal
		policy.index_episode = i
		agent.index_episode = i

		episodes = list()
		total_reward = 0
		start = time.time()
		cnt_action = list()

		# play AN episode
		for t in range(num_steps):
			action = policy.select_action(agent, np.concatenate([state, current_goal]))
			next_state, reward, done = env.step(action)
			episodes.append((np.concatenate([state, current_goal]), action, reward, np.concatenate([next_state, current_goal]), done))

			total_reward += reward
			state = next_state
			cnt_action.append(action)
			global_timestep += 1

			# if the game has ended, then break
			if done:
				break

		# Replay THE episode step-by-step while choosing "k" time-steps at random to get another goal(next_state of selected time-step)
		for t in range(len(episodes)):
			s_and_g, a, r, ns_and_g, d = episodes[t] # unpack the trajectory
			for k in strategy(n=len(episodes), k=4): # "future" strategy
				new_goal = episodes[k][-2][:num_action] # find the new goal, which is the next_state of randomly selected state
				new_reward = env.compute_reward(s_and_g[:num_action], new_goal)[1] # find the new reward accordingly
				episodes.append((np.concatenate([s_and_g[:num_action], new_goal]), a, new_reward, np.concatenate([ns_and_g[:num_action], new_goal]), d))

		# put the constructed episode into Replay Memory
		# if you want, you can use Prioritised Experience Replay at this point!
		for data in episodes:
			replay_buffer.add(*data)

		# Update Loop
		for _ in range(10):
			states, actions, rewards, next_states, dones = replay_buffer.sample(params.batch_size)

			loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)

			if np.random.rand() > 0.5:
				agent.manager.save()
				if params.update_hard_or_soft == "hard":
					agent.target_model.set_weights(agent.main_model.get_weights())
				elif params.update_hard_or_soft == "soft":
					soft_target_model_update_eager(agent.target_model, agent.main_model, tau=params.soft_update_tau)

		logging(global_timestep, params.num_frames, i, time.time() - start, total_reward, np.mean(loss),
				policy.current_epsilon(), cnt_action)


	print("===== TEST: {} Episodes =====".format(10))
	score = 0
	for ep in range(10):
		state = env.reset()
		current_goal = env.goal
		for t in range(num_steps):
			action = policy.select_action(agent, np.concatenate([state, current_goal]))
			next_state, reward, done = env.step(action)

			if done:
				score += 1
				break

			state = next_state

	print("Success Rate: {}".format(float(score)/10.0))
