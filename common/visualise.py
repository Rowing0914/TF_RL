import datetime
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def running_mean(x, n):
	# res = []
	# for i in range(len(x)):
	#     res.append( sum(x[max(i-n+1, 0): i+1])   /   min(i+1, n) )

	return [sum(x[max(i - n + 1, 0): i + 1]) / min(i + 1, n) for i in range(len(x))]

	return res


def plot_all(env, memory, trace, print_=False):
	"""
	Plot all visualisation

	TODO: i will put PCA or t-SNE to reduce the dimensionality of the state
		and make all visualisation available on any stataes

	:param env:
	:param memory:
	:param trace:
	:param print_:
	:return:
	"""
	st = trace.states[-1]
	eps = trace.epsilons[-1]

	if print_:
		last_ep_rew = trace.last_ep_reward
		reward_str = str(round(last_ep_rew, 3)) if last_ep_rew is not None else 'None'
		print(f'wall: {datetime.datetime.now().strftime("%H:%M:%S")}   '
			  f'ep: {len(trace.ep_rewards):3}   tstep: {trace.tstep:4}   '
			  f'total tstep: {trace.total_tstep:6}   '
			  f'eps: {eps:5.3f}   reward: {reward_str}')

	if len(st) == 2:
		# We are working with 2D environment,
		# plot whole Q-Value functions across whole state space
		plot_2d_environment(env, trace.total_tstep - 1,
							1000, trace, memory,
							axis_labels=['state[0]', 'state[1]'],
							action_labels=['Act 0', 'Act 1', 'Act 2'],
							action_colors=['red', 'blue', 'green'])
	else:
		# Environment is not 2D, so we can't plot whole Q-Value function
		# Instead we plot state on standard graph,
		# which is still better than nothing
		plot_generic_environment(env, trace.total_tstep, 1000, trace, memory)


def plot_generic_environment(trace):
	"""
	it will be deprecated soon

	:param trace:
	:return:
	"""

	# Plot test states
	fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=[16, 12])
	tmp_x = np.array(list(trace.q_values.keys()))
	if len(tmp_x) > 0:
		tmp_y_hat = np.array(list(trace.q_values.values()))
		tmp_y_hat = np.average(tmp_y_hat, axis=-1)  # average over actions
		lines = ax1.plot(tmp_x, tmp_y_hat, alpha=.5)
		ax1.grid()
	ax1.set_title('Q-Values')
	# ax1.set_xlabel('Time Step')
	ax1.set_ylabel('Q-Values')
	# plt.show()

	# fig, ax = plt.subplots(figsize=[16,4])
	plot_episode_rewards(trace.ep_rewards, ax2)
	ax2.grid()
	ax2.set_ylabel('Episode Reward')
	ax2.set_title('Episode Rewards')
	# plt.show()

	# fig, ax = plt.subplots(figsize=[16,4])
	states_tmp = np.array(trace.states[-1000:])
	tsteps_tmp = np.array(range(len(states_tmp))) + trace.total_tstep - trace.eval_every
	lines = ax3.plot(tsteps_tmp, states_tmp, alpha=.5)
	if trace.state_labels is not None:
		ax3.legend(lines, trace.state_labels)
	ax3.grid()
	ax3.set_title('Trajectory')
	ax3.set_xlabel('Time Step')
	ax3.set_ylabel('State Values')
	# plt.tight_layout()
	plt.show()


def plot_2d_environment(env, steps_to_plot, trace, mem,
						axis_labels, action_labels, action_colors):
	"""
	Plot all necessary visualisation

	1. 3D plot of Q max values
	2. Scatter plot of Trajectory
	3. Line graph of Q values over epsiodes
	4. Region map of policy filling with its values
	5. Scatter plot of observations in the replay memory
	6. Scatter plot of Epsisode rewards over Episodes with Trend line

	:param env:
	:param steps_to_plot:
	:param trace:
	:param mem:
	:param axis_labels:
	:param action_labels:
	:param action_colors:
	:return:
	"""

	# final Q-values
	q_arr = trace.q_values[list(trace.q_values.keys())[-1]]
	states = trace.states[-steps_to_plot:]
	actions = trace.actions[-steps_to_plot:]

	fig = plt.figure(figsize=[12, 8])

	ax = fig.add_subplot(231, projection='3d')
	plot_q_max_3d(q_arr, env, title='Q_Max', labels=axis_labels, alpha=.5, axis=ax)

	ax = fig.add_subplot(232)
	plot_trajectory(states, actions, env, title='Trajectory', labels=axis_labels, axis=ax)

	ax = fig.add_subplot(233)
	values_tmp = np.array(list(trace.q_values.values()))  # shape [n, 128, 128, act]
	values = values_tmp[:, values_tmp.shape[1] // 2, values_tmp.shape[2] // 2, :]
	for i in range(values.shape[-1]):
		ax.plot(values[:, i], color=action_colors[i])
	ax.set_title('Q Values')
	ax.grid()

	ax = fig.add_subplot(234)
	plot_policy(q_arr, env, labels=axis_labels,
				colors=action_colors, collab=action_labels, axis=ax)

	ax = fig.add_subplot(235)
	st, act, rew_1, st_1, dones_1, _ = mem.pick_last(len(mem))
	plot_trajectory(st, act, env, title='Memory Buffer', labels=axis_labels, alpha=0.5, axis=ax)

	ax = fig.add_subplot(236)
	plot_episode_rewards(trace.ep_rewards, ax)
	ax.grid()
	ax.set_xlabel('Time Step')
	ax.set_ylabel('Episode Reward')
	ax.set_title('Episode Rewards')

	plt.tight_layout()
	plt.show()


def plot_episode_rewards(episode_rewards_dict, axis=None):
	tsteps = []  # episodes end tsteps
	rewards = []  # episodes rewards

	for time_step, reward in episode_rewards_dict.items():
		tsteps.append(time_step)
		rewards.append(reward)

	rewards_avg = running_mean(rewards, 100)

	if axis is None:
		fig = plt.figure()
		axis = fig.add_subplot(111)

	if len(tsteps) > 0:
		axis.scatter(tsteps, rewards, alpha=1, s=1, label='Episode reward')
		axis.plot(tsteps, rewards_avg, alpha=1, color='orange', label='Avg. 100 episodes')


def plot_trajectory(states, actions, env, title, labels, alpha=1.0, axis=None):
	if not isinstance(states, np.ndarray): states = np.array(states)
	if not isinstance(actions, np.ndarray): actions = np.array(actions)

	colours = cm.rainbow(np.linspace(0, 1, num=env.action_space.n))

	for action, colour in zip(range(0, env.action_space.n + 1), colours):
		axis.scatter(states[actions == action, 0], states[actions == action, 1], marker='.', s=1, color=colour,
					 alpha=1., label="A_{}".format(action))

	x_min, x_max = env.observation_space.low[0], env.observation_space.high[0]
	y_min, y_max = env.observation_space.low[1], env.observation_space.high[1]
	axis.set_xticks(np.linspace(x_min, x_max, 5))
	axis.set_yticks(np.linspace(y_min, y_max, 5))

	axis.set_xlabel('x')
	axis.set_ylabel('y')
	axis.set_title('Policy')
	axis.legend()


def plot_policy(q_arr, env, labels, colors, collab, axis=None):
	q_pol = np.argmax(q_arr, axis=-1)

	cmap = ListedColormap(colors)

	if axis is None:
		fig = plt.figure()
		axis = fig.add_subplot(111)

	heatmap = axis.pcolormesh(q_pol.T, cmap=cmap)
	axis.set_aspect('equal', 'datalim')
	cbar = plt.colorbar(heatmap)
	cbar.set_ticks(range(len(collab)))
	cbar.set_ticklabels(collab)

	x_min, x_max = env.observation_space.low[0], env.observation_space.high[0]
	y_min, y_max = env.observation_space.low[1], env.observation_space.high[1]
	axis.set_xticks([0, q_arr.shape[0]])
	axis.set_xticklabels([x_min, x_max])
	axis.set_yticks([0, q_arr.shape[1]])
	axis.set_yticklabels([y_min, y_max])

	axis.set_xlabel(labels[0])
	axis.set_ylabel(labels[1])
	axis.set_title('Policy')


def plot_q_max_3d(q_arr, env, color='#1f77b4', alpha=1., title='', labels=['x', 'y'], axis=None):
	"""

	Plot the final Q-values over states space

	we can see what kind state has high Q-value

	"""
	q_max = np.max(q_arr, axis=-1)  # calculate the maximum value w.r.t the most right feature in Q values

	x_min, x_max = env.observation_space.low[0], env.observation_space.high[0]
	y_min, y_max = env.observation_space.low[1], env.observation_space.high[1]
	x_space = np.linspace(x_min, x_max, num=q_max.shape[0])
	y_space = np.linspace(y_min, y_max, num=q_max.shape[1])
	Y, X = np.meshgrid(y_space, x_space)

	axis.plot_surface(X, Y, q_max, cmap=cm.coolwarm, alpha=1.)
	axis.set_xlabel('x')
	axis.set_ylabel('y')
	axis.set_xticks(np.linspace(x_min, x_max, 5))
	axis.set_yticks(np.linspace(y_min, y_max, 5))
	axis.set_title('Q max')
	axis.view_init(40, -70)


def eval_state_action_space(model, env, split=[32, 32]):
	"""Evaluate 2d Q-function on area and return as 3d array

	Params:
		model     - function approximator with method: model.eval(state, action) -> float
		env       - environment with members:
					  st_low - state space low boundry e.g. [-1.2, -0.07]
					  st_high - state space high boundry
					  act_space - action space, e.g. [0, 1, 2]
		split     - number of data points in each dimensions, e.g. [20, 20]
	"""
	# prep states to evaluate
	x_min, x_max = env.observation_space.low[0], env.observation_space.high[0]
	y_min, y_max = env.observation_space.low[1], env.observation_space.high[1]
	x_split, y_split = split
	x_space = np.linspace(x_min, x_max, x_split)
	y_space = np.linspace(y_min, y_max, y_split)
	Y, X = np.meshgrid(y_space, x_space)
	states = np.stack([X, Y], axis=-1)
	states = states.reshape([-1, 2])

	q_arr = model.eval(states=states)
	return q_arr.reshape([128, 128, -1])
