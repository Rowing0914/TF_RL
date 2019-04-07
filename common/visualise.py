import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


class Plotting:
	def __init__(self, nb_actions, env, model, tracker):
		self.nb_actions = nb_actions
		self.env = env # game env
		self.model = model # model/agent
		self.tracker = tracker # tracking method to contain all important values arise in training phase
		self.nb_plots = 6 # number of currently available visualisation methods
		self.window_size = 1000 # window size of rolling mean

		self.x_min, self.x_max = self.env.observation_space.low[0], self.env.observation_space.high[0]
		self.y_min, self.y_max = self.env.observation_space.low[1], self.env.observation_space.high[1]

		# associating colours with each action in the env
		self.colous = plt.get_cmap('jet', self.nb_actions)
		self.colous.set_under('gray')

		fig, ax = plt.subplots()
		self.Q_max_3D_plot(ax)
		self.State_spcace_2D_scatter(ax)
		self.Q_values_line_graph(ax)
		self.Policy_reggion_map(ax)
		self.Rewards_with_trend(ax)
		self.Loss_line_graph(ax)


	def Q_max_3D_plot(self, ax):
		"""

		Plot the final Q-values over states space
		we can see what kind state has high Q-value

		"""

		q_max = np.max(self.tracker.q_values, axis=-1)  # calculate the maximum value w.r.t the most right feature in Q values

		x_space = np.linspace(self.x_min, self.x_max, num=q_max.shape[0])
		y_space = np.linspace(self.y_min, self.y_max, num=q_max.shape[1])
		Y, X = np.meshgrid(y_space, x_space)

		ax.plot_surface(X, Y, q_max, cmap=cm.coolwarm, alpha=1.)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_xticks(np.linspace(self.x_min, self.x_max, 5))
		ax.set_yticks(np.linspace(self.y_min, self.y_max, 5))
		ax.set_title('Q max')
		ax.view_init(40, -70)

	def State_spcace_2D_scatter(self, ax):
		states, actions = self.tracker.states, self.tracker.actions

		for action, colour in zip(range(0, self.nb_actions + 1), self.colours):
			ax.scatter(states[actions == action, 0], states[actions == action, 1], marker='.', s=1, color=colour,
						 alpha=1., label="A_{}".format(action))

		ax.set_xticks(np.linspace(self.x_min, self.x_max, 5))
		ax.set_yticks(np.linspace(self.y_min, self.y_max, 5))
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_title('State_spcace_2D')
		ax.legend()

	def Q_values_line_graph(self, ax):
		q_values = self.tracker.q_values
		ax.plot(q_values)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_title('Q values over episodes')
		ax.legend()

	def Policy_reggion_map(self, ax):
		q_values = self.tracker.q_values

		heatmap = ax.pcolormesh(q_values.T, cmap=self.colous)
		ax.set_aspect('equal', 'datalim')
		# cbar = plt.colorbar(heatmap)
		# cbar.set_ticks(range(len(collab)))
		# cbar.set_ticklabels(collab)
		ax.set_xticks(np.linspace(self.x_min, self.x_max, 5))
		ax.set_yticks(np.linspace(self.y_min, self.y_max, 5))
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_title('Policy')

	def Rewards_with_trend(self, ax):
		rewards = self.tracker.rewards
		rolling_mean_reward = self._rolling_window_mean(rewards, self.window_size)
		ax.scatter(rewards, alpha=1, s=1, label='Episode reward')
		ax.plot(rolling_mean_reward, alpha=1, color='orange', label='Avg. 100 episodes')

	def _rolling_window_mean(self, x, n):
		indices = np.arange(0, len(x) + 1, n)[1:]
		prev_i = 0
		result = list()
		for i in indices:
			result.append(sum(x[prev_i: i]))
			prev_i = i
		return result

	def Loss_line_graph(self, ax):
		losses = self.tracker.losses
		rolling_mean_loss = self._rolling_window_mean(losses, self.window_size)
		ax.scatter(losses, alpha=1, s=1, label='Episode reward')
		ax.plot(rolling_mean_loss, alpha=1, color='orange', label='Avg. 100 episodes')
