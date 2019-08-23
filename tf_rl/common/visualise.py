import os
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


def visualise_act_and_dist(epochs, action_buffer, distance_buffer, env_name, file_dir="../../logs/plots/"):
    """ DDPG eval visualisation method """

    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

    for i in range(distance_buffer.shape[0]):
        plt.hist(distance_buffer[i], bins=100, alpha=0.3, density=True, label="Epoch_{}".format(epochs[i]))
        plt.xlabel("Distance")
        plt.ylabel("Density")
        plt.title("Histogram of Distance: {}".format(env_name))
        # plt.axes()
        plt.grid()
        plt.legend()
    # plt.savefig(file_dir + "distance_density.eps", metadata="eps")
    plt.savefig(file_dir + "distance_density.png")

    for i in range(distance_buffer.shape[0]):
        plt.hist(action_buffer[i], bins=100, alpha=0.3, density=True, label="Epoch_{}".format(epochs[i]))
        plt.xlabel("Mean Squared Action")
        plt.ylabel("Density")
        plt.title("Histogram of Mean Squared Action: {}".format(env_name))
        # plt.axes()
        plt.grid()
        plt.legend()
    # plt.savefig(file_dir + "action_density.eps", metadata="eps")
    plt.savefig(file_dir + "action_density.png")

    for i in range(distance_buffer.shape[0]):
        plt.plot(np.cumsum(distance_buffer[i]), label="Epoch_{}".format(epochs[i]))
        plt.xlabel("Time-Step")
        plt.ylabel("Acc Distance Over Timestep")
        plt.title("Histogram of Accumulated Distance Over Time-step: {}".format(env_name))
        # plt.axes()
        plt.grid()
        plt.legend()
    # plt.savefig(file_dir + "acc_distance_over_time.eps", metadata="eps")
    plt.savefig(file_dir + "acc_distance_over_time.png")
    np.save(file_dir + "ms_action", action_buffer)
    np.save(file_dir + "distance", distance_buffer)
    np.save(file_dir + "acc_distance", np.cumsum(distance_buffer))

def plot_comparison_graph(model_names):
    """
    2D Plot rewards arose in games(model_names)

    :param model_names:
    :return:
    """
    for model_name in model_names:
        rewards = np.load("../logs/value/rewards_{}.npy".format(model_name))
        plt.plot(rewards, label=model_name)
        plt.title("Score over Episodes")
        plt.xlabel("Episodes")
        plt.ylabel("Score")
    plt.legend()
    plt.show()


def plot_Q_values(data, xmin=-1, xmax=4, ymin=0, ymax=2):
    """
    Real time Bar plot of Q_values

    :param data:
    :param xmin:
    :param xmax:
    :param ymin:
    :param ymax:
    :return:
    """
    plt.axis([xmin, xmax, ymin, ymax])
    # print(data)
    length = np.arange(data.shape[0])
    plt.bar(length, data, align='center')
    plt.xticks(length)
    plt.xlabel("Actions")
    plt.ylabel("Q_values")
    plt.pause(0.02)
    plt.clf()


class Plotting:
    def __init__(self, nb_actions, env, model, tracker):
        self.nb_actions = nb_actions
        self.tracker = tracker  # tracking method to contain all important values arise in training phase
        self.nb_plots = 6  # number of currently available visualisation methods
        self.window_size = 1000  # window size of rolling mean

        self.x_min, self.x_max = env.observation_space.low[0], env.observation_space.high[0]
        self.y_min, self.y_max = env.observation_space.low[1], env.observation_space.high[1]

        # associating colours with each action in the env
        self.colours = plt.get_cmap('jet', self.nb_actions)
        self.colours.set_under('gray')

        fig, ax = plt.subplots()

        self.Q_max_3D_plot(ax)
        self.State_spcace_2D_scatter(ax)
        self.Q_values_line_graph(ax)
        self.Policy_region_map(ax)
        self.Rewards_with_trend(ax)
        self.Loss_line_graph(ax)

    def Q_max_3D_plot(self, ax):
        """

        Plot the final Q-values over states space
        we can see what kind state has high Q-value

        """

        q_max = np.max(self.tracker.q_values,
                       axis=-1)  # calculate the maximum value w.r.t the most right feature in Q values

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

    def Policy_region_map(self, ax):
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
