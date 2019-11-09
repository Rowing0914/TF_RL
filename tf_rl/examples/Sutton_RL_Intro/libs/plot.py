import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_result(stats):
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(stats.shape[0]), stats[:, 0])
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")

    plt.subplot(3, 1, 2)
    rewards_smoothed = pd.Series(stats[:, 1]).rolling(10, min_periods=10).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(10))

    plt.subplot(3, 1, 3)
    plt.plot(np.cumsum(stats[:, 0]), np.arange(stats.shape[0]))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    plt.show()


# def compare_plots(stats_1, label_1, stats_2, label_2):
#     if stats_1.shape[0] != stats_2.shape[0]:
#         return "the dimension of the matrices does not match"
#     else:
#         plt.subplot(3, 1, 1)
#         plt.plot(np.arange(stats_1.shape[0]), stats_1[:,0], label=label_1)
#         plt.plot(np.arange(stats_2.shape[0]), stats_2[:,0], label=label_2)
#         plt.xlabel("Episode")
#         plt.ylabel("Episode Length")
#         plt.title("Episode Length over Time")
#         plt.legend()
#
#         plt.subplot(3, 1, 2)
#         rewards_smoothed_1 = pd.Series(stats_1[:,1]).rolling(10, min_periods=10).mean()
#         rewards_smoothed_2 = pd.Series(stats_2[:,1]).rolling(10, min_periods=10).mean()
#         plt.plot(rewards_smoothed_1, label=label_1)
#         plt.plot(rewards_smoothed_2, label=label_2)
#         plt.xlabel("Episode")
#         plt.ylabel("Episode Reward (Smoothed)")
#         plt.title("Episode Reward over Time (Smoothed over window size {})".format(10))
#         plt.legend()
#
#         plt.subplot(3, 1, 3)
#         plt.plot(np.cumsum(stats_1[:,0]), np.arange(stats_1.shape[0]), label=label_1)
#         plt.plot(np.cumsum(stats_2[:,0]), np.arange(stats_2.shape[0]), label=label_2)
#         plt.xlabel("Time Steps")
#         plt.ylabel("Episode")
#         plt.title("Episode per time step")
#         plt.legend()
#
#         plt.show()

def compare_plots(**kwargs):
    """
    this takes input as dictionary contains defined as below, then produce the graph
    consisting of multiple stats
    :param kwargs: {label_name: stats}
    :return: none
    """
    plt.subplot(3, 1, 1)
    for label, data in kwargs.items():
        plt.plot(np.arange(data.shape[0]), data[:, 0], label=label)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.legend()

    plt.subplot(3, 1, 2)
    for label, data in kwargs.items():
        rewards_smoothed = pd.Series(data[:, 1]).rolling(10, min_periods=10).mean()
        plt.plot(rewards_smoothed, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(10))
    plt.legend()

    plt.subplot(3, 1, 3)
    for label, data in kwargs.items():
        plt.plot(np.cumsum(data[:, 0]), np.arange(data.shape[0]), label=label)
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    plt.legend()

    plt.show()
