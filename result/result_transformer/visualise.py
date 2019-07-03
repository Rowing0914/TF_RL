import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

DDPG_ENV_LIST = {
    "Ant-v2": 3500,
    "HalfCheetah-v2": 7000,
    "Hopper-v2": 1500,
    "Humanoid-v2": 2000,
    "InvertedDoublePendulum-v2": 6000,
    "InvertedPendulum-v2": 800,
    "Reacher-v2": -6,
    "Swimmer-v2": 40,
    "Walker2d-v2": 2500
}

with_target_line = "./result_graphs/with_target_line"
without_target_line = "./result_graphs/without_target_line"

if not os.path.isdir(with_target_line):
    os.makedirs(with_target_line)

if not os.path.isdir(without_target_line):
    os.makedirs(without_target_line)

# Env list
for env_name, goal_score in DDPG_ENV_LIST.items():
    print("==== {} ====".format(env_name))
    current_dir = "./logs/{}/".format(env_name)
    # get filenames
    filenames = os.listdir(current_dir)

    # pick the bigger file
    if len(filenames) > 1:
        if os.path.getsize(current_dir + filenames[0]) > os.path.getsize(current_dir + filenames[1]):
            target = current_dir + filenames[0]
        else:
            target = current_dir + filenames[1]
    else:
        target = current_dir + filenames[0]

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'histograms': 1
    }

    # load it by EventAccumulator
    event_acc = EventAccumulator(target, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    # print(event_acc.Tags())
    # asdf

    # get reward as list/ndarray
    ma_reward = event_acc.Scalars("Moving_Ave_Reward")
    reward = event_acc.Scalars("reward")
    _ma_reward = np.array([ma_reward[i][2] for i in range(len(ma_reward))])
    _reward = np.array([reward[i][2] for i in range(len(reward))])

    # visualise
    plt.subplot(2, 1, 1)
    plt.plot(_ma_reward, label="Result")
    plt.axhline(y=goal_score, linewidth=2, color='r', label="Target")
    plt.title("Moving Ave R(100ep): {}".format(env_name))
    plt.xlabel("episode(100ep)")
    plt.ylabel("MA Reward")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(_reward, label="Result")
    plt.axhline(y=goal_score, linewidth=2, color='r', label="Target")
    plt.xlabel("episode")
    plt.ylabel("Reward")
    plt.title("Episode R: {}".format(env_name))
    plt.legend()
    plt.savefig("./result_graphs/with_target_line/{}.png".format(env_name))
    # clear graph
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.plot(_ma_reward, label="Result")
    plt.title("Moving Ave R(100ep): {}, Goal: {}".format(env_name, goal_score))
    plt.xlabel("episode(100ep)")
    plt.ylabel("MA Reward")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(_reward, label="Result")
    plt.xlabel("episode")
    plt.ylabel("Reward")
    plt.title("Episode R: {}, Goal: {}".format(env_name, goal_score))
    plt.legend()
    plt.savefig("./result_graphs/without_target_line/{}.png".format(env_name))

    # clear graph and data
    plt.clf()
    del ma_reward;
    del reward;
    del _ma_reward;
    del _reward;
