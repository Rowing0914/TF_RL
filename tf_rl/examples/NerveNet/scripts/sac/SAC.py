import gym
import argparse
import tensorflow as tf
from datetime import datetime
from collections import deque
from tf_rl.common.monitor import Monitor
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import eager_setup
from tf_rl.agents.SAC import SAC, SAC_debug
from tf_rl.common.train import train_SAC
from tf_rl.common.networks import SAC_Actor as Actor, SAC_Critic as Critic

from environments.register import register

eager_setup()

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="Hopper-v2", help="Env title")
parser.add_argument("--seed", default=20, type=int, help="seed for randomness")
parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
parser.add_argument("--eval_interval", default=100_000, type=int, help="a frequency of evaluation in training phase")
parser.add_argument("--memory_size", default=100_000, type=int, help="memory size in a training")
parser.add_argument("--learning_start", default=10_000, type=int, help="length before training")
parser.add_argument("--batch_size", default=256, type=int, help="batch size of each iteration of update")
parser.add_argument("--reward_buffer_ep", default=5, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
parser.add_argument("--alpha", default=0.2, type=float, help="Temperature param for the relative importance of entropy")
parser.add_argument("--soft_update_tau", default=0.005, type=float, help="soft-update")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.goal = 0
params.test_episodes = 10

now = datetime.now()

params.log_dir = "../logs/logs/SAC-seed{}/{}".format(params.seed, str(params.env_name.split("-")[0]))
params.actor_model_dir = "../logs/models/SAC-seed{}/{}/actor/".format(params.seed, str(params.env_name.split("-")[0]))
params.critic_model_dir = "../logs/models/SAC-seed{}/{}/critic/".format(params.seed, str(params.env_name.split("-")[0]))
params.video_dir = "../logs/video/SAC-seed{}/{}".format(params.seed, str(params.env_name.split("-")[0]))
params.plot_path = "../logs/plots/SAC-seed{}/{}".format(params.seed, str(params.env_name.split("-")[0]))

env = gym.make(params.env_name)
env = Monitor(env, params.video_dir)

# set seed
env.seed(params.seed)
tf.random.set_random_seed(params.seed)

if params.debug_flg:
    agent = SAC_debug(Actor, Critic, env.action_space.shape[0], params)
else:
    agent = SAC(Actor, Critic, env.action_space.shape[0], params)

replay_buffer = ReplayBuffer(params.memory_size)
reward_buffer = deque(maxlen=params.reward_buffer_ep)
summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)
train_SAC(agent, env, replay_buffer, reward_buffer, summary_writer)
