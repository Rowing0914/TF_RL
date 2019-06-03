import gym
import argparse
import tensorflow as tf
from datetime import datetime
from collections import deque
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import eager_setup
from tf_rl.common.params import ROBOTICS_ENV_LIST
from tf_rl.common.train_parallised import train_HER
from tf_rl.common.networks import DDPG_Actor as Actor, DDPG_Critic as Critic
from tf_rl.agents.DDPG import DDPG

eager_setup()

"""
# defined in params.py
ROBOTICS_ENV_LIST = {
    "FetchPickAndPlace-v1": 0,
    "FetchPush-v1": 0,
    "FetchReach-v1": 0,
    "FetchSlide-v1": 0
}
"""

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="FetchPickAndPlace-v1", help="Env title")
parser.add_argument("--seed", default=123, type=int, help="seed for randomness")
parser.add_argument("--num_epochs", default=200, type=int, help="number of epochs in a training")
parser.add_argument("--num_cycles", default=50, type=int, help="number of cycles in epoch")
parser.add_argument("--num_episodes", default=16, type=int, help="number of episodes in cycle")
parser.add_argument("--num_updates", default=40, type=int, help="number of updates in cycle")
parser.add_argument("--eval_interval", default=5_000, type=int, help="a frequency of evaluation occurring in training phase")
parser.add_argument("--memory_size", default=100_000, type=int, help="memory size in a training")
parser.add_argument("--learning_start", default=10_000, type=int, help="frame number which specifies when to start updating the agent")
parser.add_argument("--batch_size", default=128, type=int, help="batch size of each iteration of update")
parser.add_argument("--gamma", default=0.98, type=float, help="discount factor => gamma > 1.0 or negative => does not converge!!")
parser.add_argument("--soft_update_tau", default=0.05, type=float, help="soft-update needs tau to define the ratio of main model remains")
parser.add_argument("--L2_reg", default=0.5, type=float, help="magnitude of L2 regularisation")
parser.add_argument("--log_dir", default="../../logs/logs/HER/", help="directory for log")
parser.add_argument("--model_dir", default="../../logs/models/HER/", help="directory for trained model")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
params = parser.parse_args()
params.goal = ROBOTICS_ENV_LIST[params.env_name]

now = datetime.now()

if params.debug_flg:
	params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-HER/"
	params.model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-HER/"
else:
	params.log_dir = "../../logs/logs/{}".format(params.env_name)
	params.model_dir = "../../logs/models/{}".format(params.env_name)

env = gym.make(params.env_name)
# set seed
env.seed(params.seed)
tf.random.set_random_seed(params.seed)

agent = DDPG(Actor, Critic, env.action_space.shape[0], params)
replay_buffer = ReplayBuffer(params.memory_size)
reward_buffer = deque(maxlen=params.num_episodes)
summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)
train_HER(agent, env, replay_buffer, reward_buffer, summary_writer)