import gym
import argparse
import tensorflow as tf
from collections import deque
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import eager_setup
from tf_rl.agents.SAC import SAC
from tf_rl.common.train import train_SAC
from tf_rl.common.params import DDPG_ENV_LIST
from tf_rl.common.networks import DDPG_Actor as Actor, DDPG_Critic as Critic

eager_setup()

"""
this is defined in params.py
DDPG_ENV_LIST = {
	"Ant-v2": 3500,
	"HalfCheetah-v2": 7000,
	"Hopper-v2": 1500,
	"Humanoid-v2": 2000,
	"HumanoidStandup-v2": 0,
	"InvertedDoublePendulum-v2": 6000,
	"InvertedPendulum-v2": 800,
	"Reacher-v2": -6,
	"Swimmer-v2": 40,
	"Walker2d-v2": 2500
}
"""

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="Ant-v2", help="Env title")
parser.add_argument("--seed", default=123, type=int, help="seed for randomness")
parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
parser.add_argument("--train_interval", default=100, type=int, help="a frequency of training occurring in training phase")
parser.add_argument("--nb_train_steps", default=50, type=int, help="a number of training, which occurs once in train_interval above!!")
parser.add_argument("--eval_interval", default=5_000, type=int, help="a frequency of evaluation occurring in training phase")
parser.add_argument("--memory_size", default=100_000, type=int, help="memory size in a training")
parser.add_argument("--learning_start", default=10_000, type=int, help="frame number which specifies when to start updating the agent")
parser.add_argument("--batch_size", default=100, type=int, help="batch size of each iteration of update")
parser.add_argument("--reward_buffer_ep", default=5, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.99, type=float, help="discount factor => gamma > 1.0 or negative => does not converge!!")
parser.add_argument("--soft_update_tau", default=1e-2, type=float, help="soft-update needs tau to define the ratio of main model remains")
parser.add_argument("--L2_reg", default=0.5, type=float, help="magnitude of L2 regularisation")
parser.add_argument("--action_range", default=[-1., 1.], type=list, help="magnitude of L2 regularisation")
parser.add_argument("--log_dir", default="../../logs/logs/SAC/", help="directory for log")
parser.add_argument("--model_dir", default="../../logs/models/SAC/", help="directory for trained model")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.test_episodes = 10

from datetime import datetime
now = datetime.now()
if params.debug_flg:
	params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-SAC/"
	params.model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-SAC/"
else:
	params.log_dir = "../../logs/logs/{}".format(params.env_name)
	params.model_dir = "../../logs/models/{}".format(params.env_name)


env = gym.make(params.env_name)
# set seed
env.seed(params.seed)
tf.random.set_random_seed(params.seed)

params.goal = DDPG_ENV_LIST[params.env_name]
agent = SAC(Actor, Critic, env.action_space.shape[0], params)
replay_buffer = ReplayBuffer(params.memory_size)
reward_buffer = deque(maxlen=params.reward_buffer_ep)
summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)
train_SAC(agent, env, replay_buffer, reward_buffer, params, summary_writer)