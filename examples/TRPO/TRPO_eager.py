import gym
import argparse
import tensorflow as tf
from datetime import datetime
from collections import deque
from tf_rl.common.utils import eager_setup
from tf_rl.agents.TRPO import TRPO, TRPO_debug
from tf_rl.common.train import train_TRPO
from tf_rl.common.params import DDPG_ENV_LIST
from tf_rl.common.networks import TRPO_Policy as Policy, TRPO_Value as Value

eager_setup()

"""
this is defined in params.py
DDPG_ENV_LIST = {
	"Ant-v2": 3500,
	"HalfCheetah-v2": 7000,
	"Hopper-v2": 1500,
	"Humanoid-v2": 2000,
	"HumanoidStandup-v2": 0, # maybe we don't need this...
	"InvertedDoublePendulum-v2": 6000,
	"InvertedPendulum-v2": 800,
	"Reacher-v2": -6,
	"Swimmer-v2": 40,
	"Walker2d-v2": 2500
}
"""

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="Hopper-v2", help="Env title")
parser.add_argument("--seed", default=123, type=int, help="seed for randomness")
parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
parser.add_argument("--num_rollout", default=10, type=int, help="total frame in a training")
parser.add_argument("--num_updates", default=20, type=int, help="total updates in after an episode")
parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.995, type=float, help="discount factor")
parser.add_argument("--L2_reg", default=0.001, type=float, help="magnitude of L2 regularisation")
parser.add_argument("--gae_discount", default=0.98, type=float, help="Lambda for Generalized Advantage Estimation")
parser.add_argument("--kl_target", default=0.003, type=float, help="target for kl divergence")
parser.add_argument("--log_dir", default="../../logs/logs/TRPO/", help="directory for log")
parser.add_argument("--model_dir", default="../../logs/models/TRPO/", help="directory for trained model")
parser.add_argument("--debug_flg", default=True, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.test_episodes = 10
params.goal = DDPG_ENV_LIST[params.env_name]

env = gym.make(params.env_name)

# set seed
env.seed(params.seed)
tf.random.set_random_seed(params.seed)

now = datetime.now()

if params.debug_flg:
    params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-TRPO/"
    params.model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-TRPO/"
    agent = TRPO_debug(Policy, Value, env.action_space.shape[0], params)
else:
    params.log_dir = "../../logs/logs/{}".format(params.env_name)
    params.model_dir = "../../logs/models/{}".format(params.env_name)
    agent = TRPO(Policy, Value, env.action_space.shape[0], params)

reward_buffer = deque(maxlen=params.reward_buffer_ep)
summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)

train_TRPO(agent, env, reward_buffer, summary_writer)
