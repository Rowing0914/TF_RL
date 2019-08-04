import gym
import argparse
import tensorflow as tf
from datetime import datetime
from collections import deque
from tf_rl.common.monitor import Monitor
from tf_rl.common.random_process import OrnsteinUhlenbeckProcess, GaussianNoise
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import eager_setup
from tf_rl.agents.DDPG import DDPG
from tf_rl.common.train import train_DDPG_original, train_DDPG_offpolicy
from tf_rl.common.networks import DDPG_Actor as Actor, DDPG_Critic as Critic

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
parser.add_argument("--env_name", default="Ant-v2", type=str, help="Env title")
parser.add_argument("--train_flg", default="original", type=str, help="train flg: original or off-policy")
# parser.add_argument("--train_flg", default="off-policy", type=str, help="train flg: original or off-policy")
parser.add_argument("--seed", default=123, type=int, help="seed for randomness")
parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
# parser.add_argument("--num_frames", default=500_000, type=int, help="total frame in a training")
# parser.add_argument("--num_frames", default=20_000, type=int, help="total frame in a training")
parser.add_argument("--eval_interval", default=100_000, type=int, help="a frequency of evaluation in training phase")
# parser.add_argument("--eval_interval", default=5_000, type=int, help="a frequency of evaluation in training phase")
parser.add_argument("--memory_size", default=100_000, type=int, help="memory size in a training")
parser.add_argument("--learning_start", default=10_000, type=int, help="length before training")
# parser.add_argument("--learning_start", default=1_000, type=int, help="length before training")
parser.add_argument("--batch_size", default=200, type=int, help="batch size of each iteration of update")
parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
parser.add_argument("--soft_update_tau", default=1e-2, type=float, help="soft-update tau")
parser.add_argument("--L2_reg", default=0.5, type=float, help="magnitude of L2 regularisation")
parser.add_argument("--mu", default=0.3, type=float, help="magnitude of randomness")
parser.add_argument("--sigma", default=0.05, type=float, help="magnitude of randomness")
parser.add_argument("--action_range", default=[-1., 1.], type=list, help="magnitude of L2 regularisation")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.test_episodes = 1

now = datetime.now()

# params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-DDPG/"
# params.actor_model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-DDPG_actor/"
# params.critic_model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-DDPG_critic/"
# params.video_dir = "../../logs/video/video_{}".format(now.strftime("%Y%m%d-%H%M%S") + "_" + str(params.env_name))
# params.plot_path = "../../logs/plots/plot_{}/".format(now.strftime("%Y%m%d-%H%M%S") + "_" + str(params.env_name))
mu = str(params.mu).split(".")
mu = str(mu[0]+mu[1])
params.log_dir = "../../logs/logs/DDPG-{}-seed{}/{}-mu{}".format(params.train_flg, params.seed, str(params.env_name.split("-")[0]), mu)
params.actor_model_dir = "../../logs/models/DDPG-{}-seed{}/{}/actor-mu{}/".format(params.train_flg, params.seed, str(params.env_name.split("-")[0]), mu)
params.critic_model_dir = "../../logs/models/DDPG-{}-seed{}/{}/critic-mu{}/".format(params.train_flg, params.seed, str(params.env_name.split("-")[0]), mu)
params.video_dir = "../../logs/video/DDPG-{}-seed{}/{}-mu{}/".format(params.train_flg, params.seed, str(params.env_name.split("-")[0]), mu)
params.plot_path = "../../logs/plots/DDPG-{}-seed{}/{}-mu{}/".format(params.train_flg, params.seed, str(params.env_name.split("-")[0]), mu)

env = gym.make(params.env_name)
env = Monitor(env, params.video_dir)

# set seed
env.seed(params.seed)
tf.random.set_random_seed(params.seed)

replay_buffer = ReplayBuffer(params.memory_size)
reward_buffer = deque(maxlen=params.reward_buffer_ep)
summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)
random_process = OrnsteinUhlenbeckProcess(size=env.action_space.shape[0], theta=0.15, mu=params.mu, sigma=params.sigma)
# random_process = GaussianNoise(mu=params.mu, sigma=params.sigma)
agent = DDPG(Actor, Critic, env.action_space.shape[0], random_process, params)

if params.train_flg == "original":
    train_DDPG_original(agent, env, replay_buffer, reward_buffer, summary_writer)
elif params.train_flg == "off-policy":
    train_DDPG_offpolicy(agent, env, replay_buffer, reward_buffer, summary_writer)