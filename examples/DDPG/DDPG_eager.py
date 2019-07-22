import gym
import argparse
import tensorflow as tf
from datetime import datetime
from collections import deque
from gym.wrappers import Monitor
from tf_rl.common.random_process import OrnsteinUhlenbeckProcess, GaussianNoise
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import eager_setup
from tf_rl.agents.DDPG import DDPG
from tf_rl.common.train import train_DDPG
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
parser.add_argument("--seed", default=123, type=int, help="seed for randomness")
# parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
parser.add_argument("--num_frames", default=500_000, type=int, help="total frame in a training")
# parser.add_argument("--num_frames", default=40_000, type=int, help="total frame in a training")
parser.add_argument("--eval_interval", default=100_000, type=int, help="a frequency of evaluation in training phase")
# parser.add_argument("--eval_interval", default=10_000, type=int, help="a frequency of evaluation in training phase")
parser.add_argument("--memory_size", default=100_000, type=int, help="memory size in a training")
parser.add_argument("--learning_start", default=10_000, type=int, help="length before training")
parser.add_argument("--batch_size", default=200, type=int, help="batch size of each iteration of update")
parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
parser.add_argument("--soft_update_tau", default=1e-2, type=float, help="soft-update tau")
parser.add_argument("--L2_reg", default=0.5, type=float, help="magnitude of L2 regularisation")
parser.add_argument("--mu", default=0.2, type=float, help="magnitude of randomness")
parser.add_argument("--sigma", default=0.05, type=float, help="magnitude of randomness")
parser.add_argument("--action_range", default=[-1., 1.], type=list, help="magnitude of L2 regularisation")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.test_episodes = 1
params.goal = DDPG_ENV_LIST[params.env_name]

now = datetime.now()

# params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-DDPG/"
# params.actor_model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-DDPG_actor/"
# params.critic_model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-DDPG_critic/"
# params.video_dir = "../../logs/video/video_{}".format(now.strftime("%Y%m%d-%H%M%S") + "_" + str(params.env_name))
# params.plot_path = "../../logs/plots/plot_{}/".format(now.strftime("%Y%m%d-%H%M%S") + "_" + str(params.env_name))
mu = str(params.mu).split(".")
mu = str(mu[0]+mu[1])
params.log_dir = "../../logs/logs/DDPG/{}-mu{}".format(str(params.env_name.split("-")[0]), mu)
params.actor_model_dir = "../../logs/models/DDPG/{}/actor-mu{}/".format(str(params.env_name.split("-")[0]), mu)
params.critic_model_dir = "../../logs/models/DDPG/{}/critic-mu{}/".format(str(params.env_name.split("-")[0]), mu)
params.video_dir = "../../logs/video/{}-mu{}".format(str(params.env_name.split("-")[0]), mu)
params.plot_path = "../../logs/plots/{}-mu{}/".format(str(params.env_name.split("-")[0]), mu)

env = gym.make(params.env_name)
env = Monitor(env,
              params.video_dir,
              video_callable=lambda _:
              True if tf.compat.v1.train.get_global_step().numpy() % params.eval_interval==0 else False,
              force=True)

# set seed
env.seed(params.seed)
tf.random.set_random_seed(params.seed)

replay_buffer = ReplayBuffer(params.memory_size)
reward_buffer = deque(maxlen=params.reward_buffer_ep)
summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)
# random_process = OrnsteinUhlenbeckProcess(size=env.action_space.shape[0], theta=0.15, mu=0.9, sigma=0.05)
random_process = GaussianNoise(mu=params.mu, sigma=params.sigma)
agent = DDPG(Actor, Critic, env.action_space.shape[0], random_process, params)
train_DDPG(agent, env, replay_buffer, reward_buffer, summary_writer)
