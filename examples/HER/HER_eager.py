import gym
import argparse
import tensorflow as tf
from tf_rl.common.memory import HER_replay_buffer
from tf_rl.common.utils import eager_setup, her_sampler, create_log_model_directory, get_alg_name
from tf_rl.common.params import ROBOTICS_ENV_LIST
from tf_rl.common.train import train_HER
from tf_rl.common.networks import HER_Actor as Actor, HER_Critic as Critic
from tf_rl.agents.HER import HER_DDPG as HER, HER_DDPG_debug as HER_debug

eager_setup()

"""
# defined in params.py
ROBOTICS_ENV_LIST = {
    "FetchPickAndPlace-v1"
    "FetchPush-v1"
    "FetchReach-v1"
    "FetchSlide-v1"
}
"""

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="MuJoCo", help="Task mode")
parser.add_argument("--env_name", default="FetchReach-v1", help="Env title")
parser.add_argument("--seed", default=123, type=int, help="seed for randomness")
parser.add_argument("--num_epochs", default=200, type=int, help="number of epochs in a training")
parser.add_argument("--num_cycles", default=50, type=int, help="number of cycles in epoch")
parser.add_argument("--num_episodes", default=2, type=int, help="number of episodes in cycle")
parser.add_argument("--num_steps", default=50, type=int, help="number of steps in an episode")
parser.add_argument("--replay_strategy", default="future", help="replay_strategy")
parser.add_argument("--replay_k", default=4, type=int, help="number of replay strategy")
parser.add_argument("--num_updates", default=40, type=int, help="number of updates in cycle")
parser.add_argument("--memory_size", default=1000000, type=int, help="memory size in a training")
parser.add_argument("--batch_size", default=256, type=int, help="batch size of each iteration of update")
parser.add_argument("--gamma", default=0.98, type=float, help="discount factor")
parser.add_argument("--tau", default=0.05, type=float, help="soft-update tau")
parser.add_argument("--action_l2", default=1.0, type=float, help="magnitude of L2 regularisation")
parser.add_argument("--noise_eps", default=0.2, type=float, help="magnitude of noise")
parser.add_argument("--random_eps", default=0.3, type=float, help="magnitude of randomness")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.goal = ROBOTICS_ENV_LIST[params.env_name]
params.test_episodes = 10

env = gym.make(params.env_name)
params.max_action = env.action_space.high[0]
params.num_action = env.action_space.shape[0]

# set seed
env.seed(params.seed)
tf.random.set_random_seed(params.seed)

# create a directory for log/model
params = create_log_model_directory(params, get_alg_name())

if params.debug_flg:
	agent = HER_debug(Actor, Critic, env.action_space.shape[0], params)
else:
	agent = HER(Actor, Critic, env.action_space.shape[0], params)

# prep for basic stats
obs = env.reset()

env_params = {
	'obs': obs['observation'].shape[0],
	'goal': obs['desired_goal'].shape[0],
	'action': env.action_space.shape[0],
	'action_max': env.action_space.high[0],
	'max_timesteps': env._max_episode_steps
}

her_sample_func = her_sampler(params.replay_strategy, params.replay_k, env.compute_reward)
replay_buffer = HER_replay_buffer(env_params, params.memory_size, her_sample_func.sample_her_transitions)
summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)
train_HER(agent, env, replay_buffer, summary_writer)
