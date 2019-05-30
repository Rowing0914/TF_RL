import tensorflow as tf
import argparse, gym
from importlib import import_module
from tf_rl.common.wrappers import wrap_deepmind, make_atari
from tf_rl.run_experiment import Runner

def arg_parsing():
	""" Returns a parsed collection of arguments

	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--alg", default="DQN", help="Agent name")
	parser.add_argument("--env", default="Pong", help="Env title")
	parser.add_argument("--env_type", default="atari", help="Env type: atari or mujoco")
	parser.add_argument("--seed", default=132, type=int, help="Seed for randomness")
	parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
	parser.add_argument("--max_episode_steps", default=None, type=int, help="max steps in a episode")
	parser.add_argument("--train_steps", default=250_000, type=int, help="steps for one training phase")
	parser.add_argument("--eval_steps", default=125_000, type=int, help="steps for one evaluation phase")
	parser.add_argument("--train_interval", default=100, type=int,
						help="a frequency of training occurring in training phase")
	parser.add_argument("--nb_train_steps", default=50, type=int,
						help="a number of training, which occurs once in train_interval above!!")
	parser.add_argument("--eval_interval", default=50_000, type=int,
						help="a frequency of evaluation occurring in training phase")
	parser.add_argument("--memory_size", default=100_000, type=int, help="memory size in a training")
	parser.add_argument("--learning_start", default=2_000, type=int,
						help="frame number which specifies when to start updating the agent")
	parser.add_argument("--batch_size", default=32, type=int, help="batch size of each iteration of update")
	parser.add_argument("--reward_buffer_ep", default=5, type=int, help="reward_buffer size")
	parser.add_argument("--gamma", default=0.99, type=float,
						help="discount factor => gamma > 1.0 or negative => does not converge!!")
	parser.add_argument("--soft_update_tau", default=1e-2, type=float,
						help="soft-update needs tau to define the ratio of main model remains")
	parser.add_argument("--L2_reg", default=1e-2, type=float, help="magnitude of L2 regularisation")
	parser.add_argument("--action_range", default=[-1., 1.], type=list, help="magnitude of L2 regularisation")
	parser.add_argument("--log_dir", default="../../logs/logs/DDPG/", help="directory for log")
	parser.add_argument("--model_dir", default="../../logs/models/DDPG/", help="directory for trained model")
	parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
	parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
	return parser.parse_args()


def prep_agent_env(args):
	""" Returns the compiled agent and env
	"""
	agent_init = find_agent(args.alg)
	agent = build_agent(agent_init, args)
	env = find_env(args.env_type, args.env, args.seed)
	return agent, env


def build_agent(agent_init, args):
	""" Init the agent with given args

	:param agent_init:
	:return:
	"""
	agent = agent_init(
		env=env,
		seed=seed,
		total_timesteps=total_timesteps,
		**args
	)
	return agent


def find_agent(alg_name):
	""" finds the target file and import it
	"""
	try:
		# first try to import the alg module from tf_rl
		agent_init = import_module('.'.join(['tf_rl.agents', alg_name, "agent_init"]))
		# get the function to initialise the agent
		agent_init = getattr(agent_init, "init")
	except ImportError:
		agent_init = {}
		print("We cannot find the algorithm within our directory, recheck if it exists")

	return agent_init


def find_env(env_type, env_name, seed):
	""" finds and returns the env
	only supports atari single thread for now.
	"""
	env = make_env(env_name, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
	return env


def make_env(env_name, env_type, seed, wrapper_kwargs=None):
	if env_type == 'atari':
		env = wrap_deepmind(make_atari("{}NoFrameskip-v4".format(env_name)), wrapper_kwargs)
	else:
		env = gym.make(env_name)
	env.seed(seed)
	return env


# env = Monitor(env,
#               logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
#               allow_early_resets=True)

def enable_eager(seed):
	config = tf.ConfigProto(allow_soft_placement=True,
							intra_op_parallelism_threads=1,
							inter_op_parallelism_threads=1)
	config.gpu_options.allow_growth = True
	tf.enable_eager_execution(config=config)
	tf.random.set_random_seed(seed)


def main(args):
	enable_eager(args.seed)
	tf.logging.set_verbosity(tf.logging.INFO)
	agent, env = prep_agent_env(args)
	runner = Runner(agent, env, args)
	runner.run_experiment()


if __name__ == '__main__':
	main(arg_parsing())
