import argparse
from datetime import datetime
import tensorflow as tf
from collections import deque
from tf_rl.common.wrappers import wrap_deepmind, make_atari
from tf_rl.common.params import ENV_LIST_NATURE
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import AnnealingSchedule, eager_setup, gradient_clip_fn, setup_on_colab
from tf_rl.common.policy import EpsilonGreedyPolicy_eager
from tf_rl.common.train import train_DQN
from tf_rl.common.networks import Nature_DQN as Model
from tf_rl.agents.Double_DQN import Double_DQN, Double_DQN_debug

eager_setup()

"""
in addition to the params below, I am using inside a training API
- skipping frame(k=4)
- epsilon: 0.05 for evaluation phase
- learning rate decay over the same period as the epsilon
"""

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="Atari", help="game env type => Atari or CartPole")
parser.add_argument("--env_name", default="Breakout", help="game title")
parser.add_argument("--seed", default=123, help="seed of randomness")
parser.add_argument("--loss_fn", default="mse", help="types of loss function => mse or huber")
parser.add_argument("--grad_clip_flg", default="", help="types of a clipping method of gradients => by value(by_value) or global norm(norm) or nothing")
parser.add_argument("--num_frames", default=50_000_000, type=int, help="total frame in a training")
parser.add_argument("--train_interval", default=4, type=int, help="a frequency of training occurring in training phase")
parser.add_argument("--eval_interval", default=250_000, type=int, help="a frequency of evaluation occurring in training phase")
parser.add_argument("--memory_size", default=1_000_000, type=int, help="memory size in a training")
parser.add_argument("--learning_start", default=50_000, type=int, help="frame number which specifies when to start updating the agent")
parser.add_argument("--sync_freq", default=10_000, type=int, help="frequency of updating a target model")
parser.add_argument("--batch_size", default=32, type=int, help="batch size of each iteration of update")
parser.add_argument("--reward_buffer_ep", default=100, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.99, type=float, help="discount factor => gamma > 1.0 or negative => does not converge!!")
parser.add_argument("--epsilon_start", default=1.0, type=float, help="initial value of epsilon")
parser.add_argument("--epsilon_end", default=0.1, type=float, help="final value of epsilon")
parser.add_argument("--decay_steps", default=1_000_000, type=int, help="a period for annealing a value(epsilon or beta)")
parser.add_argument("--decay_type", default="linear", help="types of annealing method => linear or curved")
parser.add_argument("--log_dir", default="../../logs/logs/Double_DQN/", help="directory for log")
parser.add_argument("--model_dir", default="../../logs/models/Double_DQN/", help="directory for trained model")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.goal = ENV_LIST_NATURE["{}NoFrameskip-v4".format(params.env_name)]
params.test_episodes = 10

env = wrap_deepmind(make_atari("{}NoFrameskip-v4".format(params.env_name)))

# set seed
env.seed(params.seed)
tf.random.set_random_seed(params.seed)

now = datetime.now()

if params.google_colab:
	# mount the MyDrive on google drive and create the log directory for saving model and logging using tensorboard
	params.log_dir, params.model_dir, params.log_dir_colab, params.model_dir_colab = setup_on_colab("Double_DQN", params.env_name)
else:
	if params.debug_flg:
		params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-Double_DQN_debug/"
		params.model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-Double_DQN_debug/"
	else:
		params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-Double_DQN/"
		params.model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-Double_DQN/"

Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
policy = EpsilonGreedyPolicy_eager(Epsilon_fn=Epsilon)
replay_buffer = ReplayBuffer(params.memory_size)
reward_buffer = deque(maxlen=params.reward_buffer_ep)
anneal_lr = AnnealingSchedule(start=0.0025, end=0.00025, decay_steps=params.decay_steps, decay_type="linear")
optimizer = tf.train.RMSPropOptimizer(anneal_lr.get_value(), 0.99, 0.0, 1e-6)

if params.loss_fn == "huber":
	loss_fn = tf.losses.huber_loss
elif params.loss_fn == "mse":
	loss_fn = tf.losses.mean_squared_error
else:
	assert False, "Choose the loss_fn from either huber or mse"

grad_clip_fn = gradient_clip_fn(flag=params.grad_clip_flg)
summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)

if params.debug_flg:
	agent = Double_DQN_debug(Model, Model, env.action_space.n, params)
else:
	agent = Double_DQN(Model, optimizer, loss_fn, grad_clip_fn, env.action_space.n, params.gamma, params.model_dir)

train_DQN(agent, env, policy, replay_buffer, reward_buffer, params, summary_writer)