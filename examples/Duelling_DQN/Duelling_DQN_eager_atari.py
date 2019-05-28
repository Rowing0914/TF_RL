import os
import argparse
from datetime import datetime
import tensorflow as tf
from collections import deque
from tf_rl.common.wrappers import wrap_deepmind, make_atari
from tf_rl.common.params import ENV_LIST_NATURE
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import AnnealingSchedule, copy_dir
from tf_rl.common.policy import EpsilonGreedyPolicy_eager
from tf_rl.common.train import train_DQN
from tf_rl.agents.DQN import DQN, DQN_debug

config = tf.ConfigProto(allow_soft_placement=True,
						intra_op_parallelism_threads=1,
						inter_op_parallelism_threads=1)
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
tf.random.set_random_seed(123)

class Model(tf.keras.Model):
	def __init__(self, num_action, duelling_type="avg"):
		super(Model, self).__init__()
		self.duelling_type = duelling_type
		self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu')
		self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
		self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
		self.flat = tf.keras.layers.Flatten()
		self.fc1 = tf.keras.layers.Dense(512, activation='relu')
		self.q_value = tf.keras.layers.Dense(num_action, activation='linear')
		self.v_value = tf.keras.layers.Dense(1, activation='linear')

	@tf.contrib.eager.defun
	def call(self, inputs):
		x = self.conv1(inputs)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flat(x)
		x = self.fc1(x)
		q_value = self.q_value(x)
		v_value = self.v_value(x)

		if self.duelling_type == "avg":
			# Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
			output = tf.math.add(v_value, tf.math.subtract(q_value, tf.reduce_mean(q_value)))
		elif self.duelling_type == "max":
			# Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
			output = tf.math.add(v_value, tf.math.subtract(q_value, tf.math.reduce_max(q_value)))
		elif self.duelling_type == "naive":
			# Q(s,a;theta) = V(s;theta) + A(s,a;theta)
			output = tf.math.add(v_value, q_value)
		else:
			output = 0 # defun does not accept the variable may not be intialised, so that temporarily initialise it
			assert False, "dueling_type must be one of {'avg','max','naive'}"
		return output

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", default="Atari", help="game env type => Atari or CartPole")
	parser.add_argument("--env_name", default="Breakout", help="game title")
	parser.add_argument("--loss_fn", default="MSE", help="types of loss function => MSE or huber_loss")
	parser.add_argument("--grad_clip_flg", default="norm", help="types of a clipping method of gradients => by value(by_value) or global norm(norm) or None")
	parser.add_argument("--num_frames", default=10_000_000, type=int, help="total frame in a training")
	parser.add_argument("--train_interval", default=5, type=int, help="a frequency of training occurring in training phase")
	parser.add_argument("--eval_interval", default=250_000, type=int, help="a frequency of evaluation occurring in training phase")
	parser.add_argument("--memory_size", default=500_000, type=int, help="memory size in a training => this used for Experience Replay Memory or Prioritised Experience Replay Memory")
	parser.add_argument("--learning_start", default=20_000, type=int, help="frame number which specifies when to start updating the agent")
	parser.add_argument("--sync_freq", default=1_000, type=int, help="frequency of updating a target model")
	parser.add_argument("--batch_size", default=32, type=int, help="batch size of each iteration of update")
	parser.add_argument("--reward_buffer_ep", default=100, type=int, help="reward_buffer size")
	parser.add_argument("--gamma", default=0.99, type=float, help="discount factor => gamma > 1.0 or negative => does not converge!!")
	parser.add_argument("--update_hard_or_soft", default="hard", help="types of synchronisation method of target and main models => soft or hard update")
	parser.add_argument("--soft_update_tau", default=1e-2, type=float, help="in soft-update tau defines the ratio of main model remains and it seems 1e-2 is the optimal!")
	parser.add_argument("--epsilon_start", default=1.0, type=float, help="initial value of epsilon")
	parser.add_argument("--epsilon_end", default=0.02, type=float, help="final value of epsilon")
	parser.add_argument("--decay_steps", default=250_000, type=int, help="a period for annealing a value(epsilon or beta)")
	parser.add_argument("--decay_type", default="linear", help="types of annealing method => linear or curved")
	parser.add_argument("--log_dir", default="../../logs/logs/DQN/", help="directory for log")
	parser.add_argument("--model_dir", default="../../logs/models/DQN/", help="directory for trained model")
	parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
	parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
	params = parser.parse_args()
	params.goal = ENV_LIST_NATURE["{}NoFrameskip-v4".format(params.env_name)]
	params.test_episodes = 10

	env = wrap_deepmind(make_atari("{}NoFrameskip-v4".format(params.env_name)))
	now = datetime.now()

	if params.google_colab:
		# mount your drive on google colab
		from google.colab import drive
		drive.mount("/content/gdrive")
		params.log_dir         = "/content/TF_RL/logs/logs/Duelling_DQN/{}".format(params.env_name)
		params.model_dir       = "/content/TF_RL/logs/models/Duelling_DQN/{}".format(params.env_name)
		params.log_dir_colab   = "/content/gdrive/My Drive/logs/logs/Duelling_DQN/{}".format(params.env_name)
		params.model_dir_colab = "/content/gdrive/My Drive/logs/models/Duelling_DQN/{}".format(params.env_name)

		# create the logs directory under the root dir
		if not os.path.isdir(params.log_dir):
			os.makedirs(params.log_dir)
		if not os.path.isdir(params.model_dir):
			os.makedirs(params.model_dir)

		# if the previous directory existed in My Drive, then we would continue training on top of the previous training
		if os.path.isdir(params.log_dir_colab):
			print("=== {} IS FOUND ===".format(params.log_dir_colab))
			copy_dir(params.log_dir_colab, params.log_dir, verbose=True)
		else:
			print("=== {} IS NOT FOUND ===".format(params.log_dir_colab))
			os.makedirs(params.log_dir_colab)
			print("=== FINISHED CREATING THE DIRECTORY ===")

		if os.path.isdir(params.model_dir_colab):
			print("=== {} IS FOUND ===".format(params.model_dir_colab))
			copy_dir(params.model_dir_colab, params.model_dir, verbose=True)
		else:
			print("=== {} IS NOT FOUND ===".format(params.model_dir_colab))
			os.makedirs(params.model_dir_colab)
			print("=== FINISHED CREATING THE DIRECTORY ===")

		if params.debug_flg:
			agent = DQN_debug(Model, Model, env.action_space.n, params)
		else:
			agent = DQN(Model, Model, env.action_space.n, params)
	else:
		# run on the local machine
		if params.debug_flg:
			params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "Duelling_DQN_debug/"
			params.model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "Duelling_DQN_debug/"
			agent = DQN_debug(Model, Model, env.action_space.n, params)
		else:
			params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "Duelling_DQN/"
			params.model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "Duelling_DQN/"
			agent = DQN(Model, Model, env.action_space.n, params)

	Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
	policy = EpsilonGreedyPolicy_eager(Epsilon_fn=Epsilon)
	replay_buffer = ReplayBuffer(params.memory_size)
	reward_buffer = deque(maxlen=params.reward_buffer_ep)
	summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)
	train_DQN(agent, env, policy, replay_buffer, reward_buffer, params, summary_writer)
