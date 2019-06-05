import gym, ray, time
import argparse
from datetime import datetime
import tensorflow as tf
from collections import deque
from tf_rl.common.wrappers import MyWrapper, CartPole_Pixel
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import *
from tf_rl.common.policy import EpsilonGreedyPolicy_eager
from tf_rl.common.networks import CartPole as Model, Nature_DQN as Model_p
from tf_rl.agents.DQN import DQN_cartpole, DQN

NUM_WORKERS = 8
ray.init(num_cpus=NUM_WORKERS)
eager_setup()


@ray.remote
class PongEnv(object):
	def __init__(self):
		# Tell numpy to only use one core. If we don't do this, each actor may try
		# to use all of the cores and the resulting contention may result in no
		# speedup over the serial version. Note that if numpy is using OpenBLAS,
		# then you need to set OPENBLAS_NUM_THREADS=1, and you probably need to do
		# it from the command line (so it happens before numpy is imported).
		os.environ["MKL_NUM_THREADS"] = "1"
		self.env = gym.make("CartPole-v0")
		self.env.reset()

	def step(self, action):
		return self.env.step(action)


class _Env:
	def __init__(self, env):
		self.env = env

	def step(self, action):
		return self.env.step.remote(action)


@ray.remote
def one_episode(agent, env):
	current_state = env.reset()
	done = False
	duration = 0
	while not done:
		# env.render()
		duration += 1
		action = agent.choose_action(current_state, 0.5)
		new_state, reward, done, _ = env.step(action)
		current_state = new_state
	return duration

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="CartPole", help="game env type => Atari or CartPole")
parser.add_argument("--seed", default=123, help="seed of randomness")
parser.add_argument("--loss_fn", default="huber", help="types of loss function => MSE or huber")
parser.add_argument("--grad_clip_flg", default="", help="types of a clipping method of gradients => by value(by_value) or global norm(norm) or nothing")
parser.add_argument("--num_frames", default=10000, type=int, help="total frame in a training")
parser.add_argument("--train_interval", default=1, type=int, help="a frequency of training occurring in training phase")
parser.add_argument("--eval_interval", default=250000, type=int, help="a frequency of evaluation occurring in training phase") # temp
parser.add_argument("--memory_size", default=5000, type=int, help="memory size in a training => this used for Experience Replay Memory or Prioritised Experience Replay Memory")
parser.add_argument("--learning_start", default=100, type=int, help="frame number which specifies when to start updating the agent")
parser.add_argument("--sync_freq", default=1000, type=int, help="frequency of updating a target model")
parser.add_argument("--batch_size", default=32, type=int, help="batch size of each iteration of update")
parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.99, type=float, help="discount factor => gamma > 1.0 or negative => does not converge!!")
parser.add_argument("--update_hard_or_soft", default="hard", help="types of synchronisation method of target and main models => soft or hard update")
parser.add_argument("--soft_update_tau", default=1e-2, type=float, help="in soft-update tau defines the ratio of main model remains and it seems 1e-2 is the optimal!")
parser.add_argument("--epsilon_start", default=1.0, type=float, help="initial value of epsilon")
parser.add_argument("--epsilon_end", default=0.02, type=float, help="final value of epsilon")
parser.add_argument("--decay_steps", default=3000, type=int, help="a period for annealing a value(epsilon or beta)")
parser.add_argument("--decay_type", default="linear", help="types of annealing method => linear or curved")
parser.add_argument("--log_dir", default="../../logs/logs/DQN/", help="directory for log")
parser.add_argument("--model_dir", default="../../logs/models/DQN/", help="directory for trained model")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.goal = 195
params.test_episodes = 10

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
now = datetime.now()

if params.google_colab:
	# mount the MyDrive on google drive and create the log directory for saving model and logging using tensorboard
	params.log_dir, params.model_dir, params.log_dir_colab, params.model_dir_colab = setup_on_colab("DQN", params.mode)
else:
	if params.debug_flg:
		params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-DQN_debug/"
		params.model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-DQN_debug/"
	else:
		params.log_dir = "../../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-DQN/"
		params.model_dir = "../../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-DQN/"

if params.mode == "CartPole":
	env = MyWrapper(gym.make("CartPole-v0"))
	agent = DQN_cartpole(Model, optimizer, loss_fn, grad_clip_fn, env.action_space.n, params)
elif params.mode == "CartPole-p":
	env = CartPole_Pixel(gym.make("CartPole-v0"))
	agent = DQN(Model_p, optimizer, loss_fn, grad_clip_fn, env.action_space.n, params)

# set seed
env.seed(params.seed)
tf.random.set_random_seed(params.seed)

# global_timestep = tf.train.get_or_create_global_step()
#
# for episode in range(n_episodes):
# 	agent_id = ray.put(agent)
# 	env_id = ray.put(env)
#
# 	result = list()
# 	for _ in range(NUM_WORKERS):
# 		duration = one_episode.remote(agent_id, env_id)
# 		result.append(duration)
#
# 	result = ray.get(result)
# 	print(result)
# 	sadf


get_ready(agent.params)
global_timestep = tf.train.get_or_create_global_step()
time_buffer = list()
log = logger(agent.params)
with summary_writer.as_default():
	# for summary purpose, we put all codes in this context
	with tf.contrib.summary.always_record_summaries():

		for i in itertools.count():
			state = env.reset()
			total_reward = 0
			start = time.time()
			cnt_action = list()
			done = False
			while not done:
				action = policy.select_action(agent, state)
				next_state, reward, done, info = env.step(action)
				replay_buffer.add(state, action, reward, next_state, done)

				global_timestep.assign_add(1)
				total_reward += reward
				state = next_state
				cnt_action.append(action)

				# for evaluation purpose
				if global_timestep.numpy() % agent.params.eval_interval == 0:
					agent.eval_flg = True

				if (global_timestep.numpy() > agent.params.learning_start) and (
						global_timestep.numpy() % agent.params.train_interval == 0):
					states, actions, rewards, next_states, dones = replay_buffer.sample(agent.params.batch_size)

					loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)

				# synchronise the target and main models by hard or soft update
				if (global_timestep.numpy() > agent.params.learning_start) and (
						global_timestep.numpy() % agent.params.sync_freq == 0):
					agent.manager.save()
					agent.target_model.set_weights(agent.main_model.get_weights())

			"""
			===== After 1 Episode is Done =====
			"""

			tf.contrib.summary.scalar("reward", total_reward, step=i)
			tf.contrib.summary.scalar("exec time", time.time() - start, step=i)
			if i >= agent.params.reward_buffer_ep:
				tf.contrib.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=i)
			tf.contrib.summary.histogram("taken actions", cnt_action, step=i)

			# store the episode reward
			reward_buffer.append(total_reward)
			time_buffer.append(time.time() - start)

			if global_timestep.numpy() > agent.params.learning_start and i % agent.params.reward_buffer_ep == 0:
				log.logging(global_timestep.numpy(), i, np.sum(time_buffer), reward_buffer, np.mean(loss),
							policy.current_epsilon(), cnt_action)
				time_buffer = list()

			if agent.eval_flg:
				test_Agent(agent, env)
				agent.eval_flg = False

			# check the stopping condition
			if global_timestep.numpy() > agent.params.num_frames:
				print("=== Training is Done ===")
				test_Agent(agent, env, n_trial=agent.params.test_episodes)
				env.close()
				break