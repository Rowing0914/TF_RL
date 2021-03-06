import argparse
import tensorflow as tf
from collections import deque
from tf_rl.common.memory import PrioritizedReplayBuffer
from tf_rl.common.utils import gradient_clip_fn, eager_setup, create_loss_func, create_log_model_directory, \
    invoke_agent_env, get_alg_name
from tf_rl.common.policy import EpsilonGreedyPolicy_eager
from tf_rl.common.train import train_DQN_PER
from tf_rl.common.networks import CartPole as Model
from tf_rl.agents.DQN import DQN as DQN_PER, DQN_cartpole as DQN_PER_cartpole

eager_setup()

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="CartPole", help="game env type: Atari or CartPole")
parser.add_argument("--seed", default=123, help="seed of randomness")
parser.add_argument("--loss_fn", default="huber", help="types of loss function: MSE or huber")
parser.add_argument("--grad_clip_flg", default="",
                    help="gradient clippings: by value(by_value) or global norm(norm) or nothing")
parser.add_argument("--num_frames", default=10000, type=int, help="total frame in a training")
parser.add_argument("--train_interval", default=1, type=int, help="a frequency of training occurring in training phase")
parser.add_argument("--eval_interval", default=2500, type=int,
                    help="a frequency of evaluation occurring in training phase")  # temp
parser.add_argument("--memory_size", default=5000, type=int, help="memory size in a training")
parser.add_argument("--learning_start", default=100, type=int,
                    help="frame number which specifies when to start updating the agent")
parser.add_argument("--sync_freq", default=1000, type=int, help="frequency of updating a target model")
parser.add_argument("--batch_size", default=32, type=int, help="batch size of each iteration of update")
parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.99, type=float,
                    help="discount factor: gamma > 1.0 or negative => does not converge!!")
parser.add_argument("--tau", default=1e-2, type=float, help="soft update tau")
parser.add_argument("--ep_start", default=1.0, type=float, help="initial value of epsilon")
parser.add_argument("--ep_end", default=0.02, type=float, help="final value of epsilon")
parser.add_argument("--lr_start", default=0.0025, type=float, help="initial value of lr")
parser.add_argument("--lr_end", default=0.00025, type=float, help="final value of lr")
parser.add_argument("--decay_steps", default=3000, type=int, help="a period for annealing a value(epsilon or beta)")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.goal = 195
params.test_episodes = 10
params.alpha = 0.6
params.beta_start = 0.4
params.beta_end = 1.0
params.prioritized_replay_noise = 1e-6

# init global time-step
global_timestep = tf.train.get_or_create_global_step()

# instantiate annealing funcs for ep and lr
anneal_ep = tf.train.polynomial_decay(params.ep_start, global_timestep, params.decay_steps, params.ep_end)
anneal_lr = tf.train.polynomial_decay(params.lr_start, global_timestep, params.decay_steps, params.lr_end)
beta = tf.train.polynomial_decay(params.beta_start, global_timestep, params.decay_steps, params.beta_end)

# prep for training
policy = EpsilonGreedyPolicy_eager(Epsilon_fn=anneal_ep)
optimizer = tf.train.RMSPropOptimizer(anneal_lr, 0.99, 0.0, 1e-6)
replay_buffer = PrioritizedReplayBuffer(params.memory_size, alpha=params.alpha)
reward_buffer = deque(maxlen=params.reward_buffer_ep)
loss_fn = create_loss_func(params.loss_fn)
grad_clip_fn = gradient_clip_fn(flag=params.grad_clip_flg)

# create a directory for log/model
params = create_log_model_directory(params, get_alg_name())
summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)

# choose env and instantiate the agent correspondingly
agent, env = invoke_agent_env(params, get_alg_name())
agent = eval(agent)(Model, optimizer, loss_fn, grad_clip_fn, env.action_space.n, params)

train_DQN_PER(agent, env, policy, replay_buffer, reward_buffer, beta, summary_writer)
