"""
This is not compatible with Cartpole env, but this converges faster than TF implementation on Atari tasks.

Reference:
- https://github.com/TianhongDai/reinforcement-learning-algorithms/tree/master/01_dqn_algos
"""

import torch
import numpy as np
import argparse
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.set_up import set_up_for_training
from examples.DQN.pytorch.agent import DQN
from examples.DQN.pytorch.train import train
from examples.DQN.main import prep_env
from examples.DQN.utils.policy import EpsilonGreedyPolicy_torch


# set random seeds for the pytorch, numpy and random
def set_seeds(seed, cuda=True, rank=0):
    # set seeds for the numpy
    np.random.seed(seed + rank)
    # set seeds for the pytorch
    torch.manual_seed(seed + rank)
    if cuda:
        torch.cuda.manual_seed(seed + rank)


# linear exploration schedule
class linear_schedule:
    def __init__(self, total_timesteps, final_ratio, init_ratio=1.0):
        self.total_timesteps = total_timesteps
        self.final_ratio = final_ratio
        self.init_ratio = init_ratio

    def get_value(self, timestep):
        frac = min(float(timestep) / self.total_timesteps, 1.0)
        return self.init_ratio - frac * (self.init_ratio - self.final_ratio)


def train_eval(log_dir_name="PytorchDQN",
               random_seed=123,
               env_name="Pong",
               eps_start=1.0,
               eps_end=0.01,
               learning_rate=1e-4,
               decay_rate=0.1,
               num_frames=1000000,
               train_freq=4,
               memory_size=10000,
               hot_start=10000,
               sync_freq=1000,
               batch_size=32,
               interval_MAR=100,
               gamma=0.99,
               num_eval_episodes=1,
               eval_interval=250000,
               cuda=True):
    # init global time-step
    global_timestep = 0

    # instantiate annealing funcs for ep
    anneal_ep = linear_schedule(int(num_frames * decay_rate), eps_end, eps_start)

    # prep for training
    log_dir = set_up_for_training(log_dir_name=log_dir_name, env_name=env_name, seed=random_seed)
    env = prep_env(env_name=env_name, video_path=log_dir["video_path"])
    replay_buffer = ReplayBuffer(memory_size, traj_dir=log_dir["traj_path"])
    reward_buffer = deque(maxlen=interval_MAR)
    summary_writer = SummaryWriter(log_dir=log_dir["summary_path"])

    agent = DQN(num_action=env.action_space.n,
                policy=EpsilonGreedyPolicy_torch(num_action=env.action_space.n, epsilon_fn=anneal_ep),
                summary_writer=summary_writer,
                learning_rate=learning_rate,
                gamma=gamma,
                model_path=log_dir["model_path"],
                cuda=cuda)

    train(global_timestep,
          agent,
          env,
          replay_buffer,
          reward_buffer,
          summary_writer,
          num_eval_episodes,
          num_frames,
          eval_interval,
          hot_start,
          train_freq,
          batch_size,
          sync_freq,
          interval_MAR)


def main(params):
    set_seeds(seed=params.random_seed)
    train_eval(log_dir_name=params.log_dir_name,
               random_seed=params.random_seed,
               env_name=params.env_name,
               eps_start=params.eps_start,
               eps_end=params.eps_end,
               learning_rate=params.learning_rate,
               decay_rate=params.decay_rate,
               num_frames=params.num_frames,
               train_freq=params.train_freq,
               memory_size=params.memory_size,
               hot_start=params.hot_start,
               sync_freq=params.sync_freq,
               batch_size=params.batch_size,
               interval_MAR=params.interval_MAR,
               gamma=params.gamma,
               num_eval_episodes=params.num_eval_episodes,
               eval_interval=params.eval_interval,
               cuda=params.cuda)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Pong", help="name of log directory")
    parser.add_argument("--random_seed", default=123, help="seed of randomness")
    parser.add_argument("--eps_start", default=1.0, help="name of log directory")
    parser.add_argument("--eps_end", default=0.01, help="name of log directory")
    parser.add_argument("--learning_rate", default=1e-4, help="name of log directory")
    parser.add_argument("--decay_rate", default=0.1, help="name of log directory")
    parser.add_argument("--num_frames", default=1000000, help="name of log directory")
    parser.add_argument("--train_freq", default=4, help="name of log directory")
    parser.add_argument("--memory_size", default=10000, help="name of log directory")
    parser.add_argument("--hot_start", default=10000, help="name of log directory")
    parser.add_argument("--sync_freq", default=1000, help="name of log directory")
    parser.add_argument("--batch_size", default=32, help="name of log directory")
    parser.add_argument("--interval_MAR", default=100, help="name of log directory")
    parser.add_argument("--gamma", default=0.99, help="name of log directory")
    parser.add_argument("--num_eval_episodes", default=1, help="name of log directory")
    parser.add_argument("--eval_interval", default=250000, help="name of log directory")
    parser.add_argument("--cuda", default=True, help="name of log directory")
    parser.add_argument("--log_dir_name", default="PyTorchDQN", help="name of log directory")
    params = parser.parse_args()

    main(params)
