import os
import gym
import argparse
import tensorflow as tf
from collections import deque
from tf_rl.common.wrappers import MyWrapper, CartPole_Pixel
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import AnnealingSchedule
from tf_rl.common.policy import EpsilonGreedyPolicy
# from tf_rl.common.train import train_DQN
from tf_rl.agents.unstable.DQN_model import DQN, DQN_CartPole, train_DQN

config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1)
config.gpu_options.allow_growth = True


# tf.random.set_random_seed(123)


class Model(tf.keras.Model):
    def __init__(self, num_action):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(16, activation='relu')
        self.dense3 = tf.keras.layers.Dense(16, activation='relu')
        self.pred = tf.keras.layers.Dense(num_action, activation='linear')

    @tf.contrib.eager.defun
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.pred(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="CartPole", help="game env type => Atari or CartPole")
    parser.add_argument("--env_name", default="Breakout", help="game title")  # temp
    parser.add_argument("--loss_fn", default="huber_loss", help="types of loss function => MSE or huber_loss")
    parser.add_argument("--grad_clip_flg", default="None",
                        help="types of a clipping method of gradients => by value(by_value) or global norm(norm) or None")
    parser.add_argument("--num_frames", default=10_000, type=int, help="total frame in a training")
    parser.add_argument("--train_interval", default=1, type=int,
                        help="a frequency of training occurring in training phase")
    parser.add_argument("--eval_interval", default=250_000, type=int,
                        help="a frequency of evaluation occurring in training phase")  # temp
    parser.add_argument("--memory_size", default=5_000, type=int,
                        help="memory size in a training => this used for Experience Replay Memory or Prioritised Experience Replay Memory")
    parser.add_argument("--learning_start", default=100, type=int,
                        help="frame number which specifies when to start updating the agent")
    parser.add_argument("--sync_freq", default=1_000, type=int, help="frequency of updating a target model")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size of each iteration of update")
    parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
    parser.add_argument("--gamma", default=0.99, type=float,
                        help="discount factor => gamma > 1.0 or negative => does not converge!!")
    parser.add_argument("--update_hard_or_soft", default="hard",
                        help="types of synchronisation method of target and main models => soft or hard update")
    parser.add_argument("--soft_update_tau", default=1e-2, type=float,
                        help="in soft-update tau defines the ratio of main model remains and it seems 1e-2 is the optimal!")
    parser.add_argument("--epsilon_start", default=1.0, type=float, help="initial value of epsilon")
    parser.add_argument("--epsilon_end", default=0.02, type=float, help="final value of epsilon")
    parser.add_argument("--decay_steps", default=3_000, type=int,
                        help="a period for annealing a value(epsilon or beta)")
    parser.add_argument("--decay_type", default="linear", help="types of annealing method => linear or curved")
    parser.add_argument("--log_dir", default="../../logs/logs/DQN/", help="directory for log")
    parser.add_argument("--model_dir", default="../../logs/models/DQN/", help="directory for trained model")
    parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
    parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
    params = parser.parse_args()
    params.goal = 195
    params.test_episodes = 10

    if params.mode == "CartPole":
        env = MyWrapper(gym.make("CartPole-v0"))
    elif params.mode == "CartPole-p":
        env = CartPole_Pixel(gym.make("CartPole-v0"))

    if params.google_colab:
        # mount your drive on google colab
        from google.colab import drive

        drive.mount("/content/gdrive")
        params.log_dir = "/content/gdrive/My Drive/logs/logs/DQN/{}".format(params.env_name)
        params.model_dir = "/content/gdrive/My Drive/logs/models/DQN/{}".format(params.env_name)
        os.makedirs(params.log_dir)
        os.makedirs(params.model_dir)
        assert os.path.isdir(params.log_dir), "Faild to create a directory on your My Drive, pls check it"
        assert os.path.isdir(params.model_dir), "Faild to create a directory on your My Drive, pls check it"
        agent = DQN(params, env.action_space.n)
    else:
        agent = DQN(params, env.action_space.n)

Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
policy = EpsilonGreedyPolicy(Epsilon_fn=Epsilon)
replay_buffer = ReplayBuffer(params.memory_size)
reward_buffer = deque(maxlen=params.reward_buffer_ep)
summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)
train_DQN(agent, env, policy, replay_buffer, reward_buffer, params, summary_writer)
