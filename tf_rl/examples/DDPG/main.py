import gin
import gym
import argparse
import numpy as np
import tensorflow as tf
from collections import deque
from tf_rl.common.monitor import Monitor
from tf_rl.common.set_up import set_up_for_training
from tf_rl.common.random_process import OrnsteinUhlenbeckProcess, GaussianNoise
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import eager_setup
from tf_rl.examples.DDPG.utils.network import Actor, Critic
from tf_rl.examples.DDPG.utils.agent import DDPG
from tf_rl.examples.DDPG.utils.train import train

eager_setup()


@gin.configurable
def train_eval(log_dir="DDPG",
               prev_log="",
               google_colab=False,
               seed=123,
               gpu_id=0,
               env_name="HalfCheetah-v2",
               num_frames=10000,
               tau=1e-2,
               memory_size=5000,
               hot_start=100,
               batch_size=200,
               interval_MAR=10,
               gamma=0.99,
               L2_reg=0.5,
               random_process="ou",
               mu=0.3,
               sigma=0.2,
               num_eval_episodes=1,
               eval_interval=1000):
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed=seed)

    # prep for training
    log_dir = set_up_for_training(env_name=env_name,
                                  seed=seed,
                                  gpu_id=gpu_id,
                                  log_dir=log_dir,
                                  prev_log=prev_log,
                                  google_colab=google_colab)

    env = gym.make(env_name)
    env = Monitor(env=env, directory=log_dir["video_path"], force=True)

    replay_buffer = ReplayBuffer(memory_size, traj_dir=log_dir["traj_path"])
    reward_buffer = deque(maxlen=interval_MAR)
    summary_writer = tf.compat.v2.summary.create_file_writer(log_dir["summary_path"])

    if random_process == "ou":
        random_process = OrnsteinUhlenbeckProcess(size=env.action_space.shape[0],
                                                  theta=0.15,
                                                  mu=mu,
                                                  sigma=sigma)
    elif random_process == "gaussian":
        random_process = GaussianNoise(mu=mu,
                                       sigma=sigma)
    else:
        random_process = False
        assert False, "choose the random process from either gaussian or ou"

    agent = DDPG(actor=Actor,
                 critic=Critic,
                 num_action=env.action_space.shape[0],
                 random_process=random_process,
                 gamma=gamma,
                 L2_reg=L2_reg,
                 actor_model_dir=log_dir["model_path"] + "/actor",
                 critic_model_dir=log_dir["model_path"] + "/critic")

    train(agent,
          env,
          replay_buffer,
          reward_buffer,
          summary_writer,
          num_eval_episodes,
          num_frames,
          tau,
          eval_interval,
          hot_start,
          batch_size,
          interval_MAR,
          log_dir,
          google_colab)


def main(gin_file, gin_params, log_dir, prev_log, google_colab):
    eager_setup()
    gin.parse_config_file(gin_file)
    if gin_params:
        gin_params_flat = [param[0] for param in gin_params]
        gin.parse_config_files_and_bindings([params.gin_file], gin_params_flat)
    train_eval(log_dir=log_dir,
               prev_log=prev_log,
               google_colab=google_colab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--gin_file", default="./config/test.gin", help="gin config")
    parser.add_argument("--gin_file", default="./config/main.gin", help="gin config")
    parser.add_argument("--gin_params", default=None, action='append', nargs='+', help="extra gin params to override")
    parser.add_argument("--log_dir", default="DDPG", help="name of log directory")
    parser.add_argument("--prev_log", default="", help="Previous training directories")
    parser.add_argument("--google_colab", default=False, help="if you run this on google_colab")
    params = parser.parse_args()

    main(gin_file=params.gin_file,
         gin_params=params.gin_params,
         log_dir=params.log_dir,
         prev_log=params.prev_log,
         google_colab=params.google_colab)
