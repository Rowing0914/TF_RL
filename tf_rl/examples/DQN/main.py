import gym
import gin
import argparse
import numpy as np
import tensorflow as tf
import functools
from collections import deque
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import gradient_clip_fn
from tf_rl.common.eager_util import eager_setup
from tf_rl.common.set_up import set_up_for_training
from tf_rl.common.monitor import Monitor
from tf_rl.common.wrappers import wrap_deepmind, make_atari
from tf_rl.examples.DQN.utils.policy import EpsilonGreedyPolicy_eager
from tf_rl.examples.DQN.utils.network import atari_net, cartpole_net
from tf_rl.examples.DQN.utils.agent import dqn_agent
from tf_rl.examples.DQN.utils.train import train


def prep_env(env_name, video_path):
    if env_name.lower() == "cartpole":
        env = gym.make("CartPole-v0")
        env.record_start = lambda: None
        env.record_end = lambda: None
    else:
        env = wrap_deepmind(make_atari(env_name + "NoFrameskip-v4"),
                            frame_stack=True)  # make sure to add NoFrameskip-v4
        env = Monitor(env=env, directory=video_path, force=True)
    return env


def prep_obs_processor(env_name):
    if env_name.lower() == "cartpole":
        obs_prc_fn = lambda x: x
    else:
        obs_prc_fn = lambda x: np.array(x) / 255.
    return obs_prc_fn


def prep_model(env_name, network_type=None):
    if env_name.lower() == "cartpole":
        model = cartpole_net
    else:
        model = functools.partial(atari_net, network_type=network_type)
    return model


@gin.configurable
def train_eval(log_dir="DQN",
               prev_log="",
               google_colab=False,
               seed=123,
               gpu_id=0,
               env_name="CartPole",
               network_type="fast",
               eps_start=1.0,
               eps_end=0.02,
               decay_steps=3000,
               optimizer=tf.keras.optimizers.RMSprop,
               learning_rate=0.00025,
               decay=0.95,
               momentum=0.0,
               epsilon=0.00001,
               centered=True,
               loss_fn=tf.compat.v1.losses.huber_loss,
               grad_clip_flg=None,
               num_frames=10000,
               train_freq=1,
               memory_size=5000,
               hot_start=100,
               sync_freq=1000,
               batch_size=32,
               interval_MAR=10,
               gamma=0.99,
               num_eval_episodes=1,
               eval_interval=1000):
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed=seed)

    # init global time-step
    global_timestep = tf.compat.v1.train.create_global_step()

    # instantiate annealing funcs for ep
    anneal_ep = tf.compat.v1.train.polynomial_decay(eps_start, global_timestep, decay_steps, eps_end)

    # prep for training
    log_dir = set_up_for_training(env_name=env_name,
                                  seed=seed,
                                  gpu_id=gpu_id,
                                  log_dir=log_dir,
                                  prev_log=prev_log,
                                  google_colab=google_colab)
    env = prep_env(env_name=env_name, video_path=log_dir["video_path"])
    replay_buffer = ReplayBuffer(memory_size, traj_dir=log_dir["traj_path"])
    reward_buffer = deque(maxlen=interval_MAR)
    summary_writer = tf.compat.v2.summary.create_file_writer(log_dir["summary_path"])

    agent = dqn_agent(model=prep_model(env_name, network_type=network_type),
                      policy=EpsilonGreedyPolicy_eager(num_action=env.action_space.n, epsilon_fn=anneal_ep),
                      optimizer=optimizer(learning_rate, decay, momentum, epsilon, centered),
                      loss_fn=loss_fn,
                      grad_clip_fn=gradient_clip_fn(flag=grad_clip_flg),
                      num_action=env.action_space.n,
                      model_dir=log_dir["model_path"],
                      gamma=gamma,
                      obs_prc_fn=prep_obs_processor(env_name))

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
    # parser.add_argument("--gin_file", default="./config/test.gin", help="cartpole or atari")
    # parser.add_argument("--gin_file", default="./config/cartpole.gin", help="cartpole or atari")
    parser.add_argument("--gin_file", default="./config/dopamine.gin", help="cartpole or atari")
    # parser.add_argument("--gin_file", default="./config/experimental/adam_mse.gin", help="cartpole or atari")
    # parser.add_argument("--gin_file", default="./config/experimental/adam_huber.gin", help="cartpole or atari")
    # parser.add_argument("--gin_file", default="./config/experimental/rmsprop_mse.gin", help="cartpole or atari")
    parser.add_argument("--gin_params", default=None, action='append', nargs='+', help="extra gin params to override")
    parser.add_argument("--log_dir", default="DQN", help="name of log directory")
    parser.add_argument("--prev_log", default="", help="Previous training directories")
    parser.add_argument("--google_colab", default=False, help="if you run this on google_colab")
    params = parser.parse_args()

    main(gin_file=params.gin_file,
         gin_params=params.gin_params,
         log_dir=params.log_dir,
         prev_log=params.prev_log,
         google_colab=params.google_colab)