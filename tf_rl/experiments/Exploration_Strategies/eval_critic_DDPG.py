import argparse
from tf_rl.experiments.Exploration_Strategies.utils import make_grid_env, visualise_critic_values
from tf_rl.experiments.Exploration_Strategies.eval_DDPG import Actor, Critic

from tf_rl.agents.DDPG import DDPG
from tf_rl.common.random_process import OrnsteinUhlenbeckProcess
from tf_rl.common.utils import *

eager_setup()
KERNEL_INIT = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
L2 = tf.keras.regularizers.l2(1e-2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Cont-GridWorld-v2", type=str, help="Env title")
    parser.add_argument("--train_flg", default="original", type=str, help="train flg: original or on-policy")
    parser.add_argument("--seed", default=10, type=int, help="seed for randomness")
    # parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
    parser.add_argument("--num_frames", default=500_000, type=int, help="total frame in a training")
    parser.add_argument("--eval_interval", default=100_000, type=int, help="a frequency of evaluation in training phase")
    parser.add_argument("--memory_size", default=100_000, type=int, help="memory size in a training")
    parser.add_argument("--learning_start", default=10_000, type=int, help="length before training")
    parser.add_argument("--batch_size", default=200, type=int, help="batch size of each iteration of update")
    parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
    parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
    parser.add_argument("--soft_update_tau", default=1e-2, type=float, help="soft-update tau")
    parser.add_argument("--L2_reg", default=0.5, type=float, help="magnitude of L2 regularisation")
    parser.add_argument("--mu", default=0.3, type=float, help="magnitude of randomness")
    parser.add_argument("--sigma", default=0.2, type=float, help="magnitude of randomness")
    parser.add_argument("--action_range", default=[-1., 1.], type=list, help="magnitude of L2 regularisation")
    parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
    parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
    params = parser.parse_args()
    params.test_episodes = 1

    params.log_dir = "./logs/DDPG/logs/DDPG-seed{}/".format(params.seed)
    params.actor_model_dir = "./logs/DDPG/models/DDPG-seed{}/actor/".format(params.seed)
    params.critic_model_dir = "./logs/DDPG/models/DDPG-seed{}/critic/".format(params.seed)
    params.video_dir = "./logs/DDPG/video/DDPG-seed{}/".format(params.seed)
    params.plot_path = "./logs/DDPG/plots/DDPG-seed{}/".format(params.seed)

    env = make_grid_env(plot_path=params.plot_path)

    # set seed
    env.seed(params.seed)
    tf.random.set_random_seed(params.seed)

    random_process = OrnsteinUhlenbeckProcess(size=env.action_space.shape[0], theta=0.15, mu=params.mu, sigma=params.sigma)
    agent = DDPG(Actor, Critic, env.action_space.shape[0], random_process, params)

    get_ready(agent.params)

    visualise_critic_values(env, agent, flg="DDPG")