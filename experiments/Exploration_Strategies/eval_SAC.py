import argparse

import tensorflow_probability as tfp

from experiments.Exploration_Strategies.utils import make_grid_env
from tf_rl.agents.SAC import SAC
from tf_rl.common.utils import *

tfd = tfp.distributions
XAVIER_INIT = tf.contrib.layers.xavier_initializer()

eager_setup()


class Actor(tf.keras.Model):
    """
    Policy network: Gaussian Policy.
    It outputs Mean and Std with the size of number of actions.
    And we sample from Normal dist upon resulting Mean&Std

    In Haarnoja's implementation, he uses 100 neurons for hidden layers... so it's up to you!!
    """

    def __init__(self, num_action=1):
        super(Actor, self).__init__()
        self.LOG_SIG_MAX = 2
        self.LOG_SIG_MIN = -20

        self.dense1 = tf.keras.layers.Dense(20, activation='relu', kernel_initializer=XAVIER_INIT)
        self.dense2 = tf.keras.layers.Dense(20, activation='relu', kernel_initializer=XAVIER_INIT)
        self.mean = tf.keras.layers.Dense(num_action, activation='linear', kernel_initializer=XAVIER_INIT)
        self.std = tf.keras.layers.Dense(num_action, activation='linear', kernel_initializer=XAVIER_INIT)

    @tf.contrib.eager.defun(autograph=False)
    def call(self, inputs):
        """
        As mentioned in the topic of `policy evaluation` at sec5.2(`ablation study`) in the paper,
        for evaluation phase, using a deterministic action(choosing the mean of the policy dist) works better than
        stochastic one(Gaussian Policy). So that we need to output three different values. I know it's kind of weird design..
        """
        x = self.dense1(inputs)
        x = self.dense2(x)
        mean = self.mean(x)
        std = self.std(x)
        std = tf.clip_by_value(std, self.LOG_SIG_MIN, self.LOG_SIG_MAX)
        std = tf.math.exp(std)
        dist = tfd.Normal(loc=mean, scale=std)
        x = dist.sample()
        action = tf.keras.activations.tanh(x)
        log_prob = dist.log_prob(x)
        log_prob -= tf.math.log(1. - tf.math.square(action) + 1e-6)
        log_prob = tf.math.reduce_sum(log_prob, 1, keep_dims=True)
        return action, log_prob, tf.keras.activations.tanh(mean)


class Critic(tf.keras.Model):
    """
    It contains two Q-network. And the usage of two Q-functions improves performance by reducing overestimation bias.
    """

    def __init__(self, output_shape):
        super(Critic, self).__init__()
        # Q1 architecture
        self.dense1 = tf.keras.layers.Dense(20, activation='relu', kernel_initializer=XAVIER_INIT)
        self.dense2 = tf.keras.layers.Dense(20, activation='relu', kernel_initializer=XAVIER_INIT)
        self.Q1 = tf.keras.layers.Dense(output_shape, activation='linear', kernel_initializer=XAVIER_INIT)

        # Q2 architecture
        self.dense3 = tf.keras.layers.Dense(20, activation='relu', kernel_initializer=XAVIER_INIT)
        self.dense4 = tf.keras.layers.Dense(20, activation='relu', kernel_initializer=XAVIER_INIT)
        self.Q2 = tf.keras.layers.Dense(output_shape, activation='linear', kernel_initializer=XAVIER_INIT)

    @tf.contrib.eager.defun(autograph=False)
    def call(self, obs, act):
        """ Original Implementation """
        _concat = tf.concat([obs, act], axis=-1)
        x1 = self.dense1(_concat)
        x1 = self.dense2(x1)
        Q1 = self.Q1(x1)

        x2 = self.dense3(_concat)
        x2 = self.dense4(x2)
        Q2 = self.Q2(x2)
        return Q1, Q2


parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="Cont-GridWorld-v2", help="Env title")
parser.add_argument("--seed", default=10, type=int, help="seed for randomness")
# parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
parser.add_argument("--num_frames", default=500_000, type=int, help="total frame in a training")
parser.add_argument("--eval_interval", default=100_000, type=int, help="a frequency of evaluation in training phase")
parser.add_argument("--memory_size", default=100_000, type=int, help="memory size in a training")
parser.add_argument("--learning_start", default=10_000, type=int, help="length before training")
parser.add_argument("--batch_size", default=200, type=int, help="batch size of each iteration of update")
parser.add_argument("--reward_buffer_ep", default=5, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
parser.add_argument("--alpha", default=0.2, type=float, help="Temperature param for the relative importance of entropy")
parser.add_argument("--soft_update_tau", default=0.005, type=float, help="soft-update")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.goal = 0
params.test_episodes = 1

params.log_dir = "./logs/logs/SAC-seed{}/".format(params.seed)
params.actor_model_dir = "./logs/models/SAC-seed{}/actor/".format(params.seed)
params.critic_model_dir = "./logs/models/SAC-seed{}/critic/".format(params.seed)
params.video_dir = "./logs/video/SAC-seed{}/".format(params.seed)
params.plot_path = "./logs/plots/SAC-seed{}/".format(params.seed)

env = make_grid_env(plot_path=params.plot_path)

# set seed
env.seed(params.seed)
tf.random.set_random_seed(params.seed)

agent = SAC(Actor, Critic, env.action_space.shape[0], params)

get_ready(agent.params)

traj = list()

print("=== Evaluation Mode ===")
for ep in range(params.test_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        traj.append(state)
        action = agent.eval_predict(state)
        next_state, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
        state = next_state
        episode_reward += reward

    traj = np.array(traj)
    env.vis_exploration(traj=traj,
                        file_name="eval_exploration_{}.png".format(tf.compat.v1.train.get_global_step().numpy()))
    env.vis_trajectory(traj=traj, file_name="eval_traj_{}.png".format(tf.compat.v1.train.get_global_step().numpy()))
    tf.contrib.summary.scalar("Evaluation Score", episode_reward, step=agent.index_timestep)
    print("| Ep: {}/{} | Score: {} |".format(ep + 1, params.test_episodes, episode_reward))
