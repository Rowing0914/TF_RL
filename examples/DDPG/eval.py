import gym
from gym_extensions.continuous import mujoco
import argparse
import tensorflow as tf
from collections import deque
from tf_rl.common.random_process import OrnsteinUhlenbeckProcess
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import eager_setup
from tf_rl.agents.DDPG import DDPG
from tf_rl.common.params import DDPG_ENV_LIST
from tf_rl.common.networks import DDPG_Actor as Actor, DDPG_Critic as Critic

eager_setup()

"""
this is defined in params.py
DDPG_ENV_LIST = {
	"Ant-v2": 3500,
	"HalfCheetah-v2": 7000,
	"Hopper-v2": 1500,
	"Humanoid-v2": 2000,
	"HumanoidStandup-v2": 0, # maybe we don't need this...
	"InvertedDoublePendulum-v2": 6000,
	"InvertedPendulum-v2": 800,
	"Reacher-v2": -6,
	"Swimmer-v2": 40,
	"Walker2d-v2": 2500
}
"""

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="Humanoid-v2", help="Env title")
parser.add_argument("--seed", default=123, type=int, help="seed for randomness")
parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
parser.add_argument("--train_interval", default=100, type=int,  help="a frequency of training in training phase")
parser.add_argument("--nb_train_steps", default=50, type=int, help="a number of training after one episode")
parser.add_argument("--eval_interval", default=50_000, type=int, help="a frequency of evaluation in training phase")
parser.add_argument("--memory_size", default=100_000, type=int, help="memory size in a training")
parser.add_argument("--learning_start", default=10_000, type=int, help="length before training")
parser.add_argument("--batch_size", default=100, type=int, help="batch size of each iteration of update")
parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
parser.add_argument("--soft_update_tau", default=1e-2, type=float, help="soft-update tau ")
parser.add_argument("--L2_reg", default=0.5, type=float, help="magnitude of L2 regularisation")
parser.add_argument("--action_range", default=[-1., 1.], type=list, help="magnitude of L2 regularisation")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.test_episodes = 10
params.goal = DDPG_ENV_LIST[params.env_name]

params.actor_model_dir = "./models/actor/"
params.critic_model_dir = "./models/critic/"

# available env list: https://github.com/Rowing0914/gym-extensions/blob/mujoco200/tests/all_tests.py
env = gym.make("HumanoidBigTorso-v1")

# set seed
env.seed(params.seed)
tf.random.set_random_seed(params.seed)

replay_buffer = ReplayBuffer(params.memory_size)
reward_buffer = deque(maxlen=params.reward_buffer_ep)
# summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)
random_process = OrnsteinUhlenbeckProcess(size=env.action_space.shape[0], theta=0.15, mu=0.0, sigma=0.05)
agent = DDPG(Actor, Critic, env.action_space.shape[0], random_process, params)

n_trial = 10
all_rewards = list()
for ep in range(n_trial):
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        env.render()
        action = agent.eval_predict(state)
        # scale for execution in env (in DDPG, every action is clipped between [-1, 1] in agent.predict)
        next_state, reward, done, _ = env.step(action * env.action_space.high)
        state = next_state
        episode_reward += reward

    all_rewards.append(episode_reward)
    tf.contrib.summary.scalar("Evaluation Score", episode_reward, step=agent.index_timestep)
    print("| Ep: {}/{} | Score: {} |".format(ep + 1, n_trial, episode_reward))