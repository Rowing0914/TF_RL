import gym
import argparse
import tensorflow as tf
from tf_rl.common.utils import eager_setup
from tf_rl.agents.DDPG import DDPG
from tf_rl.common.networks import DDPG_Actor as Actor, DDPG_Critic as Critic
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

eager_setup()

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="Ant-v2", help="Env title")
parser.add_argument("--seed", default=123, type=int, help="seed for randomness")
parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
parser.add_argument("--train_interval", default=100, type=int, help="a frequency of training in training phase")
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
parser.add_argument("--mu", default=0.2, type=float, help="magnitude of randomness")
parser.add_argument("--sigma", default=0.05, type=float, help="magnitude of randomness")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.test_episodes = 10
params.goal = 0

mu = str(params.mu).split(".")
mu = str(mu[0]+mu[1])
params.actor_model_dir = "../../../logs/models/DDPG-original/{}/actor-mu{}/".format(str(params.env_name.split("-")[0]), mu)
params.critic_model_dir = "../../../logs/models/DDPG-original/{}/critic-mu{}/".format(str(params.env_name.split("-")[0]), mu)

env = gym.make(params.env_name)
tf.random.set_random_seed(params.seed)
random_process = None
agent = DDPG(Actor, Critic, env.action_space.shape[0], random_process, params)

# set seed
env.seed(params.seed)

state = env.reset()
done = False

actions = list()

while not done:
    # deterministic policy
    action = agent.eval_predict(state)
    # action = env.action_space.sample()

    # scale for execution in env (in DDPG, every action is clipped between [-1, 1] in agent.predict)
    next_state, reward, done, info = env.step(action * env.action_space.high)
    actions.append(action)
    state = next_state

env.close()

import numpy as np
actions = np.array(actions)

for i in range(len(action)):
    print(actions[:, i].shape)
