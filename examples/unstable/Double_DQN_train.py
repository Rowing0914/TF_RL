import argparse
import gym
import tensorflow as tf
import os
import numpy as np
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import AnnealingSchedule
from tf_rl.common.wrappers import make_atari, wrap_deepmind, MyWrapper
from tf_rl.common.policy import EpsilonGreedyPolicy, BoltzmannQPolicy
from tf_rl.common.params import Parameters
from tf_rl.agents.unstable.DQN_model import DQN_CartPole, DQN_Atari
from tf_rl.agents.unstable.Double_DQN_model import train_Double_DQN

try:
    os.system("rm -rf ../logs/summary_Double_main")
except:
    pass

# initialise a graph in a session
tf.reset_default_graph()

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="CartPole", help="game env type")
args = parser.parse_args()

if args.mode == "CartPole":
    env = MyWrapper(gym.make("CartPole-v0"))
    params = Parameters(mode="CartPole")
    main_model = DQN_CartPole("Double_main", env, "huber_loss", grad_clip_flg=params.grad_clip_flg)
    target_model = DQN_CartPole("Double_target", env, "huber_loss", grad_clip_flg=params.grad_clip_flg)
    replay_buffer = ReplayBuffer(params.memory_size)
    if params.policy_fn == "Eps":
        Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
        policy = EpsilonGreedyPolicy(Epsilon_fn=Epsilon)
    elif params.policy_fn == "Boltzmann":
        policy = BoltzmannQPolicy()
elif args.mode == "Atari":
    env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
    params = Parameters(mode="Atari")
    main_model = DQN_Atari("Double_main", env, "huber_loss", grad_clip_flg=params.grad_clip_flg)
    target_model = DQN_Atari("Double_target", env, "huber_loss", grad_clip_flg=params.grad_clip_flg)
    replay_buffer = ReplayBuffer(params.memory_size)
    if params.policy_fn == "Eps":
        Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
        policy = EpsilonGreedyPolicy(Epsilon_fn=Epsilon)
    elif params.policy_fn == "Boltzmann":
        policy = BoltzmannQPolicy()
else:
    print("Select 'mode' either 'Atari' or 'CartPole' !!")

all_rewards, losses = train_Double_DQN(main_model, target_model, env, replay_buffer, policy, params)

np.save("../logs/value/rewards_Double_DQN.npy", np.array(all_rewards))
