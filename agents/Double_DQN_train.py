import gym
import tensorflow as tf
import os
import numpy as np
from common.memory import ReplayBuffer
from common.utils import AnnealingSchedule
from common.wrappers import make_atari, wrap_deepmind, MyWrapper
from common.policy import EpsilonGreedyPolicy, BoltzmannQPolicy
from common.params import Parameters
from agents.DQN_model import DQN_CartPole, DQN_Atari
from agents.Double_DQN_model import train_Double_DQN

try:
    os.system("rm -rf ../logs/summary_Double_main")
except:
    pass

# initialise a graph in a session
tf.reset_default_graph()

mode = "CartPole"
# mode = "Atari"

if mode == "CartPole":
    env = MyWrapper(gym.make("CartPole-v0"))
    params = Parameters(mode="CartPole")
    main_model = DQN_CartPole("Double_main", env, "huber_loss")
    target_model = DQN_CartPole("Double_target", env, "huber_loss")
    replay_buffer = ReplayBuffer(params.memory_size)
    if params.policy_fn == "Eps":
        Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
        policy = EpsilonGreedyPolicy(Epsilon_fn=Epsilon)
    elif params.policy_fn == "Boltzmann":
        policy = BoltzmannQPolicy()
elif mode == "Atari":
    env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
    params = Parameters(mode="Atari")
    main_model = DQN_Atari("Double_main", env, "huber_loss")
    target_model = DQN_Atari("Double_target", env, "huber_loss")
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

