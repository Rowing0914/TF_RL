import argparse
import gym
import tensorflow as tf
import os
import numpy as np
from common.memory import PrioritizedReplayBuffer
from common.utils import AnnealingSchedule
from common.wrappers import make_atari, wrap_deepmind, MyWrapper
from common.policy import EpsilonGreedyPolicy, BoltzmannQPolicy
from common.params import Parameters
from agents.DQN_PER_model import train_DQN_PER, DQN_PER_CartPole, DQN_PER_Atari

try:
	os.system("rm -rf ../logs/summary_PER_main")
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
	main_model = DQN_PER_CartPole("PER_main", env, loss_fn="huber_loss", grad_clip_flg=params.grad_clip_flg)
	target_model = DQN_PER_CartPole("PER_target", env, loss_fn="huber_loss", grad_clip_flg=params.grad_clip_flg)
	replay_buffer = PrioritizedReplayBuffer(params.memory_size, alpha=params.prioritized_replay_alpha)
	if params.policy_fn == "Eps":
		Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
		policy = EpsilonGreedyPolicy(Epsilon_fn=Epsilon)
	elif params.policy_fn == "Boltzmann":
		policy = BoltzmannQPolicy()
	Beta = AnnealingSchedule(start=params.prioritized_replay_beta_start, end=params.prioritized_replay_beta_end,
	                         decay_steps=params.decay_steps)
elif args.mode == "Atari":
	env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
	params = Parameters(mode="Atari")
	main_model = DQN_PER_Atari("PER_main", env, loss_fn="huber_loss", grad_clip_flg=params.grad_clip_flg)
	target_model = DQN_PER_Atari("PER_target", env, loss_fn="huber_loss", grad_clip_flg=params.grad_clip_flg)
	replay_buffer = PrioritizedReplayBuffer(params.memory_size, alpha=params.prioritized_replay_alpha)
	if params.policy_fn == "Eps":
		Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
		policy = EpsilonGreedyPolicy(Epsilon_fn=Epsilon)
	elif params.policy_fn == "Boltzmann":
		policy = BoltzmannQPolicy()
	Beta = AnnealingSchedule(start=params.prioritized_replay_beta_start, end=params.prioritized_replay_beta_end, decay_steps=params.decay_steps)
else:
	print("Select 'mode' either 'Atari' or 'CartPole' !!")


all_rewards, losses = train_DQN_PER(main_model, target_model, env, replay_buffer, policy, Beta, params)

np.save("../logs/value/rewards_DQN_PER.npy", np.array(all_rewards))