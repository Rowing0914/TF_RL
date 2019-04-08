import gym
import tensorflow as tf
import os
from common.memory import PrioritizedReplayBuffer
from common.utils import AnnealingSchedule
from common.wrappers import make_atari, wrap_deepmind, MyWrapper
from common.policy import EpsilonGreedyPolicy, BoltzmannQPolicy
from common.params import Parameters
from agents.Duelling_Double_DQN_PER_model import train_Duelling_Double_DQN_PER, Duelling_Double_DQN_PER_CartPole, Duelling_Double_DQN_PER_Atari

try:
	os.system("rm -rf ../logs/summary_DDD_PER_main")
except:
	pass

# initialise a graph in a session
tf.reset_default_graph()

mode = "CartPole"
# mode = "Atari"

if mode == "CartPole":
	env = MyWrapper(gym.make("CartPole-v0"))
	params = Parameters(mode="CartPole")
	main_model = Duelling_Double_DQN_PER_CartPole("DDD_PER_main", "naive", env, loss_fn="huber_loss")
	target_model = Duelling_Double_DQN_PER_CartPole("DDD_PER_target", "naive", env, loss_fn="huber_loss")
	replay_buffer = PrioritizedReplayBuffer(params.memory_size, alpha=params.prioritized_replay_alpha)
	Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
	# policy = EpsilonGreedyPolicy(Epsilon_fn=Epsilon)
	policy = BoltzmannQPolicy()
	Beta = AnnealingSchedule(start=params.prioritized_replay_beta_start, end=params.prioritized_replay_beta_end, decay_steps=params.decay_steps)
elif mode == "Atari":
	env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
	params = Parameters(mode="Atari")
	main_model = Duelling_Double_DQN_PER_Atari("DDD_PER_main", "naive", env, loss_fn="huber_loss")
	target_model = Duelling_Double_DQN_PER_Atari("DDD_PER_target", "naive", env, loss_fn="huber_loss")
	replay_buffer = PrioritizedReplayBuffer(params.memory_size, alpha=params.prioritized_replay_alpha)
	Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
	# policy = EpsilonGreedyPolicy(Epsilon_fn=Epsilon)
	policy = BoltzmannQPolicy()
	Beta = AnnealingSchedule(start=params.prioritized_replay_beta_start, end=params.prioritized_replay_beta_end, decay_steps=params.decay_steps)
else:
	print("Select 'mode' either 'Atari' or 'CartPole' !!")


all_rewards, losses = train_Duelling_Double_DQN_PER(main_model, target_model, env, replay_buffer, policy, Beta, params)

# temporal visualisation
# plt.subplot(2, 1, 1)
# plt.plot(all_rewards)
# plt.title("Score over time")
# plt.xlabel("Timestep")
# plt.ylabel("Score")
# plt.subplot(2, 1, 2)
# plt.plot(losses)
# plt.title("Loss over time")
# plt.xlabel("Timestep")
# plt.ylabel("Loss")
# plt.savefig("../logs/graphs/Duelling_Double_DQN_PER_train.png")

# plt.show()
