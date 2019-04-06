import gym
import tensorflow as tf
import matplotlib.pyplot as plt

from common.memory import PrioritizedReplayBuffer
from common.utils import AnnealingSchedule
from common.wrappers_Atari import make_atari, wrap_deepmind
from common.policy import EpsilonGreedyPolicy, BoltzmannQPolicy
from agents.DQN_model import Parameters
from agents.DQN_PER_model import train_DQN_PER, DQN_PER_CartPole, DQN_PER_Atari


# initialise a graph in a session
tf.reset_default_graph()

mode = "CartPole"
# mode = "Atari"

if mode == "CartPole":
	env = gym.make("CartPole-v0")
	params = Parameters(mode="CartPole")
	main_model = DQN_PER_CartPole("main", env, loss_fn="huber_loss")
	target_model = DQN_PER_CartPole("target", env, loss_fn="huber_loss")
	replay_buffer = PrioritizedReplayBuffer(params.memory_size, alpha=params.prioritized_replay_alpha)
	Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
	# policy = EpsilonGreedyPolicy(Epsilon_fn=Epsilon)
	policy = BoltzmannQPolicy()
	Beta = AnnealingSchedule(start=params.prioritized_replay_beta_start, end=params.prioritized_replay_beta_end, decay_steps=params.decay_steps)
elif mode == "Atari":
	env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
	params = Parameters(mode="Atari")
	main_model = DQN_PER_Atari("main", env, loss_fn="huber_loss")
	target_model = DQN_PER_Atari("target", env, loss_fn="huber_loss")
	replay_buffer = PrioritizedReplayBuffer(params.memory_size, alpha=params.prioritized_replay_alpha)
	Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
	# policy = EpsilonGreedyPolicy(Epsilon_fn=Epsilon)
	policy = BoltzmannQPolicy()
	Beta = AnnealingSchedule(start=params.prioritized_replay_beta_start, end=params.prioritized_replay_beta_end, decay_steps=params.decay_steps)
else:
	print("Select 'mode' either 'Atari' or 'CartPole' !!")


all_rewards, losses = train_DQN_PER(main_model, target_model, env, replay_buffer, policy, Beta, params)

# temporal visualisation
plt.subplot(2, 1, 1)
plt.plot(all_rewards)
plt.title("Score over time")
plt.xlabel("Timestep")
plt.ylabel("Score")

plt.subplot(2, 1, 2)
plt.plot(losses)
plt.title("Loss over time")
plt.xlabel("Timestep")
plt.ylabel("Loss")
plt.show()