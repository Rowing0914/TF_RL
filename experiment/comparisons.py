import gym
import tensorflow as tf
import matplotlib.pyplot as plt

from common.memory import ReplayBuffer
from common.utils import AnnealingSchedule
from common.wrappers_Atari import make_atari, wrap_deepmind
from agents.Double_DQN_model import train_Double_DQN
from agents.Duelling_DQN_model import Duelling_DQN_Atari, Duelling_DQN_CartPole
from agents.DQN_model import train_DQN, DQN_CartPole, DQN_Atari, Parameters


mode = "CartPole"
# mode = "Atari"

# we iterate through the all models implemented in agents
if mode == "CartPole":
	models = [DQN_CartPole, Duelling_DQN_CartPole, DQN_CartPole]
	models_name = ["DQN", "Duelling_DQN", "Double_DQN"]
elif mode == "Atari":
	models = [DQN_Atari, Duelling_DQN_Atari, DQN_Atari]
	models_name = ["DQN", "Duelling_DQN", "Double_DQN"]


for model, model_name in zip(models, models_name):
	print("===== MODEL: {} TRAINING BEGIN =====".format(model_name))
	# initialise a graph in a session
	tf.reset_default_graph()

	if model_name == "Duelling_DQN":
		if mode == "CartPole":
			env = gym.make("CartPole-v0")
			params = Parameters(mode="CartPole")
			main_model = Duelling_DQN_CartPole("main", "max", env)
			target_model = Duelling_DQN_CartPole("target", "max", env)
			replay_buffer = ReplayBuffer(params.memory_size)
			Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end,
										decay_steps=params.decay_steps)
		elif mode == "Atari":
			env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
			params = Parameters(mode="Atari")
			main_model = Duelling_DQN_Atari("main", "naive", env)
			target_model = Duelling_DQN_Atari("target", "naive", env)
			replay_buffer = ReplayBuffer(params.memory_size)
			Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end,
										decay_steps=params.decay_steps)
		else:
			print("Select 'mode' either 'Atari' or 'CartPole' !!")
	else:
		if mode == "CartPole":
			env = gym.make("CartPole-v0")
			params = Parameters(mode="CartPole")
			main_model = model("main", env)
			target_model = model("target", env)
			replay_buffer = ReplayBuffer(params.memory_size)
			Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
		elif mode == "Atari":
			env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
			params = Parameters(mode="Atari")
			main_model = model("main", env)
			target_model = model("target", env)
			replay_buffer = ReplayBuffer(params.memory_size)
			Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
		else:
			print("Select 'mode' either 'Atari' or 'CartPole' !!")


	# in case of Double DQN, we use different training method from DQN
	if model_name == "Double_DQN":
		all_rewards, losses = train_Double_DQN(main_model, target_model, env, replay_buffer, Epsilon, params)
	else:
		all_rewards, losses = train_DQN(main_model, target_model, env, replay_buffer, Epsilon, params)


	# temporal visualisation
	plt.subplot(2, 1, 1)
	plt.plot(all_rewards, label=model_name)
	plt.title("Score over time")
	plt.xlabel("Timestep")
	plt.ylabel("Score")

	plt.subplot(2, 1, 2)
	plt.plot(losses, label=model_name)
	plt.title("Loss over time")
	plt.xlabel("Timestep")
	plt.ylabel("Loss")

plt.legend()
plt.show()