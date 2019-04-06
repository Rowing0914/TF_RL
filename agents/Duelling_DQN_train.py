import gym
import tensorflow as tf
import matplotlib.pyplot as plt

from common.memory import ReplayBuffer
from common.utils import AnnealingSchedule
from common.wrappers_Atari import make_atari, wrap_deepmind
from common.policy import EpsilonGreedyPolicy, BoltzmannQPolicy
from agents.DQN_model import train_DQN, Parameters
from agents.Duelling_DQN_model import Duelling_DQN_CartPole, Duelling_DQN_Atari

# initialise a graph in a session
tf.reset_default_graph()

mode = "CartPole"
# mode = "Atari"

if mode == "CartPole":
    env = gym.make("CartPole-v0")
    params = Parameters(mode="CartPole")
    main_model = Duelling_DQN_CartPole("main", "max", env, "huber_loss")
    target_model = Duelling_DQN_CartPole("target", "max", env, "huber_loss")
    replay_buffer = ReplayBuffer(params.memory_size)
    Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
    # policy = EpsilonGreedyPolicy(Epsilon_fn=Epsilon)
    policy = BoltzmannQPolicy()
elif mode == "Atari":
    env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
    params = Parameters(mode="Atari")
    main_model = Duelling_DQN_Atari("main", "naive", env, "huber_loss")
    target_model = Duelling_DQN_Atari("target", "naive", env, "huber_loss")
    replay_buffer = ReplayBuffer(params.memory_size)
    Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
    # policy = EpsilonGreedyPolicy(Epsilon_fn=Epsilon)
    policy = BoltzmannQPolicy()
else:
    print("Select 'mode' either 'Atari' or 'CartPole' !!")


all_rewards, losses = train_DQN(main_model, target_model, env, replay_buffer, policy, params)

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