import gym
import matplotlib.pyplot as plt
import tensorflow as tf
from common.memory import ReplayBuffer
from common.utils import AnnealingEpsilon
from DQN_model import train_DQN, DQN_CartPole, DQN_Atari, Parameters


# initialise a graph in a session
tf.reset_default_graph()

mode = "CartPole"

if mode == "CartPole":
    env = gym.make("CartPole-v0")
    params = Parameters(mode="CartPole")
    main_model = DQN_CartPole("main", env)
    target_model = DQN_CartPole("target", env)
    replay_buffer = ReplayBuffer(params.memory_size)
    Epsilon = AnnealingEpsilon(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
elif mode == "Atari":
    env = gym.make("PongNoFrameskip-v0")
    params = Parameters(mode="Atari")
    main_model = DQN_Atari("main", env)
    target_model = DQN_Atari("target", env)
    replay_buffer = ReplayBuffer(params.memory_size)
    Epsilon = AnnealingEpsilon(start=params.epsilon_start, end=params.epsilon_end, decay_steps=params.decay_steps)
else:
    print("Select 'mode' either 'Atari' or 'CartPole' !!")


all_rewards, losses = train_DQN(main_model, target_model, env, replay_buffer, Epsilon, params)

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