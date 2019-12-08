import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
# from run_expert import OBSERVATION_SPACE, NB_ACTIONS

OBSERVATION_SPACE = (4,) # for CartPole-v0
NB_ACTIONS = 2           # for CartPole-v0

def DQN_Agent(weights_filename="./models/dqn_CartPole-v0_weights.h5f"):
  """
  1. Initialise and compile DQN Agent using Keras sequential API
  2. Load the trained agent on the OpenAI game Env(default is CartPole-v0)

  Args:
    weights_filename: weights file location

  Returns:
    dqn: the trained agent
  """

  model = Sequential()
  model.add(Flatten(input_shape=(1,) + OBSERVATION_SPACE))
  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dense(NB_ACTIONS, activation='linear'))
  print(model.summary())

  memory = SequentialMemory(limit=50000, window_length=1)
  policy = BoltzmannQPolicy()
  dqn = DQNAgent(model=model, nb_actions=NB_ACTIONS, memory=memory, nb_steps_warmup=10,
                 target_model_update=1e-2, policy=policy)
  dqn.compile(Adam(lr=1e-3), metrics=['mae'])
  dqn.load_weights(weights_filename)
  return dqn

def Duelling_DQN_Agent(weights_filename="./models/duel_dqn_CartPole-v0_weights.h5f"):
  """
  1. Initialise and compile Duelling DQN Agent using Keras sequential API
  2. Load the trained agent on the OpenAI game Env(default is CartPole-v0)

  Args:
    weights_filename: weights file location

  Returns:
    dqn: the trained Duelling DQN Agent
  """

  model = Sequential()
  model.add(Flatten(input_shape=(1,) + OBSERVATION_SPACE))
  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dense(NB_ACTIONS, activation='linear'))
  print(model.summary())
  memory = SequentialMemory(limit=50000, window_length=1)
  policy = BoltzmannQPolicy()
  dqn = DQNAgent(model=model, nb_actions=NB_ACTIONS, memory=memory, nb_steps_warmup=10,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
  dqn.compile(Adam(lr=1e-3), metrics=['mae'])
  dqn.load_weights(weights_filename)
  return dqn