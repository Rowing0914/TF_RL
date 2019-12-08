import gym, argparse
from gym.wrappers import Monitor
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from run_expert import expert_play, OBS_FILE_PATH, ACT_FILE_PATH, ENV_NAME, DQN_Agent_model
from load_policy import DQN_Agent

OBSERVATION_SPACE = (4,) # for CartPole-v0
NB_ACTIONS = 2           # for CartPole-v0
EPOCHS = 5
BATCH_SIZE = 32
NUM_EPISODES = 10
BETA = 1

def create_model():
  """
  Using the same architecture as the one of DQN Agent

  Return:
    Keras Sequential compiled model
  """
  model = Sequential()
  model.add(Flatten(input_shape=(1,) + OBSERVATION_SPACE))
  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dense(NB_ACTIONS, activation='softmax'))
  print(model.summary())

  # For a classification problem of actions
  model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  return model

def load_dataset():
  """
  Loading the dataset which is the demo of the expert in `run_expert.py`.
  
  Returns: 
    X: Observations of the game
    Y: Actions
  """
  X = np.load(OBS_FILE_PATH)
  Y = np.load(ACT_FILE_PATH)
  print(X.shape, Y.shape)

  X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
  Y = np_utils.to_categorical(Y)
  return X, Y

def train(X, Y, model):
  """
  Train the model with the dataset,
  then save it in h5 format after the training

  Returns:
    trained model
  """
  model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE)
  model.save_weights("./weights/DAgger_weights.h5")
  model.load_weights("./weights/DAgger_weights.h5")
  return model

def prep_dataset(observations, actions):
  """
  Reshape and format the training dataset

  Args:
    observations: a list of observations in an episode
    actions:  a list of actions in an episode

  Returns:
    X: Observations of the game
    Y: Actions
  """
  X = np.array(observations)
  X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
  Y = np.array(actions)
  Y = np_utils.to_categorical(Y)
  return X, Y

def DAgger(expert, model, env, type_policy="deterministic"):
  """
  This is the implemnetation of DAgger algorithm.
  While the agent plays with the environmen, it remembers the encountered states and actions in a single episode
  Then when an episode ends, it will update the model with the collected data

  Args:
    model: Keras trained model
    env: Open AI arcade game
    type_policy: type of policy
      - deterministic => a model chooses the action having the maximum value of its predicting result
      - stochastic    => a model randomly chooses the action based on its predicting result
  """
  rewards = 0
  for i_episode in range(NUM_EPISODES):
    observation = env.reset()
    env.render()
    done = False
    observations, actions = list(), list()
    while not done:
      action_agent = model.predict(observation.reshape(1,1,4))[0]
      if type_policy == "deterministic":
        # deterministic policy
        action_agent = np.argmax(action_agent)
      elif type_policy == "stochastic":
        # stochastic policy
        action_agent = np.random.choice(np.arange(NB_ACTIONS), p=action_agent)

      # ===== if you want to use an expert for collecting the dataset, open this part! ===== 
      # we assume that beta is 1, so we only rely on the expert for collecting the dataset
      action_expert = expert.forward(observation)
      action = BETA*action_expert + (1 - BETA)*action_agent

      observation, reward, done, info = env.step(action)
      rewards += reward

      observations.append(observation)
      actions.append(action)

      if done:
        print("Score: ", rewards)
        rewards = 0
        X, Y = prep_dataset(observations, actions)
        model = train(X, Y, model)
        break

def DAgger_play(model, env, type_policy="deterministic"):
  """
  This is the implemnetation of DAgger algorithm.
  While the agent plays with the environmen, it remembers the encountered states and actions in a single episode
  Then when an episode ends, it will update the model with the collected data

  Args:
    model: Keras trained model
    env: Open AI arcade game
    type_policy: type of policy
      - deterministic => a model chooses the action having the maximum value of its predicting result
      - stochastic    => a model randomly chooses the action based on its predicting result
  """
  rewards = 0
  for i_episode in range(NUM_EPISODES):
    observation = env.reset()
    env.render()
    done = False
    while not done:
      action = model.predict(observation.reshape(1,1,4))[0]
      if type_policy == "deterministic":
        # deterministic policy
        action = np.argmax(action)
      elif type_policy == "stochastic":
        # stochastic policy
        action = np.random.choice(np.arange(NB_ACTIONS), p=action)

      observation, reward, done, info = env.step(action)
      rewards += reward

      if done:
        print("Score: ", rewards)
        rewards = 0
        break

def random_play(env):
  for i_episode in range(NUM_EPISODES):
      observation = env.reset()
      done = False
      t = 0
      rewards = 0
      while not done:
          env.render()
          action = env.action_space.sample()
          observation, reward, done, info = env.step(action)
          rewards += reward
          if done:
              print("Score: {0}".format(rewards))
              break
          t += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weights', type=str, default="./weights/DAgger_weights.h5")
    parser.add_argument('--type_policy', type=str, default="deterministic")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--random', action="store_true")
    args = parser.parse_args()
    env = gym.make(ENV_NAME)
    expert = DQN_Agent(DQN_Agent_model)

    if args.test:
      model = create_model()
      model.load_weights("./weights/DAgger_weights.h5")
      DAgger(expert, model, env, args.type_policy)
      DAgger_play(model, env, args.type_policy)
    elif args.random:
      random_play(env)
    else:
      model = create_model()
      X, Y = load_dataset()
      model = train(X, Y, model)
      DAgger(expert, model, env, args.type_policy)
      DAgger_play(model, env, args.type_policy)