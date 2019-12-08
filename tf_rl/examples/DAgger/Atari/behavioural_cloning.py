import gym, argparse
from gym.wrappers import Monitor
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from run_expert import expert_play, OBS_FILE_PATH, ACT_FILE_PATH, ENV_NAME

OBSERVATION_SPACE = (4,) # for CartPole-v0
NB_ACTIONS = 2           # for CartPole-v0
EPOCHS = 50
BATCH_SIZE = 32
NUM_EPISODES = 100

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
  model.save_weights("./weights/weights.h5")
  model.load_weights("./weights/weights.h5")
  return model

def play(model, env, type_policy="deterministic"):
  """
  Let a model play with the game(OpenAI env)
  For now, we only aggregate rewards in a episode and print a score in a episode

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
      # predict the action based on the curret state
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
      while not done:
          env.render()
          action = env.action_space.sample()
          observation, reward, done, info = env.step(action)
          if done:
              print("Episode finished after {} timesteps".format(t+1))
              break
          t += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weights', type=str, default="./weights/weights.h5")
    parser.add_argument('--type_policy', type=str, default="deterministic")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--random', action="store_true")
    args = parser.parse_args()
    env = gym.make(ENV_NAME)

    if args.test:
      model = create_model()
      model.load_weights("./weights/weights.h5")
      play(model, env, args.type_policy)
    elif args.random:
      random_play(env)
    else:
      model = create_model()
      X, Y = load_dataset()
      model = train(X, Y, model)
      play(model, env, args.type_policy)