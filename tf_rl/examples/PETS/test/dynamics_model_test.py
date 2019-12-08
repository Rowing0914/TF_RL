import tensorflow as tf
from eager.memory_eager import ReplayBuffer
from eager.dynamics_models_eager import HalfCheetahModel
import gym
import pybulletgym  # register PyBullet enviroments with open ai gym

tf.compat.v1.enable_eager_execution()

env = gym.make("HalfCheetahMuJoCoEnv-v0")
memory = ReplayBuffer(size=1000)

MODEL_IN, MODEL_OUT, ENSEMBLE_SIZE = env.observation_space.shape[0], 18, 2
BATCH_SIZE = 32

model = HalfCheetahModel(in_features=MODEL_IN,
                         out_features=MODEL_OUT,
                         ensemble_size=ENSEMBLE_SIZE)

state = env.reset()
for t in range(500):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    memory.add(state, action, reward, next_state, done)

states, actions, rewards, next_states, dones = memory.sample(batch_size=BATCH_SIZE)
print(states.shape, actions.shape, next_states.shape, dones.shape)
mean, logvar = model(states)
print(mean.shape, logvar.shape)