import gym
import tensorflow as tf
import pybulletgym  # register PyBullet enviroments with open ai gym
from eager.Agent_eager import Agent
from eager.MPC_eager import MPC

tf.compat.v1.enable_eager_execution()

env = gym.make("HalfCheetahMuJoCoEnv-v0")
policy = MPC(env=env)
agent = Agent(env=env, horizon=4, policy=policy)
trajs = agent.sample()

obs = trajs["obs"]
ac = trajs["ac"]
reward_sum = trajs["reward_sum"]
rewards = trajs["rewards"]
print(obs.shape, ac.shape, reward_sum.shape, rewards.shape)
