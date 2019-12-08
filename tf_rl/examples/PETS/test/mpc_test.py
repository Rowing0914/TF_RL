from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tensorflow as tf
import pybulletgym
from eager.MPC_eager import MPC

tf.compat.v1.enable_eager_execution()

env = gym.make("HalfCheetahMuJoCoEnv-v0")

controller = MPC(env=env)

