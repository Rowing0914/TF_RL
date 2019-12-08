import tensorflow as tf
import numpy as np
from optimizers import CEMOptimizer
import gym
import pybulletgym  # register PyBullet enviroments with open ai gym

tf.compat.v1.enable_eager_execution()

env = gym.make("HalfCheetahMuJoCoEnv-v0")

plan_hor = 5
dU = env.action_space.shape[0]
ac_lb, ac_ub = env.action_space.low, env.action_space.high
_compile_cost = lambda x: np.random.randn(x.shape[0], x.shape[1]).mean(axis=0)
init_mean = np.tile((ac_lb + ac_ub) / 2, [plan_hor])
init_var = np.tile(np.square(ac_ub - ac_lb) / 16, [plan_hor])

optimizer = CEMOptimizer(
    sol_dim=plan_hor * dU,
    lower_bound=np.tile(ac_lb, [plan_hor]),
    upper_bound=np.tile(ac_ub, [plan_hor]),
    cost_function=_compile_cost,
    max_iters=5,
    popsize=500,
    num_elites=50,
    alpha=0.1
)

mean = optimizer.obtain_solution(init_mean, init_var)
print(mean.shape)