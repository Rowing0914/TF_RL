import gym
import tensorflow as tf
import numpy as np
import pybulletgym  # register PyBullet enviroments with open ai gym

from eager.Agent_eager import Agent
from eager.MPC_eager import MPC

tf.compat.v1.enable_eager_execution()

NINIT_ROLLOUTS = 1
NEVAL = 1
TASK_HORIZON = 1000
NTRAIN_ITERS = 300
NROLLOUTS_PER_ITER = 1
PLAN_HOR = 30

env = gym.make("HalfCheetahMuJoCoEnv-v0")
policy = MPC(env=env)
agent = Agent(env=env, horizon=TASK_HORIZON, policy=policy)

traj_obs, traj_acs, traj_rets, traj_rews = [], [], [], []

# Perform initial rollouts
samples = []
print("=== Initial Rollout ===")
for i in range(NINIT_ROLLOUTS):
    samples.append(agent.sample())
    traj_obs.append(samples[-1]["obs"])
    traj_acs.append(samples[-1]["ac"])
    traj_rews.append(samples[-1]["rewards"])

print("=== Initial Training ===")
if NINIT_ROLLOUTS > 0:
    agent.policy.train(
        [sample["obs"] for sample in samples],
        [sample["ac"] for sample in samples],
        [sample["rewards"] for sample in samples]
    )
    agent.has_been_trained = True

print("=== Training Loop ===")
# Training loop
for i in range(NTRAIN_ITERS):
    samples = []

    for j in range(max(NROLLOUTS_PER_ITER, NEVAL)):
        samples.append(agent.sample())

    traj_obs.extend([sample["obs"] for sample in samples[:NROLLOUTS_PER_ITER]])
    traj_acs.extend([sample["ac"] for sample in samples[:NROLLOUTS_PER_ITER]])
    traj_rets.extend([sample["reward_sum"] for sample in samples[:NEVAL]])
    traj_rews.extend([sample["rewards"] for sample in samples[:NROLLOUTS_PER_ITER]])
    samples = samples[:NROLLOUTS_PER_ITER]

    if i < NTRAIN_ITERS - 1:
        loss = policy.train(
            [sample["obs"] for sample in samples],
            [sample["ac"] for sample in samples],
            [sample["rewards"] for sample in samples]
        )
        print("| Iter: {} | Loss: {:.3f} | Rewards: {:.3f}".format(
            i+1, loss, np.mean([sample["reward_sum"] for sample in samples[:NEVAL]]))
              )
