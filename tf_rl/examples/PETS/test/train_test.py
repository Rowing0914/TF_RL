import gym
import tensorflow as tf
import pybulletgym  # register PyBullet enviroments with open ai gym

from eager.Agent_eager import Agent
from eager.MPC_eager import MPC

tf.compat.v1.enable_eager_execution()

ninit_rollouts = 3
nrollouts_per_iter = 5
ntrain_iters = 5
neval = 1

env = gym.make("HalfCheetahMuJoCoEnv-v0")
policy = MPC(env=env)
agent = Agent(env=env, horizon=4, policy=policy)

traj_obs, traj_acs, traj_rets, traj_rews = [], [], [], []

# Perform initial rollouts
samples = []
for i in range(ninit_rollouts):
    samples.append(agent.sample())
    traj_obs.append(samples[-1]["obs"])
    traj_acs.append(samples[-1]["ac"])
    traj_rews.append(samples[-1]["rewards"])

if ninit_rollouts > 0:
    agent.policy.train(
        [sample["obs"] for sample in samples],
        [sample["ac"] for sample in samples],
        [sample["rewards"] for sample in samples]
    )

# Training loop
for i in range(ntrain_iters):
    print("####################################################################")
    print("Starting training iteration %d." % (i + 1))

    samples = []

    for j in range(max(nrollouts_per_iter, neval)):
        samples.append(agent.sample())

    print("Rewards obtained:", [sample["reward_sum"] for sample in samples[:neval]])
    traj_obs.extend([sample["obs"] for sample in samples[:nrollouts_per_iter]])
    traj_acs.extend([sample["ac"] for sample in samples[:nrollouts_per_iter]])
    traj_rets.extend([sample["reward_sum"] for sample in samples[:neval]])
    traj_rews.extend([sample["rewards"] for sample in samples[:nrollouts_per_iter]])
    samples = samples[:nrollouts_per_iter]

    if i < ntrain_iters - 1:
        policy.train(
            [sample["obs"] for sample in samples],
            [sample["ac"] for sample in samples],
            [sample["rewards"] for sample in samples]
        )
