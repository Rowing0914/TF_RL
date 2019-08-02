import gym

ENVS = [
    "Ant-v2",
    "HalfCheetah-v2",
    "Hopper-v2",
    "Humanoid-v2",
    "Reacher-v2",
    "Swimmer-v2",
    "Walker2d-v2"
]

for env_name in ENVS:
    env = gym.make(env_name)
    print(env_name, env.action_space.shape, env.observation_space.shape)
    env.close()
