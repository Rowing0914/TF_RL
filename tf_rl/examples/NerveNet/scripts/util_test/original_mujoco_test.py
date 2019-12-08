import gym

# env_name = "Ant-v2"
env_name = "HalfCheetah-v2"

env = gym.make(env_name)
env.reset()
print(env.action_space.shape[0])

while True:
    for i in range(1000):
        env.render(annotation_flg=True)
        action = env.action_space.sample()
        env.step(action)