import gym

env = gym.make("Swimmer-v2")
print(env.action_space.shape, env.observation_space.shape)
asdf
env.reset()
done = False
while not done:
    # env.render()
    action = env.action_space.sample()
    s, r, done, info = env.step(action)