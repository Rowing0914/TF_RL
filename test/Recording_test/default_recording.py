import gym
from gym.wrappers.monitor import Monitor

env = gym.make('CartPole-v0')
env = Monitor(env=env, directory="./video/cartpole", force=True)

state = env.reset()
for t in range(100):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    if done:
        break

env.close()
