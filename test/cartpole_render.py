import gym
from tf_rl.common.wrappers import CartPole_Pixel

env = CartPole_Pixel(gym.make('CartPole-v0'))
for ep in range(2):
    env.reset()
    for t in range(100):
        o, r, done, _ = env.step(env.action_space.sample())
        print(o.shape, o.min(), o.max())
        if done:
            break
env.close()
