import gym
from gym.wrappers import Monitor

# env = gym.make("CartPole-v0")
env = gym.make("HalfCheetah-v2")
env = Monitor(env, "./video", video_callable=lambda episode_id: episode_id%10==0, force=True)

env.reset()

for t in range(100):
    s, r, done, _ = env.step(env.action_space.sample())
    if done:
        break
env.close()
