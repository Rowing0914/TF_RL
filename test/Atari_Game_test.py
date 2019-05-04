"""
Open AI Gym Env Playground

URL: https://gym.openai.com/docs/
Author: Norio Kosaka

"""

import gym
from tf_rl.common.wrappers import wrap_deepmind, make_atari

# env = gym.make("BreakoutDeterministic-v4")
env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))

for i in range(10):
    state = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        state = next_state

env.close()
