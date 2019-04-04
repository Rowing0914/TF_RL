"""
Open AI Gym Env Playground

URL: https://gym.openai.com/docs/
Author: Norio Kosaka

"""

import gym
import time

env = gym.make("BreakoutDeterministic-v4")

for i in range(10):
    state = env.reset()
    for t in range(100):
        env.render()
        print(state)
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        state = next_state

env.close()
