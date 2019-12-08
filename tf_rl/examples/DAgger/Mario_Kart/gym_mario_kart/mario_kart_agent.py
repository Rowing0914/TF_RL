#!/bin/python
"""
Action(1x5 vector):
- Joystick X-axis (L/R): value from -80 to 80
- Joystick Y-axis (U/D): value from -80 to 80
- A Button: value of 0 or 1
- B Button: value of 0 or 1
- RB Button: value of 0 or 1
"""

# to remove unnecessary path from python path
# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import gym, gym_mupen64plus
from numpy import genfromtxt

def load_demo():
    return genfromtxt("play_log.csv",delimiter=",")

def imitate(env, demo):
    env.render()
    print('Game started')
    for i in range(demo.shape[0]):
        (obs, rew, end, info) = env.step(demo[i].tolist())
        env.render()
    env.close()

def main(env):
    env.render()
    print('Game started')

    for i in range(18):
        (obs, rew, end, info) = env.step([0, 0, 0, 0, 0]) # NOOP until green light
        env.render()

    for i in range(20):
        (obs, rew, end, info) = env.step([0, 0, 1, 0, 0]) # Drive straight
        env.render()

    for i in range(20):
        (obs, rew, end, info) = env.step([50, 0, 1, 0, 0]) # Drive straight
        env.render()

    for i in range(20):
        (obs, rew, end, info) = env.step([-50, 0, 1, 0, 0]) # Drive straight
        env.render()

    env.close()

if __name__ == '__main__':
    env = gym.make('Mario-Kart-Luigi-Raceway-v0')
    env.reset()
    imitate(env, load_demo())
    # main(env)