import gym
from gym_extensions.continuous import mujoco

# available env list: https://github.com/Rowing0914/gym-extensions/blob/mujoco200/tests/all_tests.py
env = gym.make("PusherMovingGoal-v1")

env.reset()
for _ in range(100):
    env.render()
    s, r, d, i = env.step(env.action_space.sample()) # take a random action
    print(s.shape, r, d, i)
env.close()