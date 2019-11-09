import gym
import humanoid_maze

env = gym.make('HumanoidMaze-v0')

env.reset()
for _ in range(2000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()