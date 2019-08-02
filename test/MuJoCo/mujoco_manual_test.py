# run this from the terminal and make sure you are loading appropriate environment variables
# $ echo $LD_LIBRARY_PATH

import gym

env = gym.make("HalfCheetah-v2")
env.reset()
done = False
while not done:
    env.render()
    # action = env.action_space.sample()
    action = [-1.0, 1.0, 1.0, -1.0, 1.0, -1.0]
    # print(action)
    s, r, done, info = env.step(action)  # take a random action
