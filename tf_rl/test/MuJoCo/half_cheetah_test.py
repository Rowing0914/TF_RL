# run this from the terminal and make sure you are loading appropriate environment variables
# $ echo $LD_LIBRARY_PATH

import gym

env = gym.make("HalfCheetah-v2")
env.reset()
done = False
distance, actions = list(), list()
while not done:
    # env.render()
    action = env.action_space.sample()
    s, r, done, info = env.step(action)  # take a random action
    actions.append(action.mean()**2)
    distance.append(info['reward_run'])

import matplotlib.pyplot as plt

plt.subplot(211)
plt.hist(distance, bins=100)
plt.xlabel("Distance")
plt.ylabel("Occurence")

plt.subplot(212)
plt.hist(actions, bins=100)
plt.xlabel("Action")
plt.ylabel("Occurence")
plt.show()