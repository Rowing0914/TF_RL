# run this from the terminal and make sure you are loading appropriate environment variables
# $ echo $LD_LIBRARY_PATH

import gym
from tf_rl.common.monitor import Monitor

video_dir = "./video/"
temp = 2

env = gym.make("HalfCheetah-v2")
env = Monitor(env, video_dir, force=True)

for ep in range(10):
    if ep == 0:
        print("recording")
        env.record_start()

    env.reset()
    done = False
    while not done:
        # env.render()
        action = env.action_space.sample()
        s, r, done, info = env.step(action)  # take a random action
    if ep == 0:
        env.record_end()