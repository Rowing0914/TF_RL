"""
Description:
    Test of how we capture the video and how to select when to capture the video.
    Because shooting a video certainly harms the speed of training so that we don't wanna make it happen in training.

    Setting lambda func to the `video_callable`, we can set the func to select the time when we capture the video.

"""

import gym
from gym.wrappers import Monitor
import tensorflow as tf
from tf_rl.common.utils import eager_setup

eager_setup()

global_ts = tf.compat.v1.train.create_global_step()

# env = gym.make("CartPole-v0")
env = gym.make("HalfCheetah-v2")
env = Monitor(env, "./video", video_callable= lambda ep_id: True if tf.compat.v1.train.get_global_step().numpy()==2000 else False, force=True)

for ep in range(10):
    print(ep)
    env.reset()
    while True:
        global_ts.assign_add(1)
        s, r, done, _ = env.step(env.action_space.sample())
        if done:
            break
env.close()
