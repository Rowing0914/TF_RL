# run this from the terminal and make sure you are loading appropriate environment variables
# $ echo $LD_LIBRARY_PATH

import gym
import numpy as np
import time
from tf_rl.common.params import ROBOTICS_ENV_LIST

for env_name, goal_score in ROBOTICS_ENV_LIST.items():
    env = gym.make(env_name)
    state = env.reset()
    for t in range(50):
        env.render()
        # time.sleep(10)
        if t % 2 == 0:
            action = np.ones(env.action_space.shape[0]) * 10
        else:
            action = np.ones(env.action_space.shape[0])

        print(action)
        next_state, reward, done, _ = env.step(action)
        print(next_state, reward, done, _)
        state = next_state
