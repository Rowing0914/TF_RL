# run this from the terminal and make sure you are loading appropriate environment variables
# $ echo $LD_LIBRARY_PATH

import gym
from tf_rl.common.params import DDPG_ENV_LIST

for env_name, goal_score in DDPG_ENV_LIST.items():
    env = gym.make(env_name)
    env.reset()
    done = False
    cnt = 0
    while not done:
        env.render()
        s, r, done, info = env.step(env.action_space.sample())  # take a random action
        cnt += 1
    print("Env: {} ends at {}".format(env_name, cnt))
