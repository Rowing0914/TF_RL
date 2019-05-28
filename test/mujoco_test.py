# run this from the terminal and make sure you are loading appropriate environment variables
# $ echo $LD_LIBRARY_PATH

import gym
from tf_rl.common.params import DDPG_ENV_LIST

for env_name in DDPG_ENV_LIST:
	env = gym.make(env_name)
	env.reset()
	for _ in range(100):
		env.render()
		env.step(env.action_space.sample()) # take a random action