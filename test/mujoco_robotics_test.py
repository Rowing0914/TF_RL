# run this from the terminal and make sure you are loading appropriate environment variables
# $ echo $LD_LIBRARY_PATH

import gym
from tf_rl.common.params import ROBOTICS_ENV_LIST

for env_name, goal_score in ROBOTICS_ENV_LIST.items():
	env = gym.make(env_name)
	state = env.reset()
	for t in range(50):
		# env.render()
		next_state, reward, done, _ = env.step(env.action_space.sample())
		print(next_state, reward, done, _)
		state = next_state