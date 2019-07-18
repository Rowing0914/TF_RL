import gym
from gym.wrappers import Monitor
env = gym.make('HalfCheetah-v2')
env = Monitor(env, './video', force=True)
env.reset()
while True:
	env.render()
	obs, r, done, info = env.step([0,0,0,0,0,0])
	if done: break