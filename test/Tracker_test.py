import gym

from common.memory import ReplayBuffer
from common.utils import Tracker

env = gym.make("CartPole-v0")
memory = ReplayBuffer(1000)
tracker = Tracker(save_freq=100)

for i in range(100):
	state = env.reset()
	for t in range(100):
		# env.render()
		action = env.action_space.sample()
		next_state, reward, done, info = env.step(action)

		# memory format is: state, action, reward, next_state, done
		memory.add(state, action, reward, next_state, done)

		# format is: state, q_value, action, reward, done, loss, gradient
		tracker.add(state, 0.2, action, reward, done, 0.3, 0.01)

		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break
		state = next_state

env.close()
print(tracker.saved_cnt)
tracker._save_file()