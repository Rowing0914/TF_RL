from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.wrappers import wrap_deepmind, make_atari

# env = gym.make("CartPole-v0")
env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
memory = ReplayBuffer(500_000)

print("Memory contains {0} timesteps".format(len(memory)))

for i in range(10000):
	state = env.reset()
	for t in range(1000):
		# env.render()
		action = env.action_space.sample()
		next_state, reward, done, info = env.step(action)
		memory.add(state, action, reward, next_state, done)

		if done:
			print("Episode finished after {} timesteps".format(t + 1))
			print("Memory contains {0} timesteps".format(len(memory)))
			break
	state = next_state

env.close()

print("Memory contains {0} timesteps".format(len(memory)))
# state, action, reward, next_state, done, weights, indices = memory.sample(batch_size=10, beta=Beta.get_value(1))
print(memory.sample(1))
