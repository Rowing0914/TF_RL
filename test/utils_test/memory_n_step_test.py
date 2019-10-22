import gym
from tf_rl.common.memory import ReplayBuffer
size = 1000
env = gym.make("CartPole-v0")
memory = ReplayBuffer(size, n_step=5, flg_seq=True)

print("Memory contains {0} timesteps".format(len(memory)))

state = env.reset()
action = env.action_space.sample()
next_state, reward, done, info = env.step(action)
env.close()

for _ in range(size):
    memory.add(state, action, reward, next_state, done)
print(len(memory))
memory.save()

print("Memory contains {0} timesteps".format(len(memory)))
states, actions, rewards, next_states, dones = memory.sample(batch_size=10)
print(states.shape, state.shape)

for _ in range(1000): memory.sample(batch_size=10)
