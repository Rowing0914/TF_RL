from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.wrappers import wrap_deepmind, make_atari

size = 1000

env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
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

for _ in range(size): memory.sample(batch_size=10)
