from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.wrappers import wrap_deepmind, make_atari

env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
memory = ReplayBuffer(1000, n_step=5, flg_seq=True)

print("Memory contains {0} timesteps".format(len(memory)))

for i in range(5):
    state = env.reset()
    for t in range(1000):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        memory.add(state, action, reward, next_state, done)
        state = next_state
        if done: break

env.close()

print("Memory contains {0} timesteps".format(len(memory)))
states, actions, rewards, next_states, dones = memory.sample(batch_size=10)
print(states.shape, state.shape)

for _ in range(1000): memory.sample(batch_size=10)
