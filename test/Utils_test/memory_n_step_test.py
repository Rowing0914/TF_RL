from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.wrappers import wrap_deepmind, make_atari

env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
memory = ReplayBuffer(1000, n_step=5, flg_seq=True)

print("Memory contains {0} timesteps".format(len(memory)))

for i in range(1):
    state = env.reset()
    for t in range(1000):
        # env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        memory.add(state, action, reward, next_state, done)
        state = next_state

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            print("Memory contains {0} timesteps".format(len(memory)))
            break

env.close()

print("Memory contains {0} timesteps".format(len(memory)))
state, action, reward, next_state, done = memory.sample(batch_size=10)
print(state.shape, action.shape)
