from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.wrappers import wrap_deepmind, make_atari

env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
memory = ReplayBuffer(size=10000, traj_dir="./traj/")

for _ in range(10):
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        memory.add(state, action, reward, next_state, done)
        state = next_state
env.close()
print(len(memory))
memory.save_np(_save_id=0)

del memory
memory = ReplayBuffer(size=10000, recover_data=True, traj_dir="./traj/")
print(len(memory))