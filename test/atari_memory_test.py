from tf_rl.common.wrappers import wrap_deepmind, make_atari
from tf_rl.common.memory import ReplayBuffer

# for env_name , goal_score in ENV_LIST_NIPS.items():
env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
memory_size = 1000
replay_buffer = ReplayBuffer(memory_size)
state = env.reset()
for t in range(memory_size):
    env.render()
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    replay_buffer.add(state, action, reward, next_state, done)
    state = next_state
    if t % 10000 == 0:
        print(t)
env.close()
# replay_buffer.save(dir="./buffer.json")
