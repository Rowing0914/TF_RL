import tensorflow as tf
from tf_rl.common.memory_tf import ReplayBuffer
from tf_rl.common.wrappers import wrap_deepmind, make_atari

env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
state = env.reset()
memory = ReplayBuffer(capacity=100,
                      n_step=3,
                      act_shape=(),
                      obs_shape=state.shape,
                      obs_dtype=tf.int8,
                      checkpoint_dir="./tmp")

done = False
for t in range(100):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    memory.add(state, action, reward, next_state, done)
    state = next_state
env.close()
print(len(memory))
obs, action, next_obs, reward, done = memory.sample(batch_size=10)
print(obs.shape, action.shape, next_obs.shape, reward.shape, done.shape)
path = memory.save()

# recover phase
print("=== Recover Phase ===")
del memory
memory = ReplayBuffer(capacity=100,
                      n_step=3,
                      act_shape=(),
                      obs_shape=state.shape,
                      obs_dtype=tf.int8,
                      checkpoint_dir="./tmp")
print(len(memory))
obs, action, next_obs, reward, done = memory.sample(batch_size=10)
print(obs.shape, action.shape, next_obs.shape, reward.shape, done.shape)

print("=== test ===")
"""
Note:
    I have conducted the performance test where we repeat sampling from the Replay Buffer over 1000 times.
    And measured the exec time to compare Eager and Eager with Tf.function.

Result:
    without function: 18.181384s
    with function:    5.03s
"""
import time

begin = time.time()
for _ in range(1000): memory.sample(batch_size=10)
print("took : {:3f}s".format(time.time() - begin))
