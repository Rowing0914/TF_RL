import tensorflow as tf
from tf_rl.common.memory_tf import ReplayBuffer as ReplayBuffer_tf
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.wrappers import wrap_deepmind, make_atari

env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
state = env.reset()
memory_tf = ReplayBuffer_tf(capacity=1000,
                      n_step=0,
                      act_shape=(),
                      obs_shape=state.shape,
                      obs_dtype=tf.int8,
                      checkpoint_dir="./tmp")
memory = ReplayBuffer(size=1000)
done = False
for t in range(100):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    memory.add(state, action, reward, next_state, done)
    memory_tf.add(state, action, reward, next_state, done)
    state = next_state
env.close()

print("=== test ===")
"""
Note:
    I have conducted the performance test where we repeat sampling from the Replay Buffer over 1000 times.
    And measured the exec time to compare Eager and Eager with Tf.function.

Result:
    without function: 9.03s
    with function:    1.13s
"""
import time

begin = time.time()
for _ in range(1000): memory_tf.sample_tf(batch_size=10)
print("with tf.function took : {:3f}s".format(time.time() - begin))

begin = time.time()
for _ in range(1000): memory_tf.sample(batch_size=10)
print("w/o tf. function took : {:3f}s".format(time.time() - begin))

begin = time.time()
for _ in range(1000): memory.sample(batch_size=10)
print("original memory took : {:3f}s".format(time.time() - begin))