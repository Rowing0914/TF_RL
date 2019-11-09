import tensorflow as tf
from tf_rl.common.memory_tf import ReplayBuffer
from tf_rl.common.wrappers import wrap_deepmind, make_atari

env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
state = env.reset()
memory = ReplayBuffer(capacity=100,
                      n_step=0,
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
                      n_step=0,
                      act_shape=(),
                      obs_shape=state.shape,
                      obs_dtype=tf.int8,
                      checkpoint_dir="./tmp")
print(len(memory))
obs, action, next_obs, reward, done = memory.sample(batch_size=10)
print(obs.shape, action.shape, next_obs.shape, reward.shape, done.shape)
