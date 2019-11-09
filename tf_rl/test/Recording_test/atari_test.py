import itertools
from tf_rl.common.monitor import Monitor
from tf_rl.common.wrappers import wrap_deepmind, make_atari

env_name = "PongNoFrameskip-v4"

env = wrap_deepmind(make_atari(env_name))
env = Monitor(env=env, directory="./video/{}".format(env_name), force=True)
env.record_start()
state = env.reset()
for t in itertools.count():
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    state = next_state
    if done:
        break
print("End at {}".format(t+1))
env.record_end()
env.close()
