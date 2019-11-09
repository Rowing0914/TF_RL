import gym
from tf_rl.common.monitor import Monitor

ENVS = [
    "Ant-v2",
    "HalfCheetah-v2",
    "Hopper-v2",
    "Humanoid-v2",
    # "Reacher-v2",
    # "Swimmer-v2",
    "Walker2d-v2"
]

DEFAULT = 250

for env_name in ENVS:
    env = gym.make(env_name)
    env = Monitor(env, "./video/{}".format(env_name), force=True)
    print(env_name)
    env.record_start()
    env.reset()
    done = False
    while not done:
        # env.render(mode="human", annotation_flg=False)
        s, r, done, i = env.step(env.action_space.sample())
    env.record_end()
    env.close()