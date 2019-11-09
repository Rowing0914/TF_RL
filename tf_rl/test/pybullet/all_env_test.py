import gym
from gym.wrappers.monitor import Monitor
from tf_rl.env.pybullet.env_list import ENVS

for key, env_name in ENVS.items():
    print(env_name)
    env = gym.make(env_name)
    env = Monitor(env=env, directory="./video/{}".format(key), force=True)

    state = env.reset()
    for t in range(100):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            break

    env.close()
