import gym
from tf_rl.common.monitor import Monitor

env = gym.make('CartPole-v0')
env = Monitor(env=env, directory="./video/cartpole", force=True)

for ep in range(20):
    if ep == 0: env.record_start()
    state = env.reset()
    for t in range(1000):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            if ep == 0: env.record_end()
            break

env.close()
