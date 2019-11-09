import time, gym
from tf_rl.common.wrappers import wrap_deepmind, make_atari, ReplayResetEnv

env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
env = gym.wrappers.Monitor(env, "./video")
env = ReplayResetEnv(env)

state = env.reset()

for t in range(1, 1000):
    env.render()
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    state = next_state

    if t == 300:
        time.sleep(0.5)
        recover_state = env.get_checkpoint_state()

    if (t > 300) and (t % 100 == 0):
        env.recover(recover_state)
        env.step(0)  # 1 extra step to burn the current state on ALE's RAM is required!!
        env.render()
        time.sleep(0.5)

env.close()
