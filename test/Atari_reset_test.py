from tf_rl.common.wrappers import wrap_deepmind, make_atari, ReplayResetEnv

env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))
env = ReplayResetEnv(env)

state = env.reset()
init_state = env.get_checkpoint_state()

for t in range(1, 1000):
    env.render()
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    state = next_state
    if t % 100 == 0:
        print("done", t)
        env.recover(init_state)
        env.step(0)  # 1 extra step to burn the current state on ALE's RAM is required!!

env.close()
