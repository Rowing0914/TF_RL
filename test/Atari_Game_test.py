"""
Open AI Gym Env Playground

URL: https://gym.openai.com/docs/
Author: Norio Kosaka

"""

from tf_rl.common.wrappers import wrap_deepmind, make_atari
from tf_rl.common.params import ENV_LIST_NATURE, ENV_LIST_NIPS


# for env_name , goal_score in ENV_LIST_NIPS.items():
for env_name , goal_score in ENV_LIST_NATURE.items():
    env = wrap_deepmind(make_atari(env_name))
    state = env.reset()
    for t in range(10):
        # env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        # print(reward, next_state)
        state = next_state
        if done:
            break
    print("{}: Episode finished after {} timesteps".format(env_name, t + 1))
    env.close()
