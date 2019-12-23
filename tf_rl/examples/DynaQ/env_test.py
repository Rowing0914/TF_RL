from tf_rl.examples.Sutton_RL_Intro.libs.envs.grid_world import GridworldEnv
from tf_rl.examples.Sutton_RL_Intro.libs.envs.windy_gridworld import WindyGridworldEnv
from tf_rl.examples.Sutton_RL_Intro.libs.envs.cliff_walking import CliffWalkingEnv

envs = [GridworldEnv, WindyGridworldEnv, CliffWalkingEnv]

for env in envs:
    print("=== Env: {} ===".format(env.__name__))
    env = env()
    observation = env.reset()
    for t in range(2):
        env.render()
        print(env.step(env.action_space.sample()))

    env.close()