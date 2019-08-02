import matplotlib.pyplot as plt
import gym, mujoco_py

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
    env.sim.render(width=DEFAULT, height=DEFAULT, mode='offscreen')
    img = env.sim.render(width=DEFAULT, height=DEFAULT, depth=False)
    plt.imshow(img, origin='lower')
    plt.show()
    env.close()
