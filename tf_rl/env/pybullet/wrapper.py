import gym

from tf_rl.common.wrappers import WarpFrame, ScaledFloatFrame
from tf_rl.env.pybullet.env_list import ENVS


class PixelObservationWrapper(gym.ObservationWrapper):
    """ check this post: https://github.com/openai/gym/pull/740#issuecomment-470382987 """

    def __init__(self, env, img_shape=None):
        gym.ObservationWrapper.__init__(self, env)
        self.img_shape = img_shape

    def observation(self, observation):
        img = self.env.render(mode='rgb_array')
        return img if self.img_shape is None else img.image_resize(self.img_shape)

def image_wrapper(env, scale=False, grayscale=False):
    """ Configure environment for raw image observation in MuJoCo """
    env = WarpFrame(env, grayscale=grayscale)
    if scale:
        env = ScaledFloatFrame(env)
    return env

def make_env(env_name="HalfCheetah"):
    if env_name.lower() == "cartpole":
        from pybullet_envs.bullet.cartpole_bullet import CartPoleBulletEnv
        env = CartPoleBulletEnv(renders=False)
    else:
        env = gym.make(ENVS[env_name.lower()])
    env = PixelObservationWrapper(env=env)
    env = image_wrapper(env=env)
    return env
