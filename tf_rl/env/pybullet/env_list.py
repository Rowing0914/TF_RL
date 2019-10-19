# Reference: https://github.com/benelot/pybullet-gym
import pybulletgym  # register PyBullet enviroments with open ai gym

ENVS = {
    # "InvertedPendulum": "InvertedPendulumMuJoCoEnv-v0",  # this didn't work.. need to check more.
    "InvertedDoublePendulum": "InvertedDoublePendulumMuJoCoEnv-v0",
    "reacher3d": "ReacherPyBulletEnv-v0",
    "walker2d": "Walker2DMuJoCoEnv-v0",
    "halfcheetah": "HalfCheetahMuJoCoEnv-v0",
    "ant": "AntMuJoCoEnv-v0",
    "hopper": "HopperMuJoCoEnv-v0",
    "humanoid": "HumanoidMuJoCoEnv-v0",
    "pusher": "PusherPyBulletEnv-v0",
    "thrower": "ThrowerPyBulletEnv-v0",
    "striker": "StrikerPyBulletEnv-v0"
}