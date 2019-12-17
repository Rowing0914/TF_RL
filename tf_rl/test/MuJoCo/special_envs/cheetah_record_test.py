# import gym
# from gym_extensions.continuous import mujoco
# from gym.wrappers import Monitor
#
# # available env list: https://github.com/Rowing0914/gym-extensions/blob/mujoco200/tests/all_tests.py
# HalfCheetah_Env_list = [
#     "HalfCheetahGravityHalf-v1",
#     "HalfCheetahGravityThreeQuarters-v1",
#     "HalfCheetahGravityOneAndHalf-v1",
#     "HalfCheetahGravityOneAndQuarter-v1",
#     "HalfCheetahWall-v1",
#     "HalfCheetahWithSensor-v1",
#     "HalfCheetahBigTorso-v1",
#     "HalfCheetahBigThigh-v1",
#     "HalfCheetahBigLeg-v1",
#     "HalfCheetahBigFoot-v1",
#     "HalfCheetahSmallTorso-v1",
#     "HalfCheetahSmallThigh-v1",
#     "HalfCheetahSmallLeg-v1",
#     "HalfCheetahSmallFoot-v1",
#     "HalfCheetahSmallHead-v1",
#     "HalfCheetahBigHead-v1"
# ]
#
# for env_name in HalfCheetah_Env_list:
#     print(env_name)
#     env = gym.make(env_name)
#     env = Monitor(env, "./video/video_{}".format(env_name), force=True)
#
#     n_trial = 10
#     all_rewards = list()
#     env.reset()
#     done = False
#     for _ in range(100):
#         action = env.action_space.sample()
#         s, r, done, info = env.step(action)  # take a random action
#     env.close()
