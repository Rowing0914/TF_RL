"""
To check the indices of each joint in observation
since the joint name contains even `root`, it is not directly match the actuator indices
"""

import gym
from environments import register

# env_name = "CentipedeFour-v1"
env_name = "AntS-v1"
env = gym.make(env_name)

print(env.sim.data.qpos.shape)
print(env.sim.model.joint_names)
print(env.sim.model.actuator_names)

for joint_name in env.sim.model.actuator_names:
    print()
    # check more apis
    # https://github.com/openai/mujoco-py/blob/master/mujoco_py/generated/wrappers.pxi
    print("Joint name2id       ", joint_name, env.sim.model.joint_name2id(joint_name))
    print("get joint q pos addr", joint_name, env.sim.model.get_joint_qpos_addr(joint_name))
    print("get joint q vel addr", joint_name, env.sim.model.get_joint_qvel_addr(joint_name))
    print("actuator name2id    ", joint_name, env.sim.model.actuator_name2id(joint_name))
    print("get_joint_xanchor   ", joint_name, env.sim.data.get_joint_xanchor(joint_name))
