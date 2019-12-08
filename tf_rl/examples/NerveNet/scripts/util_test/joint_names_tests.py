import gym
from graph_util.mujoco_parser import get_adjacency_matrix_Ant, get_adjacency_matrix, parse_mujoco_graph, get_all_joint_names

# env_name = "CentipedeFour-v1"
env_name = "AntS-v1"

joint_names = get_all_joint_names(env_name)
env = gym.make(env_name)

for joint_name in joint_names:
    # check more apis
    # https://github.com/openai/mujoco-py/blob/master/mujoco_py/generated/wrappers.pxi
    print(joint_name, env.sim.model.joint_name2id(joint_name))
    # print(joint_name, env.sim.model.body_name2id(joint_name))
    print(joint_name, env.sim.model.get_joint_qpos_addr(joint_name))
    print(joint_name, env.sim.model.get_joint_qvel_addr(joint_name))
print("Num of joints: ", len(joint_names), env.sim.data.cfrc_ext.shape)
