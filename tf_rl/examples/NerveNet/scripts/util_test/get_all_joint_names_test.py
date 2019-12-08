import gym
from graph_util.mujoco_parser import parse_mujoco_graph, get_all_joint_names

env_name = "CentipedeFour-v1"
res = parse_mujoco_graph(task_name=env_name)
# print(res)

"""
dict(tree,
     relation_matrix
     node_type_dict,
     output_type_dict,
     input_dict,
     output_list,
     debug_info,
     node_parameters,
     para_size_dict,
     num_nodes)
"""

for key, item in res.items():
    print("========")
    print(key, item)

joint_names = get_all_joint_names(env_name)
env = gym.make(env_name)

for joint_name in joint_names:
    print(joint_name, "qpos: ", env.sim.model.get_joint_qpos_addr(joint_name))
    print(joint_name, "qvel: ", env.sim.model.get_joint_qvel_addr(joint_name))
print("Num of joints: ", len(joint_names), env.sim.data.cfrc_ext.shape)