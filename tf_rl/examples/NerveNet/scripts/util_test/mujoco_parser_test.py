from graph_util.mujoco_parser import parse_mujoco_graph

res = parse_mujoco_graph(task_name="CentipedeFour-v1")
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
