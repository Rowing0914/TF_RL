import graph_util.gnn_util as gnn_util
from graph_util.mujoco_parser import parse_mujoco_graph

_input_feat_dim = 64 # default value
node_info = parse_mujoco_graph(task_name="CentipedeFour-v1")
node_info = gnn_util.add_node_info(node_info, _input_feat_dim)

for key, value in node_info.items():
    print(key, value)

for key, value in node_info["adjacency_matrix"].items():
    print(key, value.shape)