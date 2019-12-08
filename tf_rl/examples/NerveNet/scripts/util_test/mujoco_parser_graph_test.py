"""
this is the test which has been taken from
- gated_graph_policy_network.py
    - GGNN class
        - _parse_mujoco_template function
"""

import graph_util.gnn_util as gnn_util
from graph_util.mujoco_parser import parse_mujoco_graph

_input_feat_dim = 64 # default value

node_info = parse_mujoco_graph(task_name="CentipedeFour-v1")

# step 2: check for ob size for each node type, construct the node dict
node_info = gnn_util.construct_ob_size_dict(node_info, _input_feat_dim)

# step 3: get the inverse node offsets (used to construct gather idx)
node_info = gnn_util.get_inverse_type_offset(node_info, 'node')

# step 4: get the inverse node offsets (used to gather output idx)
node_info = gnn_util.get_inverse_type_offset(node_info, 'output')

# step 5: register existing edge and get the receive and send index
node_info = gnn_util.get_receive_send_idx(node_info)

# step 6: get the stacked node params
node_info = gnn_util.get_stacked_node_params(node_info)

for key, item in node_info.items():
    print("========")
    print(key, item)
